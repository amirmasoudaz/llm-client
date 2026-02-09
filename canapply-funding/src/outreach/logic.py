# src/outreach/logic.py

import asyncio, json
from src.config import settings
from datetime import datetime, timedelta, UTC
from collections import defaultdict
from typing import Dict, Mapping
from pathlib import Path

from src.db.session import DB
from src.db.queries import (
    SELECT_PENDING_FOR_CHECK,
    SELECT_REVIEW_INPUTS,
    SELECT_SEND_INPUTS,
    UPDATE_REQUEST,
    UPSERT_DRAFT,
    MARK_PROFESSOR_REPLIED,
    MARK_REMINDER_ONE_SENT,
    MARK_REMINDER_TWO_SENT,
    MARK_REMINDER_THREE_SENT,
    MARK_REPLY_CHECKED,
    MARK_NO_MORE_CHECKS,
    MARK_SENT,
)
from src.outreach import gmail as g
from src.agents.detailed_generation.agent import DetailedEmailGenerationAgent
from src.agents.email_paraphrasing.agent import ParaphrasingAgent
from src.agents.reply_digestion.agent import ReplyDigestionAgent
from src.tools.helpers import ext_student_name_subject
from src.tools.logger import Logger


_LOG, _ = Logger().create(application="logic")

def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class OutreachLogic:
    cpd = Path(__file__).parent
    cache_dir = cpd / "cache"

    _cron_lock = asyncio.Lock()

    def __init__(self):
        self.detailed_generation_agent = DetailedEmailGenerationAgent()
        self.paraphrasing_agent = ParaphrasingAgent()
        self.reply_digestion_agent = ReplyDigestionAgent()

    @staticmethod
    async def _get_template(student_template_ids: str, draft_type: str) -> tuple[str, str]:
        student_template_ids = json.loads(student_template_ids)
        if not student_template_ids:
            return "", ""

        student_template_id = student_template_ids[draft_type]
        if not student_template_id:
            return "", ""

        query = f"SELECT formatted_content, formatted_subject FROM funding_student_templates WHERE id = %s"
        student_template = await DB.fetch_one(query, (student_template_id,))
        if not student_template:
            return "", ""

        template_body = student_template["formatted_content"]
        template_subject = student_template["formatted_subject"]

        return template_body, template_subject

    @staticmethod
    def _index_incoming_messages(messages: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        by_thread: dict[str, list[dict]] = defaultdict(list)
        by_sender: dict[str, list[dict]] = defaultdict(list)
        for msg in messages:
            if msg.get("thread_id"):
                by_thread[msg["thread_id"]].append(msg)
            if msg.get("from_email"):
                by_sender[msg["from_email"]].append(msg)
        for bucket in by_thread.values():
            bucket.sort(key=lambda m: m["date"])
        for bucket in by_sender.values():
            bucket.sort(key=lambda m: m["date"])
        return by_thread, by_sender

    @staticmethod
    def _find_reply_for_email(
            professor_email: str,
            gmail_thread_id: str,
            last_contact: datetime,
            by_thread: Mapping[str, list[dict]],
            by_sender: Mapping[str, list[dict]]
    ) -> dict | None:
        prof_email = (professor_email or "").lower()
        thr_id = gmail_thread_id

        candidates: list[dict] = []
        if thr_id and thr_id in by_thread:
            candidates.extend(by_thread[thr_id])
        if prof_email and prof_email in by_sender:
            candidates.extend(by_sender[prof_email])

        if not candidates:
            return None

        recent = [
            c for c in candidates
            if c.get("date") and c["date"] > _as_utc(last_contact)
        ]
        if not recent:
            return None
        return min(recent, key=lambda m: m["date"])

    async def _compose_main_email(
            self,
            funding_request_id: int,
            research_interest: str,
            paper_title: str,
            paper_journal: str,
            paper_year: int,
            research_connection: str,
            professor_last_name: str,
            match_status: str,
            template_body: str,
            template_subject: str,
    ) -> tuple[str, str]:
        if match_status == "high":
            body = await self.detailed_generation_agent.generate(
                funding_request_id=funding_request_id,
                research_interest=research_interest,
                paper_title=paper_title,
                paper_journal=paper_journal,
                paper_year=paper_year,
                research_connection=research_connection,
                professor_last_name=professor_last_name,
                template_body=template_body,
                regen=True
            )
        elif match_status == "medium":
            body = template_body.replace("{{ProfessorName}}", professor_last_name)
            body = body.replace("{{ProfessorInterests}}", research_interest)
        else:
            body = template_body.replace("{{ProfessorName}}", professor_last_name)

        subject = " - ".join(template_subject.split(" - ")[:2])

        if match_status in ["medium", "high"]:
            body = await self.paraphrasing_agent.paraphrase(
                content=f"Raw Email:\n\n\n\n{body}",
                identifier=f"paraphrase_funding_{funding_request_id}",
                regen=True,
            )

        return subject, body

    @staticmethod
    async def _maybe_mark_no_more_checks(row: Mapping) -> None:
        if not settings.REMINDERS_ON:
            return
        last_rem = row.get("reminder_three_sent_at") or row.get("reminder_two_sent_at") or row.get(
            "reminder_one_sent_at")
        if not last_rem:
            return
        if datetime.now(UTC) - _as_utc(last_rem) >= timedelta(days=90):
            await DB.execute(MARK_NO_MORE_CHECKS, (datetime.now(UTC), row["id"]))
            _LOG.info("Marked no_more_checks for funding_email_id=%s", row["id"])

    async def _send_reminder(
            self,
            funding_email_id: int,
            student_id: int,
            professor_name: str,
            professor_email: str,
            student_template_ids: str,
            main_email_subject: str,
            gmail_thread_id: str,
            reminder_number: int,
    ):
        if not settings.REMINDERS_ON:
            _LOG.info("Reminders disabled, skipping reminder %d for funding_email_id=%s", reminder_number,
                      funding_email_id)
            return None

        template_body, template_subject = await self._get_template(
            student_template_ids=student_template_ids,
            draft_type=f"reminder{reminder_number}"
        )
        if not template_body:
            _LOG.warning(
                "No reminder template found for student_id=%s, skipping reminder %d for funding_email_id=%s",
                student_id, reminder_number, funding_email_id)
            return None

        professor_name = professor_name or "Professor"
        email_subject = " - ".join(template_subject.split(" - ")[:2])
        email_body_raw = template_body.replace("{{ProfessorName}}", professor_name)
        email_body = await self.paraphrasing_agent.paraphrase(
            content=f"Professor Name Is: {professor_name}\n\nRaw Email:\n\n\n\n{email_body_raw}",
            identifier=f"paraphrase_reminder_{funding_email_id}_{reminder_number}",
            regen=True,
        )

        if not email_subject or not email_body:
            return None

        student_name = ext_student_name_subject(main_email_subject)
        msg_id, thr_id = await g.send_email(
            student_id=student_id,
            professor_email=professor_email,
            student_name=student_name,
            email_subject=email_subject,
            email_body=email_body,
            attachments=None,
            thread_id=gmail_thread_id,
        )

        mark_reminder_query = {
            1: MARK_REMINDER_ONE_SENT,
            2: MARK_REMINDER_TWO_SENT,
            3: MARK_REMINDER_THREE_SENT,
        }[reminder_number]
        now = datetime.now(UTC)
        await DB.execute(mark_reminder_query, (email_subject, email_body, now, now, funding_email_id))
        _LOG.info("Sent reminder %d for funding_email_id=%s (gmail_id=%s)", reminder_number, funding_email_id, msg_id)

        return None

    @staticmethod
    def _last_contact_at(row: Mapping) -> datetime:
        def pick_contact(row: Mapping, sent_key: str, at_key: str):
            return row[at_key] if (bool(row[sent_key]) and row[at_key]) else None

        contacts = []

        main = _as_utc(row.get("main_sent_at"))
        if main:
            contacts.append(main)

        contacts.append(pick_contact(row, "reminder_one_sent", "reminder_one_sent_at"))
        contacts.append(pick_contact(row, "reminder_two_sent", "reminder_two_sent_at"))
        contacts.append(pick_contact(row, "reminder_three_sent", "reminder_three_sent_at"))

        contacts = [_as_utc(d) for d in contacts if d]
        if contacts:
            return max(contacts)

        created = _as_utc(row.get("created_at"))
        return created or datetime.now(UTC) - timedelta(days=30)

    @staticmethod
    def _compute_next_reply_check_at(last_contact) -> datetime:
        now = datetime.now(UTC)
        age = now - last_contact
        if age <= timedelta(days=1):
            interval = timedelta(hours=2)
        elif age <= timedelta(days=7):
            interval = timedelta(hours=6)
        elif age <= timedelta(days=30):
            interval = timedelta(hours=12)
        else:
            interval = timedelta(days=3)
        return now + interval

    async def _check_cycle(self) -> None:
        reminders_enabled = settings.REMINDERS_ON

        funding_emails_to_check = await DB.fetch_all(SELECT_PENDING_FOR_CHECK)
        if not funding_emails_to_check:
            _LOG.info("No pending records for Reminders Check Cycle")
            return None

        _LOG.info(f"Starting Reminders Check Cycle with {len(funding_emails_to_check)} pending records")

        funding_emails_to_check_by_student: dict[int, list[Mapping]] = defaultdict(list)
        for record in funding_emails_to_check:
            funding_emails_to_check_by_student[record["student_id"]].append(record)

        for student_id, funding_emails_records in funding_emails_to_check_by_student.items():
            try:
                svc = await g.get_service(student_id)
                earliest_check = min(
                    _as_utc(r.get("last_reply_check_at"))
                    or _as_utc(r.get("main_sent_at"))
                    or datetime.now(UTC) - timedelta(days=30)
                    for r in funding_emails_records
                )
                inbox_msgs = await g.fetch_recent_inbox_messages(
                    student_id,
                    earliest_check,
                    svc=svc,
                    max_results=1000,
                )
                by_thread, by_sender = self._index_incoming_messages(inbox_msgs)

                _LOG.info(f"Reminders for Student ID: {student_id} with {len(funding_emails_records)} records and {len(inbox_msgs)} inbox messages")
                for record in funding_emails_records:
                    funding_email_id = record["funding_email_id"]
                    _LOG.info(f"Reminders - Processing funding_email_id: {record['funding_request_id']}")

                    try:
                        last_contact = self._last_contact_at(record)
                        reply_msg = self._find_reply_for_email(record, last_contact, by_thread, by_sender)
                        if reply_msg:
                            digested = await self.reply_digestion_agent.digest(
                                funding_request_id=record['funding_request_id'],
                                main_email_body=record["main_email_body"],
                                main_email_subject=record["main_email_subject"],
                                professor_reply_body=reply_msg["body"],
                                professor_name=record.get("professor_name") or "Professor",
                                regen=True
                            )
                            await self.reply_digestion_agent.upsert_digested(digested)
                            
                            is_auto = digested.get("is_auto_generated", False)
                            engagement = digested.get("engagement_label", "")
                            is_auto_engagement = engagement == "AUTO_REPLY_OR_OUT_OF_OFFICE"
                            
                            if is_auto or is_auto_engagement:
                                _LOG.info(
                                    "Auto-reply detected for funding_email_id=%s (type=%s, engagement=%s) - continuing reminder cycle",
                                    funding_email_id,
                                    digested.get("auto_generated_type", "unknown"),
                                    engagement,
                                )
                            else:
                                await DB.execute(
                                    MARK_PROFESSOR_REPLIED,
                                    (reply_msg["date"], reply_msg["body"], datetime.now(UTC), datetime.now(UTC), funding_email_id),
                                )
                                _LOG.info(
                                    "Professor replied for funding_email_id=%s at %s (msg_id=%s, engagement=%s)",
                                    funding_email_id,
                                    reply_msg["date"],
                                    reply_msg.get("id"),
                                    engagement,
                                )
                                continue

                        # If reminders are disabled, skip sending reminders
                        if not reminders_enabled:
                            continue
                        if row.get("no_more_reminders", 0):
                            continue

                        row = dict(row)  # make mutable copy
                        if not row.get("reminder_one_sent", 0) and _as_utc(row.get("main_sent_at")):
                            m_at = _as_utc(row.get("main_sent_at"))
                            if datetime.now(UTC) - m_at >= timedelta(days=5):
                                await self._send_reminder(row, 1)
                                row["reminder_one_sent"] = 1
                                row["reminder_one_sent_at"] = datetime.now(UTC)
                        elif row.get("reminder_one_sent", 0) and not row.get("reminder_two_sent", 0):
                            r1_at = _as_utc(row.get("reminder_one_sent_at"))
                            if r1_at and (datetime.now(UTC) - r1_at >= timedelta(days=10)):
                                await self._send_reminder(row, 2)
                                row["reminder_two_sent"] = 1
                                row["reminder_two_sent_at"] = datetime.now(UTC)
                        elif row.get("reminder_two_sent", 0) and not row.get("reminder_three_sent", 0):
                            r2_at = _as_utc(row.get("reminder_two_sent_at"))
                            if r2_at and (datetime.now(UTC) - r2_at >= timedelta(days=15)):
                                await self._send_reminder(row, 3)
                                row["reminder_three_sent"] = 1
                                row["no_more_reminders"] = 1
                                row["reminder_three_sent_at"] = datetime.now(UTC)

                        if row.get("reminder_three_sent", 0) or row.get("no_more_reminders", 0):
                            await self._maybe_mark_no_more_checks(row)

                        next_check_at = self._compute_next_reply_check_at(last_contact)
                        await DB.execute(
                            MARK_REPLY_CHECKED,
                            (datetime.now(UTC), next_check_at, datetime.now(UTC), funding_email_id),
                        )
                    except Exception as exc:
                        _LOG.exception(
                            "Error processing reminder/reply check for funding_email_id=%s: %s",
                            funding_email_id,
                            exc,
                        )
                        continue
            except Exception as exc:
                _LOG.exception("Error processing reminder cycle for student_id=%s: %s", student_id, exc)

        _LOG.info("Reminders Check Cycle Completed")
        return None

    async def start_reminder_cron(self):
        reminders_enabled = settings.REMINDERS_ON

        if not reminders_enabled:
            _LOG.info("Reminders disabled (REMINDERS_ON is not True), reminder cron not started")
            return

        interval_minutes = settings.CHECK_INTERVAL_MINUTES
        if not interval_minutes:
            raise ValueError("CHECK_INTERVAL_MINUTES environment variable not set")

        interval = max(1, int(interval_minutes)) * 60

        _LOG.info("Reminder cron started; interval=%s minutes", interval_minutes)
        while True:
            try:
                async with self._cron_lock:
                    await self._check_cycle()
            except Exception as exc:
                _LOG.exception("Reminder cron cycle failed: %s", exc)
            await asyncio.sleep(interval)

    async def get_review(self, *, funding_request_id: int) -> Dict:
        data = await DB.fetch_one(SELECT_REVIEW_INPUTS, (funding_request_id,))
        if not data:
            return {"status": False, "error": "funding_request_id not found"}

        student_id = data["student_id"]
        match_status = data["match_status"]
        research_interest = data["research_interest"]
        paper_title = data["paper_title"]
        paper_journal = data["paper_journal"]
        paper_year = data["paper_year"]
        research_connection = data["research_connection"]
        professor_last_name = data["professor_last_name"] or "Professor"
        professor_full_name = data["professor_full_name"] or "Professor"
        professor_email = data["professor_email"]
        email_attachments = data["email_attachments"] or "{}"
        student_template_ids = data["student_template_ids"]

        template = await self._get_template(
            student_template_ids=student_template_ids,
            draft_type="main"
        )
        template_body = template["template"]
        template_subject = template["subject"]

        match_status = {0:  "low", 1: "medium", 2: "high"}.get(match_status)
        assert match_status, "Invalid match_status value"

        email_subject, email_body = await self._compose_main_email(
            funding_request_id=funding_request_id,
            research_interest=research_interest,
            paper_title=paper_title,
            paper_journal=paper_journal,
            paper_year=paper_year,
            research_connection=research_connection,
            professor_last_name=professor_last_name,
            match_status=match_status,
            template_body=template_body,
            template_subject=template_subject
        )
        now = datetime.now(UTC)

        await DB.execute(
            UPSERT_DRAFT,
            (
                funding_request_id,
                student_id,
                professor_email,
                professor_full_name,
                email_subject,
                email_body,
                email_attachments,
                now,
                now,
            ),
        )

        await DB.execute(
            UPDATE_REQUEST,
            (
                email_subject,
                email_body,
                now,
                funding_request_id,
            ),
        )

        _LOG.info("Draft stored for funding_request_id=%d", funding_request_id)
        return {
            "status": True,
            "subject": email_subject,
            "preview": email_body,
        }

    @staticmethod
    def parse_attachments(attachments_blob: str | dict) -> dict[str, dict[str, str]]:
        try:
            return json.loads(json.loads(attachments_blob))
        except Exception as e:
            try:
                return json.loads(attachments_blob)
            except Exception as ee:
                _LOG.exception("Second attempt to load attachments also failed - First: %s; Second: %s", e, ee)
                return {}

    async def get_send(self, funding_request_id: int) -> Dict:
        req = await DB.fetch_one(SELECT_SEND_INPUTS, (funding_request_id,))

        if not req:
            return {"status": False, "error": "Request Info Not Found - run /review first"}

        funding_email_id = req["funding_email_id"]
        main_email_sent = req["main_email_sent"]
        professor_email = req["professor_email"]
        email_subject = req["email_subject"]
        email_body = req["email_body"]
        attachments = req["email_attachments"]
        student_id = req["student_id"]

        if main_email_sent and not settings.RESEND_ALLOWED:
            return {
                "status": False,
                "error": "Already sent"
            }

        try:
            student_name = ext_student_name_subject(email_subject)
            attachments = self.parse_attachments(attachments)

            msg_id, thr_id = await g.send_email(
                student_id=student_id,
                professor_email=professor_email,
                student_name=student_name,
                email_subject=email_subject,
                email_body=email_body,
                attachments=attachments
            )
        except Exception as exc:
            _LOG.exception("Gmail send failed for funding_request_id=%d: %s", funding_request_id, exc)
            return {"status": False, "error": str(exc)}

        now = datetime.now(UTC)
        await DB.execute(
            MARK_SENT,
            (
                msg_id,
                thr_id,
                now,
                now,
                email_subject,
                email_body,
                funding_email_id
            ),
        )

        _LOG.info("E-mail sent for funding_request_id=%d; thr_id=%s; msg_id=%s", funding_request_id, thr_id, msg_id)
        return {"status": True, "gmail_msg_id": msg_id, "gmail_thread_id": thr_id}


if __name__ == "__main__":
    logic = OutreachLogic()
