# src/outreach/gmail.py

from __future__ import annotations

import asyncio, base64, json, posixpath, io
from src.config import settings
from datetime import datetime, UTC, timedelta
from ftplib import FTP, error_perm
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr, parseaddr
from email.mime.text import MIMEText
from pathlib import Path
from typing import Mapping, Optional, Tuple
import functools

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.db.queries import (
    SELECT_GMAIL_TOKEN,
    UPDATE_GMAIL_TOKEN
)
from src.db.session import DB
from src.tools.logger import Logger
from src.tools.markdown import markdown_to_html


_LOG, _ = Logger().create(application="gmail")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

FTP_HOST = settings.FTP_HOST
FTP_PORT = settings.FTP_PORT
FTP_USERNAME = settings.FTP_USERNAME
FTP_PASSWORD = settings.FTP_PASSWORD
FTP_ROOT = settings.FTP_ROOT.rstrip("/")

async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

async def _load_creds(student_id: int) -> Credentials:
    row = await DB.fetch_one(SELECT_GMAIL_TOKEN, (student_id,))
    if not row:
        raise RuntimeError(f"No Gmail credentials stored for student_id={student_id}")
    token_json = json.loads(row["token_blob"])
    creds = Credentials.from_authorized_user_info(token_json, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        await _to_thread(creds.refresh, Request())
        await DB.execute(
            UPDATE_GMAIL_TOKEN,
            (creds.to_json(), student_id),
        )

    if not creds.valid:
        raise RuntimeError("Gmail credentials invalid and no refresh possible")

    return creds

async def _get_service(student_id: int):
    creds = await _load_creds(student_id)
    return await _to_thread(
        functools.partial(
            build,
            "gmail",
            "v1",
            credentials=creds,
            cache_discovery=False,
        )
    )

async def get_service(student_id: int):
    """
    Expose Gmail service creation so callers can cache per user.
    """
    return await _get_service(student_id)

def _header_value(msg: dict, name: str) -> Optional[str]:
    hdrs = msg.get("payload", {}).get("headers", [])
    for h in hdrs:
        if h.get("name", "").lower() == name.lower():
            return h.get("value")
    return None

def _clean_email(raw: str | None) -> str:
    if not raw:
        return ""
    _, addr = parseaddr(raw)
    return addr.lower()

def _extract_body(part: Mapping) -> str | None:
    if "parts" in part:
        for sub in part["parts"]:
            txt = _extract_body(sub)
            if txt:
                return txt
    elif part["mimeType"] in ("text/html", "text/plain"):
        data = part["body"].get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8")
    return None

def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

async def get_thread_reply_context(
    student_id: int,
    thread_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not thread_id:
        return None, None

    svc = await _get_service(student_id)
    thread = await _to_thread(lambda: svc.users().threads().get(
        userId="me", id=thread_id, format="full").execute())

    messages = thread.get("messages", [])
    if not messages:
        return None, None

    last = max(messages, key=lambda m: int(m.get("internalDate", "0")))
    msg_id = _header_value(last, "Message-Id") or _header_value(last, "Message-ID")
    subj = _header_value(last, "Subject")
    return msg_id, subj


async def _search_recent_from_sender(
    student_id: int,
    from_email: str,
    after_dt: datetime,
    *,
    svc=None,
    max_results: int = 50,
) -> tuple[bool, Optional[datetime], Optional[str], Optional[str]]:
    svc = svc or await _get_service(student_id)

    after_ts = int(_as_utc(after_dt).timestamp())
    q = f'from:{from_email} after:{after_ts}'

    resp = await _to_thread(lambda: svc.users().messages().list(
        userId="me", q=q, maxResults=max_results).execute())
    for item in (resp.get("messages") or []):
        m = await _to_thread(lambda mid=item["id"]:
            svc.users().messages().get(userId="me", id=mid, format="full").execute())

        labels = set(m.get("labelIds", []))
        if "SENT" in labels:
            continue

        internal_ms = int(m.get("internalDate", "0"))
        msg_dt = datetime.fromtimestamp(internal_ms / 1000.0, tz=UTC)
        if msg_dt <= after_dt:
            continue

        body = _extract_body(m.get("payload", {})) or ""
        reply_only = "\n".join([
            l for l in body.split("\n")
            if not l.startswith(">")
               and l not in ['\r', '', 'wrote:\r']
        ]).strip()

        return True, msg_dt, reply_only, m.get("threadId")

    return False, None, None, None

async def fetch_recent_inbox_messages(
    student_id: int,
    after_dt: datetime,
    *,
    svc=None,
    max_results: int = 200,
) -> list[dict]:
    """
    Pull inbound messages after a timestamp so callers can batch reply checks per user.
    """
    svc = svc or await _get_service(student_id)

    after_dt = _as_utc(after_dt) or datetime.now(UTC) - timedelta(days=90)
    after_ts = int(after_dt.timestamp())
    q = f"after:{after_ts} -in:sent -in:drafts"

    msg_ids: list[str] = []
    page_token: str | None = None
    while True:
        remaining = max_results - len(msg_ids)
        if remaining <= 0:
            break
        resp = await _to_thread(lambda: svc.users().messages().list(
            userId="me",
            q=q,
            maxResults=min(200, remaining),
            pageToken=page_token
        ).execute())
        msg_ids.extend([m["id"] for m in resp.get("messages", [])])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    messages: list[dict] = []
    for mid in msg_ids:
        m = await _to_thread(lambda mid=mid:
            svc.users().messages().get(userId="me", id=mid, format="full").execute())

        labels = set(m.get("labelIds", []))
        if "SENT" in labels:
            continue

        internal_ms = int(m.get("internalDate", "0"))
        msg_dt = datetime.fromtimestamp(internal_ms / 1000.0, tz=UTC)
        if msg_dt <= after_dt:
            continue

        from_email = _clean_email(_header_value(m, "From"))
        body = _extract_body(m.get("payload", {})) or ""
        reply_only = "\n".join([
            l for l in body.split("\n")
            if not l.startswith(">")
               and l not in ['\r', '', 'wrote:\r']
        ]).strip()

        messages.append({
            "id": m.get("id"),
            "thread_id": m.get("threadId"),
            "from_email": from_email,
            "date": msg_dt,
            "body": reply_only,
        })

    messages.sort(key=lambda m: m["date"])
    return messages

def _build_raw(
    receiver_address: str,
    sender_name: str | None,
    email_subject: str,
    email_body: str,
    attachments: Mapping[str, dict] | None,
    *,
    sender_address: str,
    reply_headers: Mapping[str, str] | None = None,
) -> str:
    outer = MIMEMultipart("mixed")
    receiver_address = receiver_address.lower()
    outer["to"], outer["subject"] = receiver_address, email_subject

    if reply_headers:
        for k, v in reply_headers.items():
            if v:
                outer[k] = v

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(email_body, "plain", "utf-8"))
    html_part = markdown_to_html(email_body)
    alt.attach(MIMEText(html_part, "html", "utf-8"))
    outer.attach(alt)

    display = sender_name or sender_address.split("@", 1)[0]
    outer["From"] = formataddr((display, sender_address))

    if attachments:
        ftp = FTP()
        ftp.connect(FTP_HOST, FTP_PORT, timeout=15)
        ftp.login(FTP_USERNAME, FTP_PASSWORD)

        try:
            for label, meta in attachments.items():
                if not meta or "path" not in meta:
                    continue  # skip None or bad entry

                rel_path = meta["path"].lstrip("/\\")  # keep sub-dirs
                ftp_path = posixpath.join(FTP_ROOT, rel_path)
                ext = Path(rel_path).suffix
                base = f"{sender_name}_{label.upper()}" if sender_name else label.upper()
                filename = f"{base}{ext}"

                buf = io.BytesIO()
                try:
                    ftp.retrbinary(f"RETR {ftp_path}", buf.write)
                except error_perm as exc:
                    _LOG.error("FTP RETR failed for %s – %s", ftp_path, exc)
                    continue

                buf.seek(0)
                subtype = ext.lstrip(".") or "octet-stream"
                part = MIMEApplication(buf.read(), _subtype=subtype)
                part.add_header("Content-Disposition", "attachment", filename=filename)
                outer.attach(part)
        except Exception as exc:
            _LOG.error("Error while fetching attachments from FTP: %s", exc)
            raise RuntimeError(f"Failed to fetch attachments: {exc}") from exc
        finally:
            ftp.quit()

    return base64.urlsafe_b64encode(outer.as_bytes()).decode()

async def send_email(
    student_id: int,
    *,
    professor_email: str,
    student_name: str | None = None,
    email_subject: str,
    email_body: str,
    attachments: Mapping[str, dict] | None,
    thread_id: str | None = None,
) -> tuple[str, str]:
    svc = await _get_service(student_id)

    profile = await _to_thread(lambda: svc.users().getProfile(userId="me").execute())
    from_email = profile.get("emailAddress") or "me"

    reply_headers = {}

    if thread_id:
        last_msg_id, last_subj = await get_thread_reply_context(student_id, thread_id)

        if last_msg_id:
            reply_headers["In-Reply-To"] = last_msg_id
            reply_headers["References"] = last_msg_id

        if last_subj:
            low = (email_subject or "").lower()
            if not low.startswith("re:"):
                email_subject = f"Re: {last_subj}"

    raw = _build_raw(
        receiver_address=professor_email,
        sender_name=student_name,
        email_subject=email_subject,
        email_body=email_body,
        attachments=attachments,
        sender_address=from_email,
        reply_headers=reply_headers or None,
    )

    body = {"raw": raw}
    if thread_id:
        body["threadId"] = thread_id

    try:
        if not settings.DEBUG_MODE:
            sent = await _to_thread(
                lambda: svc.users().messages().send(userId="me", body=body).execute()
            )
        else:
            sent = {
                "id": "mocked_message_id_12345",
                "threadId": thread_id or "mocked_thread_id_67890"
            }
            _LOG.info("DEBUG MODE: Email send skipped, mocked response used.")

        return sent["id"], sent["threadId"]
    except HttpError as exc:
        _LOG.error("Gmail API error: %s", exc)
        raise
