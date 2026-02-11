from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


MEMORY_TYPES = ("tone_style", "do_dont", "long_term_goal")

_VALIDATOR: Draft202012Validator | None = None
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def load_profile_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    schema_path = Path(__file__).resolve().parents[3] / "stuff" / "student_profile.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _VALIDATOR = Draft202012Validator(schema)
    return _VALIDATOR


def normalize_profile(profile: dict[str, Any], *, student_id: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    normalized = deepcopy(profile)

    meta = normalized.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("schema_version", "2.0.0")
    meta.setdefault("profile_id", f"student:{student_id}:{uuid.uuid4()}")
    meta.setdefault("created_at", now)
    meta["updated_at"] = now
    normalized["meta"] = meta

    general = normalized.get("general")
    if not isinstance(general, dict):
        general = {}
    first_name = _as_non_empty_string(general.get("first_name")) or "Student"
    last_name = _as_non_empty_string(general.get("last_name")) or str(student_id)
    email = _as_non_empty_string(general.get("email")) or f"student{student_id}@example.invalid"
    if not _EMAIL_RE.match(email):
        email = f"student{student_id}@example.invalid"
    general["first_name"] = first_name
    general["last_name"] = last_name
    general["email"] = email
    normalized["general"] = general

    context = normalized.get("context")
    if not isinstance(context, dict):
        context = {}
    personalization = context.get("personalization")
    if not isinstance(personalization, dict):
        personalization = {}
    background = context.get("background")
    if not isinstance(background, dict):
        background = {}
    context["personalization"] = personalization
    context["background"] = background
    normalized["context"] = context

    return normalized


def prefill_profile_from_platform(row: dict[str, Any], *, student_id: int) -> dict[str, Any]:
    onboarding = _coerce_json(row.get("user_onboarding_data"))
    if not isinstance(onboarding, dict):
        onboarding = _coerce_json(row.get("funding_template_initial_data"))
    if not isinstance(onboarding, dict):
        metas = row.get("metas")
        if isinstance(metas, dict):
            onboarding = _coerce_json(metas.get("funding_template_initial_data"))
    if not isinstance(onboarding, dict):
        onboarding = {}

    first_name = _as_non_empty_string(row.get("user_first_name"))
    last_name = _as_non_empty_string(row.get("user_last_name"))
    email = _as_non_empty_string(row.get("user_email_address"))
    mobile = _as_non_empty_string(row.get("user_phone_number"))
    date_of_birth = _as_non_empty_string(row.get("user_date_of_birth"))
    gender = _as_non_empty_string(row.get("user_gender"))
    citizenship = _as_non_empty_string(row.get("user_country_of_citizenship"))

    research_interest = _as_non_empty_string(row.get("request_research_interest"))
    onboarding_interest = _as_non_empty_string(onboarding.get("YourInterests"))
    interests = _dedupe_strings([research_interest, onboarding_interest])

    background: dict[str, Any] = {}
    if interests:
        background["research_interests"] = [{"topic": item} for item in interests]

    degree_level = _as_non_empty_string(onboarding.get("YourLastDegree"))
    institution = _as_non_empty_string(onboarding.get("UniversityName"))
    if degree_level and institution:
        degree: dict[str, Any] = {
            "degree_level": degree_level,
            "field": interests[0] if interests else "General",
            "institution": institution,
        }
        gpa = onboarding.get("YourGPA")
        if isinstance(gpa, (int, float)):
            scale = 4.0 if float(gpa) <= 4.0 else 20.0
            degree["gpa"] = {"value": float(gpa), "scale": scale}
        background["degrees"] = [degree]

    targets: dict[str, Any] = {}
    target_level = _as_non_empty_string(onboarding.get("PreferredEducationalLevel"))
    if target_level:
        targets["target_degree_levels"] = [target_level]

    personalization: dict[str, Any] = {}
    preferred_name = _as_non_empty_string(onboarding.get("YourName"))
    if preferred_name:
        personalization["preferred_name"] = preferred_name

    profile = {
        "general": {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "mobile_number": mobile,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "country_of_citizenship": citizenship,
        },
        "context": {
            "personalization": personalization,
            "background": background,
            "targets": targets if targets else None,
        },
    }
    profile = _prune_none(profile)
    return normalize_profile(profile, student_id=student_id)


def validate_profile(profile: dict[str, Any]) -> list[str]:
    validator = load_profile_validator()
    errors = sorted(validator.iter_errors(profile), key=lambda err: list(err.path))
    messages: list[str] = []
    for err in errors:
        location = ".".join(str(part) for part in err.path)
        if location:
            messages.append(f"{location}: {err.message}")
        else:
            messages.append(err.message)
    return messages


def default_required_requirements(intent_type: str) -> list[str]:
    mapping: dict[str, list[str]] = {
        "Funding.Outreach.Email.Generate": ["base_profile_complete", "background_data_complete"],
        "Funding.Outreach.Email.Optimize": ["base_profile_complete", "background_data_complete"],
        "Funding.Outreach.Alignment.Score": ["base_profile_complete", "background_data_complete"],
        "Documents.Review": ["base_profile_complete", "background_data_complete"],
        "Documents.Compose.SOP": [
            "base_profile_complete",
            "background_data_complete",
            "composer_prereqs_complete",
        ],
        "Student.Profile.Collect": ["base_profile_complete", "background_data_complete"],
    }
    return mapping.get(intent_type, ["base_profile_complete"])


def evaluate_requirements(
    profile: dict[str, Any],
    *,
    intent_type: str,
    required_requirements: list[str] | None = None,
) -> dict[str, Any]:
    general = _as_dict(profile.get("general"))
    context = _as_dict(profile.get("context"))
    background = _as_dict(context.get("background"))
    inference = _as_dict(profile.get("inference"))
    sop = _as_dict(inference.get("sop_intelligence"))
    career = _as_dict(sop.get("career_trajectory"))
    research_direction = _as_dict(sop.get("research_direction"))
    long_term = _as_dict(career.get("long_term_goal"))

    status_by_requirement = {
        "base_profile_complete": _base_profile_complete(general),
        "background_data_complete": _background_complete(background),
        "composer_prereqs_complete": (
            _has_value(research_direction.get("content")) and _has_value(long_term.get("content"))
        ),
    }

    requirements = list(required_requirements or [])
    if not requirements:
        requirements = default_required_requirements(intent_type)

    missing_requirements = [
        req for req in requirements if not bool(status_by_requirement.get(req, False))
    ]
    missing_fields_by_requirement = _missing_fields_by_requirement(
        general=general,
        background=background,
        research_direction=research_direction,
        long_term_goal=long_term,
    )
    missing_fields: list[str] = []
    for requirement in missing_requirements:
        for field_path in missing_fields_by_requirement.get(requirement, []):
            if field_path not in missing_fields:
                missing_fields.append(field_path)

    targeted_questions = _targeted_questions(missing_fields_by_requirement, missing_requirements)

    return {
        "intent_type": intent_type,
        "required_requirements": requirements,
        "status_by_requirement": status_by_requirement,
        "missing_requirements": missing_requirements,
        "missing_fields_by_requirement": missing_fields_by_requirement,
        "missing_fields": missing_fields,
        "targeted_questions": targeted_questions,
        "is_satisfied": len(missing_requirements) == 0,
    }


def merge_profile_updates(profile: dict[str, Any], updates: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized_updates = _normalize_update_payload(updates)
    merged = _deep_merge(profile, normalized_updates)
    updated_fields = _collect_paths(normalized_updates)
    return merged, updated_fields


def normalize_memory_entries(entries: Any) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        memory_type = _as_non_empty_string(item.get("type"))
        content = _as_non_empty_string(item.get("content"))
        if not memory_type or memory_type not in MEMORY_TYPES:
            continue
        if not content:
            continue
        source = _as_non_empty_string(item.get("source")) or "user"
        normalized.append({"type": memory_type, "content": content, "source": source})
    return normalized


def group_memory_by_type(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        memory_type = _as_non_empty_string(entry.get("type"))
        content = _as_non_empty_string(entry.get("content"))
        if not memory_type or not content:
            continue
        grouped.setdefault(memory_type, []).append(content)
    return grouped


def _base_profile_complete(general: dict[str, Any]) -> bool:
    first_name = _as_non_empty_string(general.get("first_name"))
    last_name = _as_non_empty_string(general.get("last_name"))
    email = _as_non_empty_string(general.get("email"))
    if not (first_name and last_name and email and _EMAIL_RE.match(email)):
        return False
    if _is_placeholder_name(first_name, last_name):
        return False
    if _is_placeholder_email(email):
        return False
    return True


def _background_complete(background: dict[str, Any]) -> bool:
    keys = (
        "research_interests",
        "degrees",
        "projects",
        "work_experience",
        "publications",
        "skills",
    )
    for key in keys:
        if _has_value(background.get(key)):
            return True
    return False


def _missing_fields_by_requirement(
    *,
    general: dict[str, Any],
    background: dict[str, Any],
    research_direction: dict[str, Any],
    long_term_goal: dict[str, Any],
) -> dict[str, list[str]]:
    missing_general: list[str] = []
    first_name = _as_non_empty_string(general.get("first_name"))
    last_name = _as_non_empty_string(general.get("last_name"))
    email = _as_non_empty_string(general.get("email"))

    if not first_name or _is_placeholder_name(first_name, last_name or ""):
        missing_general.append("general.first_name")
    if not last_name or _is_placeholder_name(first_name or "", last_name):
        missing_general.append("general.last_name")
    if not email or not _EMAIL_RE.match(email) or _is_placeholder_email(email):
        missing_general.append("general.email")

    missing_background: list[str] = []
    if not _background_complete(background):
        missing_background.append("context.background.research_interests")

    missing_composer: list[str] = []
    if not _has_value(research_direction.get("content")):
        missing_composer.append("inference.sop_intelligence.research_direction.content")
    if not _has_value(long_term_goal.get("content")):
        missing_composer.append("inference.sop_intelligence.career_trajectory.long_term_goal.content")

    return {
        "base_profile_complete": missing_general,
        "background_data_complete": missing_background,
        "composer_prereqs_complete": missing_composer,
    }


def _targeted_questions(
    missing_fields_by_requirement: dict[str, list[str]],
    missing_requirements: list[str],
) -> list[dict[str, str]]:
    question_by_field = {
        "general.first_name": "What is your first name as you want it shown in outreach emails?",
        "general.last_name": "What is your last name as you want it shown in outreach emails?",
        "general.email": "Which email address should the copilot use for your outreach drafts?",
        "context.background.research_interests": "What is your main research interest right now?",
        "inference.sop_intelligence.research_direction.content": "What research direction should your SOP emphasize?",
        "inference.sop_intelligence.career_trajectory.long_term_goal.content": "What is your long-term academic or career goal?",
    }
    questions: list[dict[str, str]] = []
    for requirement in missing_requirements:
        for field_path in missing_fields_by_requirement.get(requirement, []):
            question = question_by_field.get(field_path)
            if not question:
                continue
            questions.append(
                {
                    "requirement": requirement,
                    "field_path": field_path,
                    "question": question,
                }
            )
            if len(questions) >= 3:
                return questions
    return questions


def _normalize_update_payload(updates: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(updates)
    out: dict[str, Any] = {}
    for key, value in normalized.items():
        if "." in key:
            _set_path(out, key, value)
            continue
        if key in {"first_name", "last_name", "email", "mobile_number", "date_of_birth", "gender"}:
            _set_path(out, f"general.{key}", value)
            continue
        if key == "research_interest" and _has_value(value):
            _set_path(out, "context.background.research_interests", [{"topic": str(value)}])
            continue
        out[key] = value
    return out


def _collect_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    paths: list[str] = []
    for key, item in value.items():
        current = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            nested = _collect_paths(item, current)
            if nested:
                paths.extend(nested)
            else:
                paths.append(current)
        else:
            paths.append(current)
    deduped: list[str] = []
    for path in paths:
        if path and path not in deduped:
            deduped.append(path)
    return deduped


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    current = target
    parts = [part for part in path.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _dedupe_strings(values: list[str | None]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = _as_non_empty_string(value)
        if not text:
            continue
        if text not in deduped:
            deduped.append(text)
    return deduped


def _as_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _is_placeholder_email(email: str) -> bool:
    lowered = email.strip().lower()
    return lowered.endswith("@example.invalid") and lowered.startswith("student")


def _is_placeholder_name(first_name: str, last_name: str) -> bool:
    return first_name.strip().lower() == "student" and last_name.strip().isdigit()


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            pruned = _prune_none(item)
            if pruned is None:
                continue
            out[key] = pruned
        return out
    if isinstance(value, list):
        out_list = [_prune_none(item) for item in value]
        return [item for item in out_list if item is not None]
    return value
