from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import posixpath
import re
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

from jsonschema import Draft202012Validator, RefResolver


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_object(value: Any) -> Dict[str, Any]:
    return {"alg": "sha256", "value": sha256_hex(stable_json_dumps(value))}


def dotted_get(obj: Any, path: str) -> Any:
    if path == "":
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        return None
    return cur


def dotted_set(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Dict[str, Any] = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def json_pointer_get(doc: Any, pointer: str) -> Any:
    # pointer is either "" or starts with "/"
    if pointer in ("", "#", None):
        return doc
    if pointer.startswith("#"):
        pointer = pointer[1:]
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"Unsupported JSON pointer: {pointer}")
    cur = doc
    for raw in pointer.split("/")[1:]:
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"JSON pointer segment not found: {part} in {pointer}")
    return cur


class SchemaRegistry:
    """
    Loads repo schemas from ./schemas/**.json and validates instances.

    Notes:
    - The repo uses repo-relative `$id` values (e.g. "schemas/intents/...").
    - To keep this demo simple and reliable, we **dereference** `$ref`s into an
      inlined schema before validating with jsonschema.
    """

    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self._schemas_by_rel_id: Dict[str, Dict[str, Any]] = {}
        self._canonical_id_by_rel_id: Dict[str, str] = {}
        self._schemas_by_canonical_id: Dict[str, Dict[str, Any]] = {}
        self._load_all()
        self._build_canonical_store()

    def _load_all(self) -> None:
        schemas_dir = os.path.join(self._repo_root, "schemas")
        for dirpath, _, filenames in os.walk(schemas_dir):
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                schema_id = data.get("$id")
                if not schema_id:
                    raise ValueError(f"Schema missing $id: {path}")
                self._schemas_by_rel_id[schema_id] = data
                self._canonical_id_by_rel_id[schema_id] = f"https://canapply.local/{schema_id}"

    def _build_canonical_store(self) -> None:
        for rel_id, schema in self._schemas_by_rel_id.items():
            canonical_id = self._canonical_id_by_rel_id[rel_id]
            normalized = self._normalize_schema(schema, current_rel_id=rel_id)
            normalized["$id"] = canonical_id
            self._schemas_by_canonical_id[canonical_id] = normalized

    def _normalize_schema(self, schema: Any, *, current_rel_id: str) -> Any:
        if isinstance(schema, list):
            return [self._normalize_schema(x, current_rel_id=current_rel_id) for x in schema]
        if not isinstance(schema, dict):
            return schema

        out: Dict[str, Any] = {}
        for k, v in schema.items():
            if k == "$ref" and isinstance(v, str):
                out[k] = self._canonicalize_ref(current_rel_id, v)
            elif k == "$id" and isinstance(v, str):
                # Normalize nested $id to a stable canonical URI as well.
                if v in self._canonical_id_by_rel_id:
                    out[k] = self._canonical_id_by_rel_id[v]
                else:
                    out[k] = v
            else:
                out[k] = self._normalize_schema(v, current_rel_id=current_rel_id)
        return out

    def _canonicalize_ref(self, current_rel_id: str, ref: str) -> str:
        # Internal reference
        if ref.startswith("#"):
            return f"{self._canonical_id_by_rel_id[current_rel_id]}{ref}"

        if "#" in ref:
            path_part, frag = ref.split("#", 1)
            frag = "#" + frag
        else:
            path_part, frag = ref, ""

        target_rel = posixpath.normpath(posixpath.join(posixpath.dirname(current_rel_id), path_part))
        target_canonical = self._canonical_id_by_rel_id.get(target_rel)
        if not target_canonical:
            # If the ref is external (http/https), keep it.
            return ref
        return f"{target_canonical}{frag}"

    def validate(self, instance: Any, schema_id: str) -> None:
        canonical_id = self._canonical_id_by_rel_id[schema_id]
        schema = self._schemas_by_canonical_id[canonical_id]
        resolver = RefResolver(base_uri=canonical_id, referrer=schema, store=self._schemas_by_canonical_id)
        Draft202012Validator(schema, resolver=resolver).validate(instance)


@dataclasses.dataclass
class OperatorManifest:
    name: str
    version: str
    input_schema: str
    output_schema: str


class ManifestRegistry:
    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self.intent_schema_by_type: Dict[str, str] = {}
        self.plan_template_by_intent_type: Dict[str, str] = {}
        self.operator_manifest_by_name_version: Dict[Tuple[str, str], OperatorManifest] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._load_intents()
        self._load_capabilities()
        self._load_operator_plugins()

    def _load_intents(self) -> None:
        path = os.path.join(self._repo_root, "manifests", "intent-registry.v1.json")
        with open(path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        for item in reg["intents"]:
            self.intent_schema_by_type[item["intent_type"]] = item["schema_ref"]

    def _load_capabilities(self) -> None:
        caps_dir = os.path.join(self._repo_root, "manifests", "capabilities")
        for filename in os.listdir(caps_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(caps_dir, filename), "r", encoding="utf-8") as f:
                cap = json.load(f)["capability"]
            for pt in cap.get("plan_templates", []):
                self.plan_template_by_intent_type[pt["intent_type"]] = pt["plan_template_ref"]

    def _load_operator_plugins(self) -> None:
        ops_dir = os.path.join(self._repo_root, "manifests", "plugins", "operators")
        for filename in os.listdir(ops_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(ops_dir, filename), "r", encoding="utf-8") as f:
                m = json.load(f)
            manifest = OperatorManifest(
                name=m["name"],
                version=m["version"],
                input_schema=m["schemas"]["input"],
                output_schema=m["schemas"]["output"],
            )
            self.operator_manifest_by_name_version[(manifest.name, manifest.version)] = manifest

    def get_operator_manifest(self, name: str, version: str) -> OperatorManifest:
        return self.operator_manifest_by_name_version[(name, version)]


def resolve_template_value(value: Any, binding_ctx: Dict[str, Any]) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"from"}:
        return dotted_get(binding_ctx, value["from"])
    if isinstance(value, dict) and set(value.keys()) == {"const"}:
        return value["const"]
    if isinstance(value, dict) and set(value.keys()) == {"template"}:
        return interpolate_template(value["template"], binding_ctx)
    if isinstance(value, list):
        return [resolve_template_value(v, binding_ctx) for v in value]
    if isinstance(value, dict):
        return {k: resolve_template_value(v, binding_ctx) for k, v in value.items()}
    return value


_TEMPLATE_RE = re.compile(r"{([^{}]+)}")


def prune_nones(value: Any) -> Any:
    """
    Remove `None` values from dicts/lists recursively.

    Why: template bindings return None when a source field is missing. In JSON Schema, optional
    fields are typically expressed as "absent" (not "null"), so we prune Nones before validation.
    """

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            out[k] = prune_nones(v)
        return out
    if isinstance(value, list):
        return [prune_nones(v) for v in value if v is not None]
    return value


def _normalize_idem_part(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return stable_json_dumps(value)


def interpolate_template(template: str, binding_ctx: Dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        value = dotted_get(binding_ctx, key)
        return _normalize_idem_part(value)

    return _TEMPLATE_RE.sub(repl, template)


def pick_first_nonempty_string(values: Iterable[Any]) -> Optional[str]:
    for v in values:
        if isinstance(v, str) and v.strip() != "":
            return v
    return None


@dataclasses.dataclass
class WorkflowRun:
    query_id: str
    thread_id: int
    intent: Dict[str, Any]
    plan_template: Dict[str, Any]
    binding_ctx: Dict[str, Any]
    status: str = "running"  # running | waiting_action | completed | failed
    pending_action_id: Optional[str] = None
    resume_step_index: Optional[int] = None
    outcomes: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)  # name -> outcome
    events: list[Dict[str, Any]] = dataclasses.field(default_factory=list)  # for CLI demo


class DemoKernel:
    def __init__(self, repo_root: str):
        self._repo_root = repo_root
        self.schemas = SchemaRegistry(repo_root=repo_root)
        self.manifests = ManifestRegistry(repo_root=repo_root)
        self.platform_store_by_thread_id: Dict[int, Dict[str, Any]] = {}
        self.operator_idempotency: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.workflow_by_query_id: Dict[str, WorkflowRun] = {}
        self.workflow_by_action_id: Dict[str, str] = {}

    # -------------------------
    # Public demo entrypoints
    # -------------------------
    def submit_email_review(
        self,
        *,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
        platform_email_subject: Optional[str],
        platform_email_body: Optional[str],
        email_subject_override: Optional[str] = None,
        email_text_override: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ) -> WorkflowRun:
        # Seed the "platform DB" context for this thread.
        self.platform_store_by_thread_id[thread_id] = {
            "student": {"id": student_id, "first_name": "Demo", "last_name": "Student", "email": "demo@example.com"},
            "funding_request": {
                "id": funding_request_id,
                "email_subject": platform_email_subject,
                "email_content": platform_email_body,
            },
            "email": {"id": None, "main_email_subject": platform_email_subject, "main_email_body": platform_email_body},
            "professor": {"id": 910, "full_name": "Prof. Demo", "email_address": "prof@example.edu", "department": "CS"},
            "institute": {"institution_name": "Demo University", "country": "CA"},
        }

        intent = self._build_intent_email_review(
            thread_id=thread_id,
            student_id=student_id,
            funding_request_id=funding_request_id,
            email_subject_override=email_subject_override,
            email_text_override=email_text_override,
            custom_instructions=custom_instructions,
        )

        intent_schema = self.manifests.intent_schema_by_type[intent["intent_type"]]
        self.schemas.validate(intent, intent_schema)

        plan_template_path = self.manifests.plan_template_by_intent_type[intent["intent_type"]]
        plan_template = self._load_json(plan_template_path)
        self.schemas.validate(plan_template, "schemas/plans/plan_template.v1.json")

        query_id = str(uuid.uuid4())
        run = WorkflowRun(
            query_id=query_id,
            thread_id=thread_id,
            intent=intent,
            plan_template=plan_template,
            binding_ctx={
                "tenant_id": intent["actor"]["tenant_id"],
                "thread_id": thread_id,
                "intent": intent,
                "context": {},
                "outcome": {},
                "steps": {},
                "computed": {},
            },
        )
        self.workflow_by_query_id[query_id] = run
        return run

    def run_email_review_to_completion(self, run: WorkflowRun) -> WorkflowRun:
        self._execute_plan(run, start_step_index=0)
        return run

    def resolve_action_and_resume(self, *, action_id: str, accepted: bool, payload: Dict[str, Any]) -> WorkflowRun:
        query_id = self.workflow_by_action_id.get(action_id)
        if not query_id:
            raise KeyError(f"Unknown action_id: {action_id}")
        run = self.workflow_by_query_id[query_id]

        # Run a tiny gate-resolution plan to demonstrate the same mechanism.
        gate_intent = self._build_intent_gate_resolve(
            thread_id=run.thread_id,
            student_id=int(run.intent["actor"]["principal"]["id"]),
            action_id=action_id,
            accepted=accepted,
            payload=payload,
        )
        gate_schema = self.manifests.intent_schema_by_type[gate_intent["intent_type"]]
        self.schemas.validate(gate_intent, gate_schema)

        gate_tpl_path = self.manifests.plan_template_by_intent_type[gate_intent["intent_type"]]
        gate_tpl = self._load_json(gate_tpl_path)
        self.schemas.validate(gate_tpl, "schemas/plans/plan_template.v1.json")

        # Execute the "gate resolve" plan (in-memory operator).
        self._execute_plan(
            WorkflowRun(
                query_id=str(uuid.uuid4()),
                thread_id=run.thread_id,
                intent=gate_intent,
                plan_template=gate_tpl,
                binding_ctx={
                    "tenant_id": gate_intent["actor"]["tenant_id"],
                    "thread_id": run.thread_id,
                    "intent": gate_intent,
                    "context": {},
                    "outcome": {},
                    "steps": {},
                    "computed": {},
                },
            ),
            start_step_index=0,
        )

        if not accepted:
            run.status = "failed"
            self._emit_final(run, summary="Action declined; workflow stopped.")
            return run

        # Apply collected fields to the original intent inputs (demo behavior).
        for k in ("email_text_override", "email_subject_override", "custom_instructions"):
            if k not in payload:
                continue
            if payload[k] is None:
                run.intent["inputs"].pop(k, None)
            else:
                run.intent["inputs"][k] = payload[k]

        run.binding_ctx["intent"] = run.intent
        run.binding_ctx["computed"] = {}

        resume_at = run.resume_step_index or 0
        run.status = "running"
        self._emit_progress(run, percent=30, stage="resuming", message="Resuming after action resolution…")
        self._execute_plan(run, start_step_index=resume_at)
        return run

    # -------------------------
    # Core plan execution
    # -------------------------
    def _execute_plan(self, run: WorkflowRun, *, start_step_index: int) -> None:
        steps = run.plan_template["steps"]
        for idx in range(start_step_index, len(steps)):
            step = steps[idx]
            step_id = step["step_id"]
            kind = step["kind"]

            self._emit_progress(run, percent=min(95, 10 + idx * 20), stage=step_id, message=f"Running {step_id} ({kind})")

            if kind == "policy_check":
                ok, action = self._run_policy_check(step["check"], run.binding_ctx, thread_id=run.thread_id, query_id=run.query_id)
                if not ok:
                    action_id = action["payload"]["action_id"]
                    run.status = "waiting_action"
                    run.pending_action_id = action_id
                    run.resume_step_index = idx + 1
                    self.workflow_by_action_id[action_id] = run.query_id
                    self._emit_event(run, action)
                    self._emit_progress(run, percent=25, stage="paused", message="Paused: waiting for required user input.")
                    return
                continue

            if kind != "operator":
                raise ValueError(f"Demo executor only supports operator/policy_check steps. Got: {kind}")

            operator_name = step["operator_name"]
            operator_version = step["operator_version"]
            payload_template = step["payload"]

            payload = prune_nones(resolve_template_value(payload_template, run.binding_ctx))

            # Compute `computed.*` values when required for idempotency keys.
            self._populate_computed_for_step(run.binding_ctx, operator_name, payload)

            idempotency_key = interpolate_template(step["idempotency_template"], run.binding_ctx)

            manifest = self.manifests.get_operator_manifest(operator_name, operator_version)

            operator_call = {
                "schema_version": "1.0",
                "payload": payload,
                "idempotency_key": idempotency_key,
                "auth_context": {
                    "tenant_id": run.intent["actor"]["tenant_id"],
                    "principal": run.intent["actor"]["principal"],
                    "scopes": run.intent["actor"].get("scopes", []),
                },
                "trace_context": {
                    "correlation_id": run.intent.get("correlation_id"),
                    "workflow_id": run.intent.get("intent_id"),
                    "step_id": step_id,
                    "job_id": str(uuid.uuid4()),
                },
            }
            self.schemas.validate(operator_call, manifest.input_schema)

            operator_result = self._run_operator(operator_name, operator_call)
            self.schemas.validate(operator_result, manifest.output_schema)

            self._apply_produces(run.binding_ctx, produces=step.get("produces", []), operator_result=operator_result)
            self._capture_outcomes_from_produces(run, produces=step.get("produces", []))

        run.status = "completed"
        intent_type = run.intent.get("intent_type")
        if intent_type == "Funding.Outreach.Email.Review":
            summary = "Email review completed."
        elif intent_type == "Workflow.Gate.Resolve":
            summary = "Gate resolution recorded."
        else:
            summary = f"Workflow completed ({intent_type})."
        self._emit_final(run, summary=summary)

    def _populate_computed_for_step(self, binding_ctx: Dict[str, Any], operator_name: str, payload: Dict[str, Any]) -> None:
        if operator_name == "Email.ReviewDraft":
            effective_body = pick_first_nonempty_string([payload.get("body"), payload.get("fallback_body")]) or ""
            dotted_set(binding_ctx, "computed.email_body_hash", sha256_hex(effective_body))

    def _apply_produces(self, binding_ctx: Dict[str, Any], *, produces: list[str], operator_result: Dict[str, Any]) -> None:
        result = operator_result.get("result") or {}
        for key in produces:
            if key.startswith("context."):
                field = key[len("context.") :]
                dotted_set(binding_ctx, key, dotted_get(result, field))
                continue
            if key.startswith("outcome."):
                name = key[len("outcome.") :]
                val = result.get("outcome", result.get(name))
                dotted_set(binding_ctx, key, val)
                continue
            if key.startswith("steps."):
                # Convention: "steps.<step_id>.result.<x>" maps to operator_result.result.<x>
                if ".result." in key:
                    _, _, rest = key.partition(".result.")
                    dotted_set(binding_ctx, key, result.get(rest))
                else:
                    dotted_set(binding_ctx, key, result)
                continue
            # Fallback: set key directly if present in result.
            if key in result:
                dotted_set(binding_ctx, key, result[key])

    def _capture_outcomes_from_produces(self, run: WorkflowRun, *, produces: list[str]) -> None:
        for key in produces:
            if not key.startswith("outcome."):
                continue
            name = key[len("outcome.") :]
            out = dotted_get(run.binding_ctx, f"outcome.{name}")
            if isinstance(out, dict) and out.get("outcome_id") and out.get("outcome_type"):
                run.outcomes[name] = out

    # -------------------------
    # Policy checks (demo)
    # -------------------------
    def _run_policy_check(self, check: Dict[str, Any], binding_ctx: Dict[str, Any], *, thread_id: int, query_id: str) -> Tuple[bool, Dict[str, Any]]:
        name = check["check_name"]
        params = check.get("params", {})

        if name == "EnsureEmailPresent":
            sources = params.get("sources", [])
            vals = [dotted_get(binding_ctx, p) for p in sources]
            if pick_first_nonempty_string(vals) is not None:
                return True, {}
            action_id = str(uuid.uuid4())
            return False, self._make_action_required(
                thread_id=thread_id,
                query_id=query_id,
                action_id=action_id,
                action_type=params.get("on_missing_action_type", "collect_fields"),
                title="Email text required",
                description="Provide the email body to review (either override or ensure the platform has a draft).",
                data={"missing_sources": sources},
            )

        raise ValueError(f"Unknown check_name: {name}")

    # -------------------------
    # Operators (demo stubs)
    # -------------------------
    def _run_operator(self, operator_name: str, call: Dict[str, Any]) -> Dict[str, Any]:
        key = (operator_name, call["idempotency_key"])
        cached = self.operator_idempotency.get(key)
        if cached is not None:
            return cached

        if operator_name == "Platform.Context.Load":
            res = self._op_platform_context_load(call)
        elif operator_name == "Email.ReviewDraft":
            res = self._op_email_review_draft(call)
        elif operator_name == "Workflow.Gate.Resolve":
            res = self._op_workflow_gate_resolve(call)
        else:
            raise ValueError(f"Operator not implemented in demo: {operator_name}")

        self.operator_idempotency[key] = res
        return res

    def _op_platform_context_load(self, call: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = int(call["payload"]["thread_id"])
        platform = self.platform_store_by_thread_id.get(thread_id)
        if not platform:
            return self._op_error_result(code="PLATFORM_CONTEXT_NOT_FOUND", message=f"No platform context for thread_id={thread_id}")

        fr: Dict[str, Any] = {"id": platform["funding_request"]["id"]}
        if platform["funding_request"].get("email_subject") is not None:
            fr["email_subject"] = platform["funding_request"]["email_subject"]
        if platform["funding_request"].get("email_content") is not None:
            fr["email_content"] = platform["funding_request"]["email_content"]

        email: Dict[str, Any] = {}
        if platform.get("email", {}).get("id") is not None:
            email["id"] = platform["email"]["id"]
        if platform.get("email", {}).get("main_email_subject") is not None:
            email["main_email_subject"] = platform["email"]["main_email_subject"]
        if platform.get("email", {}).get("main_email_body") is not None:
            email["main_email_body"] = platform["email"]["main_email_body"]

        # Build a minimal object matching schemas/operators/platform_context_load.output.v1.json
        result = {
            "platform": {
                "student": platform["student"],
                "funding_request": fr,
                "email": email or None,
                "professor": platform["professor"],
                "institute": platform["institute"],
                "metas": None,
                "reply": None,
            },
            "intelligence": {
                "requirements": {},
                "preferences": {},
                "background": {},
                "composer_prereqs": {},
                "gmail_connected": False,
                "templates_finalized": False,
            },
            "platform_context_hash": hash_object(platform),
            "student_background_hash": hash_object({}),
            "composer_prereqs_hash": hash_object({}),
        }

        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": result,
            "artifacts": [],
            "metrics": {"latency_ms": 1, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    def _op_email_review_draft(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]
        subject = pick_first_nonempty_string([payload.get("subject"), payload.get("fallback_subject")]) or "(no subject)"
        body = pick_first_nonempty_string([payload.get("body"), payload.get("fallback_body")]) or ""

        issues = []
        score = 1.0
        verdict = "pass"

        if len(body.strip()) < 80:
            issues.append({"code": "TOO_SHORT", "severity": "warning", "message": "Email body is quite short."})
            score -= 0.25
            verdict = "needs_edits"

        if "dear" not in body.lower():
            issues.append({"code": "NO_GREETING", "severity": "info", "message": "Consider adding a greeting (e.g., 'Dear Prof. …')."})
            score -= 0.10

        score = max(0.0, min(1.0, score))
        if score < 0.5:
            verdict = "needs_edits"

        outcome_payload = {"verdict": verdict, "overall_score": score, "issues": issues, "notes": "Demo review."}
        outcome = {
            "schema_version": "1.0",
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "Email.Review",
            "created_at": utc_now_iso(),
            "hash": hash_object(outcome_payload),
            "producer": {"name": "Email.ReviewDraft", "version": "1.0.0", "plugin_type": "operator"},
            "payload": outcome_payload,
        }
        self.schemas.validate(outcome, "schemas/outcomes/email_review.v1.json")

        result = {"outcome": outcome}
        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": result,
            "artifacts": [],
            "metrics": {"latency_ms": 3, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    def _op_workflow_gate_resolve(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call["payload"]
        result: Dict[str, Any] = {
            "action_id": payload["action_id"],
            "status": payload["status"],
            "resolved_at": utc_now_iso(),
            "message": "Recorded gate resolution (demo).",
        }
        return {
            "schema_version": "1.0",
            "status": "succeeded",
            "result": result,
            "artifacts": [],
            "metrics": {"latency_ms": 1, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": None,
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    def _op_error_result(self, *, code: str, message: str) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "status": "failed",
            "result": None,
            "artifacts": [],
            "metrics": {"latency_ms": 0, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_total_usd": 0.0},
            "error": {"code": code, "message": message, "category": "dependency", "retryable": False},
            "nondeterminism": {"is_nondeterministic": False, "reasons": [], "stability": "high"},
        }

    # -------------------------
    # Intent builders (demo)
    # -------------------------
    def _build_intent_email_review(
        self,
        *,
        thread_id: int,
        student_id: int,
        funding_request_id: int,
        email_subject_override: Optional[str],
        email_text_override: Optional[str],
        custom_instructions: Optional[str],
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {"review_mode": "rubric"}
        if email_text_override is not None:
            inputs["email_text_override"] = email_text_override
        if email_subject_override is not None:
            inputs["email_subject_override"] = email_subject_override
        if custom_instructions is not None:
            inputs["custom_instructions"] = custom_instructions

        intent = {
            "schema_version": "1.0",
            "intent_id": str(uuid.uuid4()),
            "intent_type": "Funding.Outreach.Email.Review",
            "actor": {
                "tenant_id": 1,
                "principal": {"type": "student", "id": student_id},
                "scopes": ["funding:outreach"],
                "trust_level": 1,
            },
            "source": "api",
            "thread_id": thread_id,
            "scope": {"scope_type": "funding_request", "scope_id": str(funding_request_id)},
            "inputs": inputs,
            "constraints": {},
            "context_refs": {},
            "data_classes": ["Confidential"],
            "correlation_id": str(uuid.uuid4()),
            "created_at": utc_now_iso(),
        }
        return intent

    def _build_intent_gate_resolve(
        self,
        *,
        thread_id: int,
        student_id: int,
        action_id: str,
        accepted: bool,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "intent_id": str(uuid.uuid4()),
            "intent_type": "Workflow.Gate.Resolve",
            "actor": {"tenant_id": 1, "principal": {"type": "student", "id": student_id}, "scopes": ["funding:outreach"]},
            "source": "api",
            "thread_id": thread_id,
            "scope": {"scope_type": "funding_request", "scope_id": "demo"},
            "inputs": {
                "action_id": action_id,
                "status": "accepted" if accepted else "declined",
                "payload": payload,
            },
            "constraints": {},
            "context_refs": {},
            "data_classes": ["Internal"],
            "correlation_id": str(uuid.uuid4()),
            "created_at": utc_now_iso(),
        }

    # -------------------------
    # Events (validated)
    # -------------------------
    def _make_event_base(self, *, thread_id: int, query_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "thread_id": thread_id,
            "query_id": query_id,
            "ts": utc_now_iso(),
            "payload": payload,
        }

    def _emit_progress(self, run: WorkflowRun, *, percent: int, stage: str, message: str) -> None:
        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="progress",
            payload={"percent": percent, "stage": stage, "message": message},
        )
        self.schemas.validate(event, "schemas/sse/progress.v1.json")
        self._emit_event(run, event)

    def _make_action_required(
        self,
        *,
        thread_id: int,
        query_id: str,
        action_id: str,
        action_type: str,
        title: str,
        description: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        event = self._make_event_base(
            thread_id=thread_id,
            query_id=query_id,
            event_type="action_required",
            payload={
                "action_id": action_id,
                "action_type": action_type,
                "title": title,
                "description": description,
                "requires_user_input": True,
                "ui_hints": {"primary_button": "Submit", "secondary_button": "Cancel"},
                "data": data,
            },
        )
        self.schemas.validate(event, "schemas/sse/action_required.v1.json")
        return event

    def _emit_final(self, run: WorkflowRun, *, summary: str) -> None:
        outcomes_emitted = []
        for name, out in run.outcomes.items():
            outcomes_emitted.append({"outcome_id": out["outcome_id"], "outcome_type": out["outcome_type"], "hash": out["hash"]})

        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="final",
            payload={"summary": summary, "outcomes_emitted": outcomes_emitted, "actions_emitted": []},
        )
        self.schemas.validate(event, "schemas/sse/final.v1.json")
        self._emit_event(run, event)

    def _emit_error(self, run: WorkflowRun, *, code: str, message: str) -> None:
        event = self._make_event_base(
            thread_id=run.thread_id,
            query_id=run.query_id,
            event_type="error",
            payload={"error": {"code": code, "message": message, "category": "operator_bug", "retryable": False}},
        )
        self.schemas.validate(event, "schemas/sse/error.v1.json")
        self._emit_event(run, event)

    def _emit_event(self, run: WorkflowRun, event: Dict[str, Any]) -> None:
        # Validate against the union schema too (demo of "protocol validation").
        self.schemas.validate(event, "schemas/sse/event.v1.json")
        run.events.append(event)
        if event["event_type"] == "action_required":
            run.pending_action_id = event["payload"]["action_id"]

    # -------------------------
    # Local file loads
    # -------------------------
    def _load_json(self, repo_rel_path: str) -> Dict[str, Any]:
        path = os.path.join(self._repo_root, repo_rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
