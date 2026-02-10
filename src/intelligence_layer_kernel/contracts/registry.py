from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Iterable

from jsonschema import Draft202012Validator, RefResolver


@dataclass(frozen=True)
class ContractPaths:
    root: Path
    schemas: Path
    manifests: Path
    plan_templates: Path
    prompts: Path


@dataclass
class ContractValidationReport:
    schema_count: int
    intent_count: int
    plan_template_count: int
    operator_manifest_count: int
    capability_manifest_count: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


class ContractValidationError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__("Contract validation failed")
        self.errors = errors


class ContractRegistry:
    def __init__(self, *, root: Path | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[3] / "intelligence-layer-constitution"
        self.paths = ContractPaths(
            root=root,
            schemas=root / "schemas",
            manifests=root / "manifests",
            plan_templates=root / "plan-templates",
            prompts=root.parent / "src" / "intelligence_layer_prompts",
        )

        self._loaded = False
        self._schemas: dict[Path, dict[str, Any]] = {}
        self._schemas_by_relpath: dict[str, dict[str, Any]] = {}
        self._schema_store: dict[str, dict[str, Any]] = {}

        self._intent_registry: dict[str, Any] | None = None
        self._intent_schema_by_type: dict[str, dict[str, Any]] = {}

        self._operator_manifests: dict[tuple[str, str], dict[str, Any]] = {}
        self._capability_manifests: dict[tuple[str, str], dict[str, Any]] = {}
        self._plan_templates_by_intent: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        self._load_schemas()
        self._load_intent_registry()
        self._load_operator_manifests()
        self._load_capability_manifests()
        self._load_plan_templates()
        self._loaded = True

    def validate(self, *, raise_on_error: bool = True) -> ContractValidationReport:
        if not self._loaded:
            self.load()

        errors: list[str] = []
        self._validate_schemas(errors)
        self._validate_intent_registry(errors)
        self._validate_operator_manifests(errors)
        self._validate_plan_templates(errors)
        self._validate_capability_manifests(errors)

        report = ContractValidationReport(
            schema_count=len(self._schemas),
            intent_count=len(self._intent_schema_by_type),
            plan_template_count=len(self._plan_templates_by_intent),
            operator_manifest_count=len(self._operator_manifests),
            capability_manifest_count=len(self._capability_manifests),
            errors=errors,
        )

        if raise_on_error and report.errors:
            raise ContractValidationError(report.errors)
        return report

    def get_intent_schema(self, intent_type: str) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        schema = self._intent_schema_by_type.get(intent_type)
        if schema is None:
            raise KeyError(f"Unknown intent_type: {intent_type}")
        return schema

    def get_plan_template(self, intent_type: str) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        plan = self._plan_templates_by_intent.get(intent_type)
        if plan is None:
            raise KeyError(f"No plan template for intent_type: {intent_type}")
        return plan

    def get_operator_manifest(self, operator_name: str, version: str) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        key = (operator_name, version)
        manifest = self._operator_manifests.get(key)
        if manifest is None:
            raise KeyError(f"Unknown operator manifest: {operator_name}@{version}")
        return manifest

    def get_schema_by_ref(self, ref: str) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        schema = self._schemas_by_relpath.get(ref)
        if schema is not None:
            return schema
        schema = self._schema_store.get(ref)
        if schema is not None:
            return schema
        raise KeyError(f"Unknown schema ref: {ref}")

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_schemas(self) -> None:
        self._schemas.clear()
        self._schemas_by_relpath.clear()
        self._schema_store.clear()

        for path in sorted(self.paths.schemas.rglob("*.json")):
            schema = self._load_json(path)
            self._schemas[path] = schema

            rel = path.relative_to(self.paths.root).as_posix()
            self._schemas_by_relpath[rel] = schema

            file_uri = path.resolve().as_uri()
            self._schema_store[file_uri] = schema

            schema_id = schema.get("$id")
            if isinstance(schema_id, str) and schema_id:
                self._schema_store[schema_id] = schema
                if "://" not in schema_id:
                    schema["$id"] = file_uri

    def _load_intent_registry(self) -> None:
        path = self.paths.manifests / "intent-registry.v1.json"
        self._intent_registry = self._load_json(path)

        self._intent_schema_by_type.clear()
        intents = self._intent_registry.get("intents", []) if isinstance(self._intent_registry, dict) else []
        for entry in intents:
            if not isinstance(entry, dict):
                continue
            intent_type = str(entry.get("intent_type") or "")
            schema_ref = str(entry.get("schema_ref") or "")
            if not intent_type or not schema_ref:
                continue
            schema = self._schemas_by_relpath.get(schema_ref)
            if schema is None:
                continue
            self._intent_schema_by_type[intent_type] = schema

    def _load_operator_manifests(self) -> None:
        self._operator_manifests.clear()
        path = self.paths.manifests / "plugins" / "operators"
        if not path.exists():
            return
        for manifest_path in sorted(path.glob("*.json")):
            manifest = self._load_json(manifest_path)
            name = str(manifest.get("name") or "")
            version = str(manifest.get("version") or "")
            if name and version:
                self._operator_manifests[(name, version)] = manifest

    def _load_capability_manifests(self) -> None:
        self._capability_manifests.clear()
        path = self.paths.manifests / "capabilities"
        if not path.exists():
            return
        for manifest_path in sorted(path.glob("*.json")):
            manifest = self._load_json(manifest_path)
            cap = manifest.get("capability") if isinstance(manifest, dict) else None
            if not isinstance(cap, dict):
                continue
            name = str(cap.get("name") or "")
            version = str(cap.get("version") or "")
            if name and version:
                self._capability_manifests[(name, version)] = manifest

    def _load_plan_templates(self) -> None:
        self._plan_templates_by_intent.clear()
        if not self.paths.plan_templates.exists():
            return
        for template_path in sorted(self.paths.plan_templates.glob("*.json")):
            template = self._load_json(template_path)
            intent_type = str(template.get("intent_type") or "")
            if not intent_type:
                continue
            self._plan_templates_by_intent[intent_type] = template

    def _validate_schemas(self, errors: list[str]) -> None:
        for path, schema in self._schemas.items():
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"schema invalid: {path} -> {exc}")
                continue

            for ref in _iter_refs(schema):
                if ref.startswith("#"):
                    continue
                if ref.startswith("http://") or ref.startswith("https://"):
                    continue
                ref_path = ref.split("#", 1)[0]
                if not ref_path:
                    continue
                target = self._resolve_ref_path(base=path, ref_path=ref_path)
                if target is None or not target.exists():
                    errors.append(f"schema ref missing: {path} -> {ref}")

    def _validate_intent_registry(self, errors: list[str]) -> None:
        if self._intent_registry is None:
            errors.append("intent registry not loaded")
            return
        registry_schema = self._schemas_by_relpath.get("schemas/intents/registry.v1.json")
        if registry_schema is None:
            errors.append("missing schema: schemas/intents/registry.v1.json")
        else:
            self._validate_against_schema(
                registry_schema, self._intent_registry, "intent-registry.v1.json", errors
            )

        intents = self._intent_registry.get("intents", [])
        if not isinstance(intents, list):
            errors.append("intent registry: intents must be a list")
            return

        for entry in intents:
            if not isinstance(entry, dict):
                continue
            intent_type = str(entry.get("intent_type") or "")
            schema_ref = str(entry.get("schema_ref") or "")
            if not schema_ref:
                errors.append(f"intent registry missing schema_ref for {intent_type or '<unknown>'}")
                continue
            schema = self._schemas_by_relpath.get(schema_ref)
            if schema is None:
                errors.append(f"intent registry schema_ref missing file: {schema_ref}")
                continue
            declared_intent = _extract_intent_type(schema)
            if declared_intent and declared_intent != intent_type:
                errors.append(
                    f"intent registry mismatch: {intent_type} vs schema intent_type {declared_intent} ({schema_ref})"
                )

    def _validate_operator_manifests(self, errors: list[str]) -> None:
        for (name, version), manifest in self._operator_manifests.items():
            if manifest.get("type") != "operator":
                errors.append(f"operator manifest {name}@{version} has invalid type")
            schemas = manifest.get("schemas")
            if not isinstance(schemas, dict):
                errors.append(f"operator manifest {name}@{version} missing schemas")
                continue
            input_ref = schemas.get("input")
            output_ref = schemas.get("output")
            if not input_ref or not output_ref:
                errors.append(f"operator manifest {name}@{version} missing input/output schema refs")
                continue
            if not self._schemas_by_relpath.get(str(input_ref)):
                errors.append(f"operator manifest {name}@{version} input schema missing: {input_ref}")
            if not self._schemas_by_relpath.get(str(output_ref)):
                errors.append(f"operator manifest {name}@{version} output schema missing: {output_ref}")

            for prompt_ref in _iter_prompt_template_refs(manifest):
                normalized_prompt = _normalize_prompt_template_ref(prompt_ref)
                if normalized_prompt is None:
                    errors.append(
                        f"operator manifest {name}@{version} has invalid prompt template ref: {prompt_ref}"
                    )
                    continue
                prompt_path = self.paths.prompts / normalized_prompt
                if not prompt_path.exists():
                    errors.append(
                        f"operator manifest {name}@{version} prompt template missing: {normalized_prompt}"
                    )

    def _validate_plan_templates(self, errors: list[str]) -> None:
        template_schema = self._schemas_by_relpath.get("schemas/plans/plan_template.v1.json")
        if template_schema is None:
            errors.append("missing schema: schemas/plans/plan_template.v1.json")
            return

        for intent_type, template in self._plan_templates_by_intent.items():
            self._validate_against_schema(template_schema, template, f"plan-template:{intent_type}", errors)

            if intent_type not in self._intent_schema_by_type:
                errors.append(f"plan template intent_type not in registry: {intent_type}")

            steps = template.get("steps", [])
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("kind") != "operator":
                    continue
                operator_name = str(step.get("operator_name") or "")
                operator_version = str(step.get("operator_version") or "")
                if not operator_name or not operator_version:
                    errors.append(f"plan step missing operator name/version (intent={intent_type})")
                    continue
                if (operator_name, operator_version) not in self._operator_manifests:
                    errors.append(
                        f"plan step references unknown operator: {operator_name}@{operator_version} (intent={intent_type})"
                    )

    def _validate_capability_manifests(self, errors: list[str]) -> None:
        for (name, version), manifest in self._capability_manifests.items():
            cap = manifest.get("capability") if isinstance(manifest, dict) else None
            if not isinstance(cap, dict):
                errors.append(f"capability manifest {name}@{version} missing capability object")
                continue
            supported_intents = cap.get("supported_intents", [])
            if not isinstance(supported_intents, list):
                errors.append(f"capability {name}@{version} supported_intents must be list")
                supported_intents = []
            for intent_type in supported_intents:
                if intent_type not in self._intent_schema_by_type:
                    errors.append(f"capability {name}@{version} references unknown intent: {intent_type}")

            plan_templates = cap.get("plan_templates", [])
            if not isinstance(plan_templates, list):
                errors.append(f"capability {name}@{version} plan_templates must be list")
                continue
            for entry in plan_templates:
                if not isinstance(entry, dict):
                    continue
                plan_ref = entry.get("plan_template_ref")
                if not isinstance(plan_ref, str) or not plan_ref:
                    errors.append(f"capability {name}@{version} has invalid plan_template_ref")
                    continue
                plan_path = self.paths.root / plan_ref
                if not plan_path.exists():
                    errors.append(f"capability {name}@{version} plan_template_ref missing: {plan_ref}")

    def _validate_against_schema(
        self,
        schema: dict[str, Any],
        instance: dict[str, Any],
        label: str,
        errors: list[str],
    ) -> None:
        validator = Draft202012Validator(schema, resolver=self.resolver_for(schema))
        for err in validator.iter_errors(instance):
            path = "/".join([str(p) for p in err.path])
            path = path or "<root>"
            errors.append(f"{label} validation error at {path}: {err.message}")

    def resolver_for(self, schema: dict[str, Any]) -> RefResolver:
        base_uri = None
        for path, candidate in self._schemas.items():
            if candidate is schema:
                base_uri = path.resolve().as_uri()
                break
        if base_uri is None:
            schema_id = schema.get("$id")
            base_uri = schema_id if isinstance(schema_id, str) and schema_id else ""
        return RefResolver(base_uri=base_uri, referrer=schema, store=self._schema_store)

    def _resolve_ref_path(self, *, base: Path, ref_path: str) -> Path | None:
        if ref_path.startswith("schemas/") or ref_path.startswith("plan-templates/") or ref_path.startswith("manifests/"):
            return (self.paths.root / ref_path).resolve()
        if ref_path.startswith("/"):
            return (self.paths.root / ref_path.lstrip("/")).resolve()
        return (base.parent / ref_path).resolve()


def _iter_refs(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item)


def _extract_intent_type(schema: dict[str, Any]) -> str | None:
    # Heuristic: find properties.intent_type.const inside any allOf branch.
    if "properties" in schema:
        props = schema.get("properties", {})
        if isinstance(props, dict):
            intent = props.get("intent_type")
            if isinstance(intent, dict) and "const" in intent:
                return str(intent.get("const"))
    for branch in schema.get("allOf", []) if isinstance(schema, dict) else []:
        if not isinstance(branch, dict):
            continue
        props = branch.get("properties", {})
        if isinstance(props, dict):
            intent = props.get("intent_type")
            if isinstance(intent, dict) and "const" in intent:
                return str(intent.get("const"))
    return None


def _iter_prompt_template_refs(manifest: dict[str, Any]) -> Iterable[str]:
    direct_keys = ("prompt_template", "prompt_template_id")
    for key in direct_keys:
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            yield value

    prompt_templates = manifest.get("prompt_templates")
    if isinstance(prompt_templates, str):
        if prompt_templates.strip():
            yield prompt_templates
        return
    if isinstance(prompt_templates, list):
        for item in prompt_templates:
            if isinstance(item, str) and item.strip():
                yield item
            if isinstance(item, dict):
                template_id = item.get("template_id") or item.get("id")
                if isinstance(template_id, str) and template_id.strip():
                    yield template_id
        return
    if isinstance(prompt_templates, dict):
        for value in prompt_templates.values():
            if isinstance(value, str) and value.strip():
                yield value
            if isinstance(value, dict):
                template_id = value.get("template_id") or value.get("id")
                if isinstance(template_id, str) and template_id.strip():
                    yield template_id


def _normalize_prompt_template_ref(template_ref: str) -> str | None:
    normalized = template_ref.strip().lstrip("/")
    if not normalized:
        return None
    if not normalized.endswith(".j2"):
        normalized += ".j2"
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3:
        return None
    return "/".join(parts)
