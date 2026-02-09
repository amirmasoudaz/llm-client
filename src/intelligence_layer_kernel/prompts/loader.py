from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blake3 import blake3


@dataclass(frozen=True)
class PromptRenderResult:
    template_id: str
    template_hash: str
    text: str


class PromptTemplateLoader:
    def __init__(self, *, base_path: Path | None = None) -> None:
        if base_path is None:
            base_path = Path(__file__).resolve().parents[3] / "src" / "intelligence_layer_prompts"
        self._base_path = base_path
        self._env = _build_environment(self._base_path)

    def render(self, template_id: str, context: dict[str, Any]) -> PromptRenderResult:
        normalized = _normalize_template_id(template_id)
        template = self._env.get_template(normalized)
        text = template.render(**context)
        source = self._get_template_source(normalized)
        template_hash = blake3(source.encode("utf-8")).hexdigest()
        return PromptRenderResult(template_id=normalized, template_hash=template_hash, text=text)

    def _get_template_source(self, template_id: str) -> str:
        source, _filename, _uptodate = self._env.loader.get_source(self._env, template_id)
        return source


def _build_environment(base_path: Path):
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Jinja2 is required for prompt templates. Install with: pip install .[layer2]") from exc

    return Environment(
        loader=FileSystemLoader(str(base_path)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def _normalize_template_id(template_id: str) -> str:
    template_id = template_id.strip().lstrip("/")
    if not template_id.endswith(".j2"):
        template_id = template_id + ".j2"
    return template_id
