from __future__ import annotations

import json
from typing import Any

from blake3 import blake3

from .types import PolicyContext, PolicyDecision


class PolicyEngine:
    def __init__(self, *, name: str = "il_policy", version: str = "0.1") -> None:
        self.name = name
        self.version = version

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        payload = json.dumps(context.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        inputs_hash = blake3(payload).digest()
        return PolicyDecision(
            stage=context.stage,
            decision="ALLOW",
            reason_code="default_allow",
            reason=None,
            requirements={},
            limits={},
            redactions=[],
            transform=None,
            inputs_hash=inputs_hash,
            policy_engine_name=self.name,
            policy_engine_version=self.version,
        )
