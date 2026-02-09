from __future__ import annotations

from ..types import OperatorCall, OperatorResult
from .funding_email_draft_update_propose import FundingEmailDraftUpdateProposeOperator


class EmailApplyToPlatformProposeOperator(FundingEmailDraftUpdateProposeOperator):
    name = "Email.ApplyToPlatform.Propose"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        result = await super().run(call)
        if result.status == "succeeded" and isinstance(result.result, dict):
            result.result = {"outcome": result.result.get("outcome")}
        return result
