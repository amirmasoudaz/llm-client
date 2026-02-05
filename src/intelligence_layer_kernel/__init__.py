"""Intelligence Layer Kernel (Layer 2 foundations)."""

from .contracts.registry import ContractRegistry, ContractValidationError, ContractValidationReport
from .ids import TraceContext, new_uuid

__all__ = [
    "ContractRegistry",
    "ContractValidationError",
    "ContractValidationReport",
    "TraceContext",
    "new_uuid",
]
