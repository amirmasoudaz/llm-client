from __future__ import annotations

import sys

from .registry import ContractRegistry, ContractValidationError


def main() -> int:
    registry = ContractRegistry()
    try:
        report = registry.validate(raise_on_error=True)
    except ContractValidationError as exc:
        print("Contract validation failed:\n")
        for line in exc.errors:
            print(f"- {line}")
        return 1

    print("Contract validation OK")
    print(f"Schemas: {report.schema_count}")
    print(f"Intents: {report.intent_count}")
    print(f"Plan templates: {report.plan_template_count}")
    print(f"Operator manifests: {report.operator_manifest_count}")
    print(f"Capability manifests: {report.capability_manifest_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
