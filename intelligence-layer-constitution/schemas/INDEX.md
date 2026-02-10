# Schema index (v1)

This is a navigation aid for the schema and workflow template files in this repo.

## Core

- `schemas/common/defs.v1.json`
- `schemas/common/intent_base.v1.json`
- `schemas/common/error.v1.json`
- `schemas/common/platform_patch_proposal.v1.json`
- `schemas/common/platform_patch_receipt.v1.json`

## Intents

- `schemas/intents/thread_init.v1.json`
- `schemas/intents/funding_outreach_professor_summarize.v1.json`
- `schemas/intents/funding_outreach_alignment_score.v1.json`
- `schemas/intents/funding_outreach_email_generate.v1.json`
- `schemas/intents/funding_outreach_email_review.v1.json`
- `schemas/intents/funding_outreach_email_optimize.v1.json`
- `schemas/intents/funding_paper_metadata_extract.v1.json`
- `schemas/intents/funding_request_fields_update.v1.json`
- `schemas/intents/documents_upload.v1.json`
- `schemas/intents/documents_process.v1.json`
- `schemas/intents/documents_review.v1.json`
- `schemas/intents/documents_optimize.v1.json`
- `schemas/intents/documents_compose_sop.v1.json`
- `schemas/intents/workflow_gate_resolve.v1.json`
- `schemas/intents/student_profile_collect.v1.json`

## Plans

- `schemas/plans/plan_template.v1.json`
- `schemas/plans/plan.v1.json`
- `schemas/plans/step.v1.json`

## Outcomes

- `schemas/outcomes/outcome_base.v1.json`
- `schemas/outcomes/professor_summary.v1.json`
- `schemas/outcomes/alignment_score.v1.json`
- `schemas/outcomes/email_draft.v1.json`
- `schemas/outcomes/email_review.v1.json`
- `schemas/outcomes/conversation_suggestions.v1.json`
- `schemas/outcomes/paper_metadata.v1.json`
- `schemas/outcomes/platform_patch_proposal.v1.json`
- `schemas/outcomes/platform_patch_receipt.v1.json`
- `schemas/outcomes/document_uploaded.v1.json`
- `schemas/outcomes/document_processed.v1.json`
- `schemas/outcomes/document_review.v1.json`
- `schemas/outcomes/document_optimized.v1.json`
- `schemas/outcomes/document_composed.v1.json`
- `schemas/outcomes/artifact_pdf.v1.json`

## Operators (schemas)

- `schemas/operators/operator_call_base.v1.json`
- `schemas/operators/operator_result_base.v1.json`
- `schemas/operators/thread_create_or_load.input.v1.json`
- `schemas/operators/thread_create_or_load.output.v1.json`
- `schemas/operators/platform_context_load.input.v1.json`
- `schemas/operators/platform_context_load.output.v1.json`
- `schemas/operators/platform_attachments_list.input.v1.json`
- `schemas/operators/platform_attachments_list.output.v1.json`
- `schemas/operators/workflow_gate_resolve.input.v1.json`
- `schemas/operators/workflow_gate_resolve.output.v1.json`
- `schemas/operators/professor_profile_retrieve.input.v1.json`
- `schemas/operators/professor_profile_retrieve.output.v1.json`
- `schemas/operators/professor_summarize.input.v1.json`
- `schemas/operators/professor_summarize.output.v1.json`
- `schemas/operators/professor_alignment_score.input.v1.json`
- `schemas/operators/professor_alignment_score.output.v1.json`
- `schemas/operators/email_generate_draft.input.v1.json`
- `schemas/operators/email_generate_draft.output.v1.json`
- `schemas/operators/email_review_draft.input.v1.json`
- `schemas/operators/email_review_draft.output.v1.json`
- `schemas/operators/conversation_suggestions_generate.input.v1.json`
- `schemas/operators/conversation_suggestions_generate.output.v1.json`
- `schemas/operators/email_optimize_draft.input.v1.json`
- `schemas/operators/email_optimize_draft.output.v1.json`
- `schemas/operators/email_apply_to_platform_propose.input.v1.json`
- `schemas/operators/email_apply_to_platform_propose.output.v1.json`
- `schemas/operators/funding_email_draft_update_propose.input.v1.json`
- `schemas/operators/funding_email_draft_update_propose.output.v1.json`
- `schemas/operators/funding_email_draft_update_apply.input.v1.json`
- `schemas/operators/funding_email_draft_update_apply.output.v1.json`
- `schemas/operators/paper_metadata_extract.input.v1.json`
- `schemas/operators/paper_metadata_extract.output.v1.json`
- `schemas/operators/funding_request_fields_update_propose.input.v1.json`
- `schemas/operators/funding_request_fields_update_propose.output.v1.json`
- `schemas/operators/funding_request_fields_update_apply.input.v1.json`
- `schemas/operators/funding_request_fields_update_apply.output.v1.json`
- `schemas/operators/student_profile_load_or_create.input.v1.json`
- `schemas/operators/student_profile_load_or_create.output.v1.json`
- `schemas/operators/student_profile_update.input.v1.json`
- `schemas/operators/student_profile_update.output.v1.json`
- `schemas/operators/student_profile_requirements_evaluate.input.v1.json`
- `schemas/operators/student_profile_requirements_evaluate.output.v1.json`
- `schemas/operators/memory_upsert.input.v1.json`
- `schemas/operators/memory_upsert.output.v1.json`
- `schemas/operators/memory_retrieve.input.v1.json`
- `schemas/operators/memory_retrieve.output.v1.json`
- `schemas/operators/documents_upload.input.v1.json`
- `schemas/operators/documents_upload.output.v1.json`
- `schemas/operators/documents_import_from_platform_attachment.input.v1.json`
- `schemas/operators/documents_import_from_platform_attachment.output.v1.json`
- `schemas/operators/documents_process.input.v1.json`
- `schemas/operators/documents_process.output.v1.json`
- `schemas/operators/documents_review.input.v1.json`
- `schemas/operators/documents_review.output.v1.json`
- `schemas/operators/documents_optimize.input.v1.json`
- `schemas/operators/documents_optimize.output.v1.json`
- `schemas/operators/documents_compose_sop.input.v1.json`
- `schemas/operators/documents_compose_sop.output.v1.json`
- `schemas/operators/documents_export.input.v1.json`
- `schemas/operators/documents_export.output.v1.json`
- `schemas/operators/documents_apply_to_platform_propose.input.v1.json`
- `schemas/operators/documents_apply_to_platform_propose.output.v1.json`

## SSE

- `schemas/sse/event_base.v1.json`
- `schemas/sse/event.v1.json`
- `schemas/sse/progress.v1.json`
- `schemas/sse/token_delta.v1.json`
- `schemas/sse/action_required.v1.json`
- `schemas/sse/artifact_ready.v1.json`
- `schemas/sse/final.v1.json`
- `schemas/sse/error.v1.json`
