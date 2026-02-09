RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reply_digestion_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # Cleaning outputs
                "reply_body_cleaned": {"type": "string"},
                "is_auto_generated": {"type": "boolean"},
                "auto_generated_type": {
                    "type": "string",
                    "enum": [
                        "OUT_OF_OFFICE",
                        "DELIVERY_FAILURE",
                        "OTHER_AUTO",
                        "NONE"
                    ]
                },
                "needs_human_review": {"type": "boolean"},
                # Classification outputs
                "engagement_label": {
                    "type": "string",
                    "enum": [
                        "INTERESTED_AND_WANTS_TO_PROCEED",
                        "POTENTIALLY_INTERESTED_NEEDS_MORE_INFO",
                        "NOT_INTERESTED",
                        "NO_AVAILABLE_POSITION",
                        "REFERRAL_TO_SOMEONE_ELSE",
                        "OUT_OF_SCOPE_OR_WRONG_PERSON",
                        "AUTO_REPLY_OR_OUT_OF_OFFICE",
                        "AMBIGUOUS_OR_UNCLEAR"
                    ]
                },
                "engagement_bool": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"}
                    ]
                },
                "activity_status": {
                    "type": "string",
                    "enum": [
                        "ACTIVE_SUPERVISING",
                        "ACTIVE_NOT_SUPERVISING",
                        "CLEARLY_INACTIVE_SUPERVISION",
                        "UNKNOWN"
                    ]
                },
                "activity_bool": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"}
                    ]
                },
                "next_step_type": {
                    "type": "string",
                    "enum": [
                        "REQUEST_MEETING",
                        "REQUEST_CV",
                        "REQUEST_TRANSCRIPT",
                        "REQUEST_RESEARCH_PROPOSAL_OR_DETAILS",
                        "INVITE_APPLY_MENTION_ME",
                        "APPLY_VIA_PORTAL_OR_CONTACT_ADMISSIONS",
                        "REFER_TO_OTHER_PROFESSOR_OR_PERSON",
                        "NO_NEXT_STEP"
                    ]
                },
                "short_rationale": {"type": "string"},
                "key_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 6
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": [
                "reply_body_cleaned",
                "is_auto_generated",
                "auto_generated_type",
                "needs_human_review",
                "engagement_label",
                "engagement_bool",
                "activity_status",
                "activity_bool",
                "next_step_type",
                "short_rationale",
                "key_phrases",
                "confidence"
            ]
        }
    }
}