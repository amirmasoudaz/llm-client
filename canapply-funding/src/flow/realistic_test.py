from typing import Literal

from src.config import settings


# host and port of the funding API
host: str = settings.HOST_API
port: int = settings.PORT_API

# the location of the database to connect to
db_location: Literal["prod", "stage"] = "stage"

# if db_location is "stage", run the following command
tunnel_stage_cmd = "ssh -N -L 13306:10.188.0.2:3306 root@34.0.45.8"

# if db_location is "prod", run the following command
tunnel_prod_cmd = "ssh -N -L 23306:10.162.0.5:3306 root@34.118.152.152"

# base api url
base_api_url: str = f"http://{host}:{port}/api/v1/funding"

# sample funding request id for testing outreach endpoints
funding_request_id: int = 5994


# health endpoints

# recommender health endpoint
# response -> {"status": True, "db": db_ok, "recommender": rec_status}
get_health_url: str = f"{base_api_url}/healthz/recommender"

# outreach health endpoint
# response -> {"status": True, "db": db_ok}
get_outreach_health_url: str = f"{base_api_url}/healthz/outreach"


# outreach endpoints

# review endpoint
# response -> {"status": True, "preview": email_body, "subject": email_subject}
get_review_url: str = f"{base_api_url}/{funding_request_id}/review"

# send endpoint
# response -> {"status": True, "gmail_msg_id": msg_id, "gmail_thread_id": thr_id}
get_send_url: str = f"{base_api_url}/{funding_request_id}/send"


# recommender endpoints

# reindex endpoint
# params -> force_rebuild: bool = False
# response -> {"status": True, "message": "Reindex scheduled", "job_id": job_id}
post_reindex_url: str = f"{base_api_url}/reindex"

# reindex status endpoint
# response -> {"status": True, "jobs": payload}
get_reindex_status_url: str = f"{base_api_url}/reindex/status"

# reindex status for a specific job endpoint
# response -> {"status": True, "job": job}
get_reindex_status_job_url: str = f"{base_api_url}/reindex/status/{{job_id}}"

# query endpoint
# params -> q: str, k: int = DEFAULT_TOP_K
# response -> {
#     "status": True,
#     "q": q,
#     "suggestions": [
#         {
#             "tag": row["tag"],
#             "domain": row["domain"],
#             "subfield": row["subfield"],
#             "score": round(float(fused_score), 6),
#             "prof_count": len(profs),
#         }
#     ]
# }
get_query_url: str = f"{base_api_url}/query"

# recommend endpoint
# body -> {
#     "tags": List[str],
#     "page_size": Optional[int],
#     "page_number": Optional[int],
#     "institute_name": Optional[str],
#     "country": Optional[str],
#     "professor_name": Optional[str],
#     "department_ids": Optional[List[int]],
#     "expand_related": bool,
#     "domains": Optional[List[str]],
#     "subfields": Optional[List[str]]
# }
# response -> {
#     "status": True,
#     "professor_ids": List[int],
#     "explanations": List[Dict[str, Any]]
# }
post_recommend_url: str = f"{base_api_url}/recommend"


# FLOW OF OUTREACH ->

# step 0
# call outreach health endpoint to check if the API is running
# response -> {"status": True, "db": True}
# if "db" not ready, wait and retry

# step 1
# create an empty record on the funding_requests table with only student_id, professor_id, and match_status fields
# set these fields in the new record of funding_requests table:
student_id: int = 1583
professor_id: int = 21026
match_status: int = 2
status: Literal["incomplete", "complete", "preview", "reviewing", "sent"] = "incomplete"

# step 2
# complete the new record on the funding_requests table with the following fields
# update these fields in the new record of funding_requests table:
research_interest: str = "Low-power analog and mixed-signal integrated circuit design for biomedical applications, including neural interfaces, bio-signal acquisition systems, and energy-efficient sensor front-ends, with a focus on circuit-level innovation and system integration."
paper_title: str = "A Low-Power Chopper-Stabilized Neural Recording Front-End With Adaptive Noise Cancellation"
journal: str = "IEEE Transactions on Biomedical Circuits and Systems"
year: int = 2022
research_connection: str = "This paper presents a low-power analog front-end integrated circuit for biomedical signal acquisition applications. The proposed architecture is designed to achieve high signal fidelity under strict power and area constraints, making it suitable for long-term and wearable biomedical systems. A chopper-stabilized low-noise amplifier is employed to suppress flicker noise and DC offset, while an optimized biasing strategy minimizes power consumption without degrading performance. The front-end is combined with an energy-efficient signal conditioning stage to enhance robustness against process variations and electrode mismatch. Implemented in a standard CMOS technology, the proposed design achieves low input-referred noise and high common-mode rejection while consuming minimal power. Post-layout simulation results demonstrate the effectiveness of the proposed architecture and validate its suitability for low-power biomedical applications."
attachments: dict[str, dict[str, str]] = {"cv": {"path": "users\/documents\/1583_1768500644_CV.pdf", "source": "profile"}}
student_template_ids_stage: dict[str, int] = {"main":9,"reminder1":10,"reminder2":11,"reminder3":12}
student_template_ids_prod: dict[str, int] = {"main": 1, "reminder1": 3, "reminder2": 417, "reminder3": 4}
status: Literal["incomplete", "complete", "preview", "reviewing", "sent"] = "complete"

# step 3
# call review endpoint to get the generated email body and subject
# update these fields in the new record of funding_requests table:
status: Literal["incomplete", "complete", "preview", "reviewing", "sent"] = "preview"
ai_status: Literal["generated", "reviewing_failed", "sent"] = "generated"
ai_response: dict[str, dict[str, int | dict[str, bool | str | None]] | None] = {
    "reviewing": {
        "status": 200,
        "body": {
            "status": True,
            "error": None,
            "subject": "<<RECEIVED_SUBJECT_FROM_REVIEW_ENDPOINT>>",
            "preview": "<<RECEIVED_PREVIEW_FROM_REVIEW_ENDPOINT>>"
        }
    },
    "send": None
}
ai_updated_at: str = "2024-06-01 12:00:00" # current timestamp UTC
email_subject: str = "<<RECEIVED_SUBJECT_FROM_REVIEW_ENDPOINT>>"
email_content: str = "<<RECEIVED_PREVIEW_FROM_REVIEW_ENDPOINT>>"

# step 4
# call send endpoint to send the email
# update these fields in the new record of funding_requests table:
status: Literal["incomplete", "complete", "preview", "reviewing", "sent"] = "sent"
ai_status: Literal["generated", "reviewing_failed", "sent"] = "sent"
ai_response: dict[str, dict[str, int | dict[str, bool | str | None]] | None] = {
    "reviewing": ...,
    "send": {
        "status": 200,
        "body": {
            "status": True,
            "error": None,
            "gmail_msg_id": "<<RECEIVED_GMAIL_MSG_ID_FROM_SEND_ENDPOINT>>",
            "gmail_thread_id": "<<RECEIVED_GMAIL_THREAD_ID_FROM_SEND_ENDPOINT>>"
        }
    }
}
ai_updated_at: str = "2024-06-01 12:05:00" # current timestamp UTC


# FLOW OF RECOMMENDATION ->

# step 0
# call recommender health endpoint to check if the API is running
# response -> {"status": True, "db": True, "recommender": "ready"}
# if "recommender" not ready, wait and retry

# step 1
# call query endpoint to get tag suggestions for a user query
user_query: str = "machine learning"
top_k: int = 10
# received_suggestions: list[dict[str, str | float | int]] = <<RECEIVED_SUGGESTIONS_FROM_QUERY_ENDPOINT>>

# step 2
# call recommend endpoint to get recommended professor ids based on selected tags
selected_tags: list[str] = "<<THREE_RANDOM_TAGS_FROM_RECEIVED_SUGGESTIONS>>".split(",")
page_size: int = 10
page_number: int = 1

# step 3
# call recommend endpoint with filters to get recommended professor ids based on selected tags and filters
selected_tags: list[str] = "<<THREE_RANDOM_TAGS_FROM_RECEIVED_SUGGESTIONS>>".split(",")
institute_name: str = "Concordia University"
country: str = "Canada"
professor_name: str = "AmirMas"
department_ids: list[int] = [2045]
