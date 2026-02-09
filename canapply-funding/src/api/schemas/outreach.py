# src/api/schemas/logic.py

from typing import Optional
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    status: bool = Field(
        ...,
        description="True when the operation succeeded; False on error.",
        examples=[True],
    )
    error: Optional[str] = Field(
        None,
        description="Present only when status=False.",
        examples=["not found"],
    )

    model_config = {"extra": "allow"}   # replaces orm_mode=True if you used it


class ReviewResponse(BaseResponse):
    subject: Optional[str] = Field(
        None,
        description="Email subject (only on success).",
        examples=["Prospective student – Sharareh Salmanizadeh"],
    )
    preview: Optional[str] = Field(
        None,
        description="First 400 chars of the email (only on success).",
        examples=["Dear Prof. Smith,<br><br>I recently read your 2023 Nature..."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Draft built",
                    "value": {
                        "status": True,
                        "subject": "Prospective student – Sharareh Salmanizadeh",
                        "preview": "Dear Prof. Smith,<br><br>I recently read your 2023..."
                    },
                },
                {
                    "summary": "Professor missing",
                    "value": {
                        "status": False,
                        "error": "Professor not found",
                    },
                },
            ]
        }
    }


class SendResponse(BaseResponse):
    gmail_msg_id: Optional[str] = Field(
        None,
        description="Gmail message ID.",
        examples=["18b3e9b4c5d6e7f8"],
    )
    gmail_thread_id: Optional[str] = Field(
        None,
        description="Gmail thread ID.",
        examples=["18b3e9b4c5d6e7f8"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Mail sent",
                    "value": {
                        "status": True,
                        "gmail_msg_id": "18b3e9b4c5d6e7f8",
                        "gmail_thread_id": "18b3e9b4c5d6e7f8"
                    },
                },
                {
                    "summary": "Duplicate send attempt",
                    "value": {
                        "status": False,
                        "error": "Already sent",
                        "gmail_msg_id": "18b3e9b4c5d6e7f8",
                    },
                },
            ]
        }
    }
