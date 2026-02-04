# src/agents/letter/schema/generation.py

from typing import Optional, Literal

from pydantic import BaseModel, Field

# Letter type enumeration
LetterType = Literal["sop"]


# Statement of Purpose (SOP) Schema
class LetterGenerationRespSchema(BaseModel):
    # Recipient information
    recipient_name: str = Field(..., description="Full name of the recipient (e.g., Dr. John Smith)")
    recipient_position: str = Field(..., description="Position/title of the recipient (e.g., Professor, Lab Director)")
    recipient_institution: str = Field(..., description="Institution or department name")
    recipient_city: str = Field(..., description="City of the recipient's institution")
    recipient_country: str = Field(..., description="Country of the recipient's institution")

    # Signature/Sender information
    signature_name: str = Field(..., description="Full name of the applicant/sender")
    signature_city: str = Field(..., description="City of the applicant")
    signature_country: str = Field(..., description="Country of the applicant")
    signature_phone: str = Field(..., description="Phone number of the applicant")
    signature_email: str = Field(..., description="Email address of the applicant")
    signature_linkedin: Optional[str] = Field(default=None, description="LinkedIn URL of the applicant")

    # Letter content
    date: str = Field(..., description="Date of the letter (e.g., November 9, 2025 or use current date)")
    salutation: str = Field(..., description="Opening salutation (e.g., Dear Dr. Smith,)")
    body: str = Field(...,
                      description="Full body content of the statement of purpose, including all paragraphs. Should cover: research interest, relevant background/experience, specific fit with the program/lab, and future goals. Use \\n\\n to separate paragraphs in LaTeX.")
    closing_valediction: str = Field(...,
                                     description="Closing phrase (e.g., Sincerely, Thank you for your consideration)")
