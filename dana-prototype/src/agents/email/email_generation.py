# src/agents/schemas/email_generation.py

from typing import Optional
from pydantic import BaseModel, Field


class EmailGenerationRespSchema(BaseModel):
    """Schema for professor outreach emails."""

    # Recipient information
    recipient_name: str = Field(..., description="Full name of the recipient professor (e.g., Dr. John Smith)")
    recipient_position: Optional[str] = Field(default=None,
                                              description="Position/title of the recipient (e.g., Professor, Associate Professor)")
    recipient_institution: Optional[str] = Field(default=None, description="Institution or department name")

    # Email structure
    subject: str = Field(...,
                         description="Email subject line - should be specific, compelling, and professional (e.g., 'Research Collaboration: Neural Architecture Search')")
    greeting: str = Field(..., description="Opening salutation (e.g., 'Dear Dr. Smith,' or 'Dear Professor Smith,')")
    body: str = Field(...,
                      description="Full email body content. Should include: brief introduction, research interests/background, specific connection to professor's work, collaboration proposal or meeting request. Keep concise (150-300 words). Use \\n\\n to separate paragraphs.")
    closing: str = Field(...,
                         description="Closing phrase (e.g., 'Best regards,' 'Sincerely,' 'Thank you for your consideration,')")

    # Signature/Sender information
    signature_name: str = Field(..., description="Full name of the sender/applicant")
    signature_email: str = Field(..., description="Email address of the sender")
    signature_phone: Optional[str] = Field(default=None, description="Phone number of the sender (optional)")
    signature_linkedin: Optional[str] = Field(default=None, description="LinkedIn URL of the sender (optional)")
    signature_website: Optional[str] = Field(default=None, description="Personal website or portfolio URL (optional)")
