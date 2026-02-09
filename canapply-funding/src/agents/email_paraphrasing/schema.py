from pydantic import BaseModel, Field


class RespSchema(BaseModel):
    email: str = Field(..., description="Drafted or revised email content.")
