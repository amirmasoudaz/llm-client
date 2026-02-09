from pydantic import BaseModel


class RespSchema(BaseModel):
    email: str
