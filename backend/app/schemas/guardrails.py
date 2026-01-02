from pydantic import BaseModel

class GuardrailsOutput(BaseModel):
    is_safe: bool
    reason: str
