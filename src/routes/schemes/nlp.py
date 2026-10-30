from pydantic import BaseModel, root_validator
from typing import Optional

class ChatRequest(BaseModel):
    text: Optional[str] = None
    prompt: Optional[str] = None
    limit: Optional[int] = 5
    conversation_id: Optional[str] = None
    city: Optional[str] = None
    category: Optional[str] = None

    @root_validator(pre=True)
    def normalize_text_or_prompt(cls, values):
        if not isinstance(values, dict):
            return values
        val = values.get("text") or values.get("prompt")
        if val is not None:
            values["text"] = str(val)
        return values

class SyncActivityRequest(BaseModel):
    activity_id: int
    title: str
    city: Optional[str] = "Egypt"
    category: Optional[str] = "General"
    price: Optional[str] = "0.0 USD"
    description: Optional[str] = ""
    image_url: Optional[str] = ""
    link: Optional[str] = ""
    is_active: Optional[bool] = True
    action: Optional[str] = "upsert"
