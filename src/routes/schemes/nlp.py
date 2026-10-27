from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    conversation_id: Optional[str] = None
    city: Optional[str] = None
    category: Optional[str] = None

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
