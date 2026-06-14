from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    history: Optional[List[dict]] = None
