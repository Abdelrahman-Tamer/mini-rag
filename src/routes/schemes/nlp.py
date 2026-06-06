from pydantic import BaseModel
from typing import Optional, List

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5

class ChatRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
    city: Optional[str] = None
    doc_type: Optional[str] = None

class SourceInfo(BaseModel):
    document_name: str
    city: Optional[str] = None
    doc_type: Optional[str] = None
    score: float
    excerpt: str

class ChatResponse(BaseModel):
    signal: str
    answer: str
    sources: List[SourceInfo]
    related_questions: List[str]
