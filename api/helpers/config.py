from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):

    APP_NAME: str = "Mini RAG"
    APP_VERSION: str = "0.1.0"

    FILE_ALLOWED_TYPES: list = ["text/plain", "application/pdf", "text/csv", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    FILE_ALLOWED_EXTENSIONS: list = [".txt", ".pdf", ".csv", ".docx"]
    FILE_MAX_SIZE: int = 10
    FILE_DEFAULT_CHUNK_SIZE: int = 512000
    VECTOR_DB_COLLECTION_NAME: str = "tourism_knowledge_base"
    DEFAULT_CITY: str = "unknown"
    DEFAULT_DOC_TYPE: str = "general"

    MONGODB_URL: str = ""
    MONGODB_DATABASE: str = "mini-rag"

    GENERATION_BACKEND: str = "OPENAI"
    EMBEDDING_BACKEND: str = "OPENAI"

    OPENAI_API_KEY: Optional[str] = ""
    OPENAI_API_URL: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GENERATION_MODEL_ID: Optional[str] = "gemini-3.5-flash-lite"
    EMBEDDING_MODEL_ID: Optional[str] = "gemini-embedding-001"
    EMBEDDING_MODEL_SIZE: Optional[int] = 3072
    INPUT_DEFAULT_MAX_CHARACTERS: Optional[int] = 4096
    GENERATION_DEFAULT_MAX_TOKENS: Optional[int] = 1000
    GENERATION_DEFAULT_TEMPERATURE: Optional[float] = 0.2

    VECTOR_DB_BACKEND: str = "QDRANT"
    QDRANT_DB_URL: str = ""
    DISTANCE_METHOD: Optional[str] = "cosine"
    QDRANT_DB_API_KEY: str = ""

    PRIMARY_LANG: str = "ar"
    DEFAULT_LANG: str = "ar"

    SCORE_THRESHOLD: float = 0.2

    class Config:
        env_file = ".env"
        extra = "ignore"

def get_settings():
    return Settings()
