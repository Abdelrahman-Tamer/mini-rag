from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import List, Optional

INSECURE_PREDICTABLE_KEYS = {
    "nileguide-admin-secret-2026",
    "admin",
    "secret",
    "password",
    "changeme",
    "123456",
}

class Settings(BaseSettings):

    APP_NAME: str = "Mini RAG"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"  # "development", "staging", "production"

    FILE_ALLOWED_TYPES: list = ["text/plain", "application/pdf", "text/csv", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    FILE_ALLOWED_EXTENSIONS: list = [".txt", ".pdf", ".csv", ".docx"]
    FILE_MAX_SIZE: int = 10
    FILE_DEFAULT_CHUNK_SIZE: int = 512000
    VECTOR_DB_COLLECTION_NAME: str = Field(default="tourism_knowledge_base", validation_alias=AliasChoices("VECTOR_DB_COLLECTION_NAME", "QDRANT_COLLECTION_NAME"))
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
    QDRANT_DB_URL: str = Field(default="", validation_alias=AliasChoices("QDRANT_DB_URL", "QDRANT_URL"))
    DISTANCE_METHOD: Optional[str] = "cosine"
    QDRANT_DB_API_KEY: str = ""

    PRIMARY_LANG: str = "ar"
    DEFAULT_LANG: str = "ar"

    SCORE_THRESHOLD: float = 0.2

    # Internal server-to-server admin service credentials.
    # NEVER defaulted to predictable credentials in production.
    ADMIN_API_KEY: Optional[str] = None
    SYNC_API_KEY: Optional[str] = None
    DEV_ALLOW_DEFAULT_KEY: bool = False

    # Public abuse prevention & rate limiting configuration
    RATE_LIMIT_CHAT_REQUESTS: int = 15  # Max requests per window
    RATE_LIMIT_CHAT_WINDOW_SECONDS: int = 60  # Window size in seconds
    RATE_LIMIT_STREAM_REQUESTS: int = 10  # Max streaming requests per window
    RATE_LIMIT_STREAM_WINDOW_SECONDS: int = 60
    CHAT_MAX_MESSAGE_LENGTH: int = 1000  # Max allowed prompt characters
    REDIS_URL: Optional[str] = None
    REQUIRE_REDIS_RATE_LIMITER: bool = False
    TRUSTED_PROXIES: list = ["127.0.0.1", "::1"]
    TRUSTED_CLOUDFLARE_PROXIES: list = ["127.0.0.1", "::1"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def validate_admin_security_config(settings: Settings) -> None:
    """
    Validates required administrative security credentials at startup.
    Fails closed in production if required keys are missing or set to insecure predictable defaults.
    """
    env = (settings.ENVIRONMENT or "development").strip().lower()
    is_prod = env in ("production", "prod", "staging")

    active_keys = []
    if settings.ADMIN_API_KEY and settings.ADMIN_API_KEY.strip():
        active_keys.append(settings.ADMIN_API_KEY.strip())
    if settings.SYNC_API_KEY and settings.SYNC_API_KEY.strip():
        active_keys.append(settings.SYNC_API_KEY.strip())

    if is_prod:
        if settings.REQUIRE_REDIS_RATE_LIMITER and (not settings.REDIS_URL or not settings.REDIS_URL.strip()):
            raise ValueError(
                "CRITICAL SECURITY CONFIGURATION ERROR: REQUIRE_REDIS_RATE_LIMITER is enabled in production, "
                "but REDIS_URL is unconfigured. A distributed Redis cluster is mandatory."
            )

        if not active_keys:
            raise ValueError(
                "CRITICAL SECURITY CONFIGURATION ERROR: ADMIN_API_KEY or SYNC_API_KEY must be set in production. "
                "The service cannot start without an authentic administrative secret."
            )
        for key in active_keys:
            if key.lower() in INSECURE_PREDICTABLE_KEYS or len(key) < 16:
                raise ValueError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: Predictable or weak administrative key detected in production. "
                    "Use a cryptographically secure random key of at least 16 characters."
                )

def validate_embedding_config(settings: Settings) -> None:
    """
    Validates embedding configuration, backend alignment, and vector dimension compatibility.
    Fails startup if backend, model, URL, or vector dimensions are mismatched.
    """
    backend = (settings.EMBEDDING_BACKEND or "").strip().upper()
    if backend != "OPENAI":
        raise ValueError(
            f"CRITICAL CONFIGURATION ERROR: Unsupported EMBEDDING_BACKEND: '{backend}'. "
            f"Only 'OPENAI' (including Gemini OpenAI compatibility) is supported."
        )

    model_id = (settings.EMBEDDING_MODEL_ID or "").strip().lower()
    model_size = settings.EMBEDDING_MODEL_SIZE
    api_url = (settings.OPENAI_API_URL or "").strip().lower()

    if not model_id:
        raise ValueError("CRITICAL CONFIGURATION ERROR: EMBEDDING_MODEL_ID must not be empty.")

    if not model_size or model_size <= 0:
        raise ValueError(f"CRITICAL CONFIGURATION ERROR: EMBEDDING_MODEL_SIZE must be a positive integer, got {model_size}.")

    # Case A: Google Gemini OpenAI-compatible endpoint
    if "generativelanguage.googleapis.com" in api_url or "googleapis.com" in api_url:
        if "gemini" not in model_id and "text-embedding" not in model_id:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Embedding backend mismatch: OPENAI_API_URL targets Google Gemini ({settings.OPENAI_API_URL}), "
                f"but EMBEDDING_MODEL_ID is set to '{settings.EMBEDDING_MODEL_ID}'. "
                f"Google Gemini endpoints require Gemini embedding models (e.g., 'gemini-embedding-001' or 'text-embedding-004')."
            )
        if "text-embedding-004" in model_id and model_size != 768:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Dimension mismatch: Model '{settings.EMBEDDING_MODEL_ID}' produces 768 dimensions, "
                f"but EMBEDDING_MODEL_SIZE is configured as {model_size}."
            )
        if "gemini-embedding-001" in model_id and (model_size > 3072 or model_size < 128):
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Invalid dimension: Model '{settings.EMBEDDING_MODEL_ID}' supports 128 to 3072 dimensions, "
                f"got EMBEDDING_MODEL_SIZE={model_size}."
            )

    # Case B: Standard OpenAI endpoint (api.openai.com or omitted)
    elif "api.openai.com" in api_url or not api_url:
        if "gemini" in model_id:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Embedding backend mismatch: OPENAI_API_URL targets official OpenAI ({settings.OPENAI_API_URL or 'https://api.openai.com/v1'}), "
                f"but EMBEDDING_MODEL_ID is set to Google Gemini model '{settings.EMBEDDING_MODEL_ID}'. "
                f"To use Gemini models, set OPENAI_API_URL='https://generativelanguage.googleapis.com/v1beta/openai/'."
            )
        if "text-embedding-3-small" in model_id and model_size > 1536:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Dimension mismatch: Model '{settings.EMBEDDING_MODEL_ID}' has max dimension 1536, "
                f"but EMBEDDING_MODEL_SIZE is configured as {model_size}."
            )
        if "text-embedding-ada-002" in model_id and model_size != 1536:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Dimension mismatch: Model '{settings.EMBEDDING_MODEL_ID}' strictly produces 1536 dimensions, "
                f"but EMBEDDING_MODEL_SIZE is configured as {model_size}."
            )
        if "text-embedding-3-large" in model_id and model_size > 3072:
            raise ValueError(
                f"CRITICAL CONFIGURATION ERROR: Dimension mismatch: Model '{settings.EMBEDDING_MODEL_ID}' has max dimension 3072, "
                f"got EMBEDDING_MODEL_SIZE={model_size}."
            )

def get_settings():
    return Settings()
