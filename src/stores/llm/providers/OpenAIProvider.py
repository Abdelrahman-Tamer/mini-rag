from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
from typing import Any
import logging
import uuid
import re


def _sanitize_error_message(msg: str) -> str:
    """Mask potential credentials or authorization headers from error messages."""
    if not msg:
        return ""
    sanitized = re.sub(r'AIza[0-9A-Za-z-_]{20,}', '[REDACTED_API_KEY]', msg)
    sanitized = re.sub(r'sk-[0-9A-Za-z-_]{20,}', '[REDACTED_API_KEY]', sanitized)
    sanitized = re.sub(r'bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer [REDACTED_TOKEN]', sanitized, flags=re.IGNORECASE)
    return sanitized


def _is_transient_error(ex: Exception) -> bool:
    """
    Determine if an upstream exception represents a transient, recoverable failure.
    Transient (eligible for fallback): HTTP 429, 500, 502, 503, 504, APITimeoutError, APIConnectionError.
    Permanent client errors (must NOT retry or trigger fallback): HTTP 400, 401, 403.
    """
    status_code = getattr(ex, "status_code", None)
    if status_code in (400, 401, 403):
        return False

    err_type_name = type(ex).__name__
    if err_type_name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError"):
        return False

    if err_type_name in ("APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"):
        return True

    if status_code in (429, 500, 502, 503, 504):
        return True

    msg = str(ex).lower()
    if any(k in msg for k in ("invalid_argument", "invalid api key", "unauthorized", "permission_denied", "permission denied")):
        return False
    if any(k in msg for k in ("503", "502", "500", "504", "429", "high demand", "timed out", "timeout", "unavailable", "rate limit")):
        return True

    return False

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                    default_input_max_characters: int=1000,
                    default_generation_max_output_tokens: int=1000,
                    default_generation_temperature: float=0.1,
                    timeout: float = 7.0):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        self.timeout = timeout

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(
            api_key = self.api_key,
            base_url = self.api_url if self.api_url and len(self.api_url) else None,
            timeout = self.timeout,
            max_retries = 0
        )

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        if not text or not isinstance(text, str):
            return str(text or "")
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: Any, chat_history: list = None, max_output_tokens: int = None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        if isinstance(prompt, list):
            messages = list(prompt)
        else:
            messages = list(chat_history) if chat_history else []
            messages.append(
                self.construct_prompt(prompt=str(prompt or ""), role=OpenAIEnums.USER.value)
            )

        corr_id = uuid.uuid4().hex[:8]

        # Verified dual-model failover strategy: primary gemini-3.5-flash-lite <-> verified fallback gemini-3.5-flash
        models_to_try = [self.generation_model_id]
        if self.generation_model_id == "gemini-3.5-flash-lite":
            models_to_try.append("gemini-3.5-flash")
        elif self.generation_model_id == "gemini-3.5-flash":
            models_to_try.append("gemini-3.5-flash-lite")
        elif "gemini" in (self.generation_model_id or "").lower():
            # If an unverified Gemini model is set (e.g. gemini-3.8-flash), fall back to verified gemini-3.5-flash-lite
            models_to_try.append("gemini-3.5-flash-lite")

        for idx, target_model in enumerate(models_to_try):
            try:
                response = self.client.chat.completions.create(
                    model = target_model,
                    messages = messages,
                    max_tokens = max_output_tokens,
                    temperature = temperature
                )
                if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
                    if idx > 0:
                        self.logger.info(
                            f"Model fallback succeeded [corr_id={corr_id}]: primary '{models_to_try[0]}' -> fallback '{target_model}'"
                        )
                    return response.choices[0].message.content
            except Exception as ex:
                sanitized_err = _sanitize_error_message(str(ex))
                is_transient = _is_transient_error(ex)

                # Non-retryable client errors (400, 401, 403) must NOT trigger fallback
                if not is_transient:
                    self.logger.error(
                        f"Non-retryable client error calling OpenAI/Gemini API [corr_id={corr_id}] (model='{target_model}'): {type(ex).__name__} - {sanitized_err}"
                    )
                    return None

                # Transient errors: attempt verified fallback if candidate available
                if idx < len(models_to_try) - 1:
                    self.logger.warning(
                        f"Transient failure on model '{target_model}' [corr_id={corr_id}]: {type(ex).__name__} - {sanitized_err}. Attempting verified fallback to '{models_to_try[idx+1]}'..."
                    )
                else:
                    self.logger.error(
                        f"All generation models failed [corr_id={corr_id}]. Last attempt '{target_model}': {type(ex).__name__} - {sanitized_err}"
                    )

        return None

    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        create_kwargs = {
            "model": self.embedding_model_id,
            "input": text,
        }
        if self.embedding_size and "ada" not in (self.embedding_model_id or "").lower():
            create_kwargs["dimensions"] = self.embedding_size

        corr_id = uuid.uuid4().hex[:8]
        try:
            response = self.client.embeddings.create(**create_kwargs)
        except Exception as ex:
            if "dimensions" in str(ex).lower() and "dimensions" in create_kwargs:
                self.logger.warning(
                    f"Embedding API rejected 'dimensions' argument [corr_id={corr_id}]; falling back without it: {_sanitize_error_message(str(ex))}"
                )
                create_kwargs.pop("dimensions", None)
                try:
                    response = self.client.embeddings.create(**create_kwargs)
                except Exception as ex2:
                    self.logger.error(
                        f"Error calling OpenAI/Gemini embeddings API on retry [corr_id={corr_id}]: {type(ex2).__name__} - {_sanitize_error_message(str(ex2))}"
                    )
                    return None
            else:
                self.logger.error(
                    f"Error calling OpenAI/Gemini embeddings API [corr_id={corr_id}]: {type(ex).__name__} - {_sanitize_error_message(str(ex))}"
                )
                return None

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error(f"Error while embedding text with OpenAI [corr_id={corr_id}]")
            return None

        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
