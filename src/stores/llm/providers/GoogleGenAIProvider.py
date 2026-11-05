from ..LLMInterface import LLMInterface
from ..LLMEnums import GoogleGenAIEnums
from google import genai
from google.genai import types
from typing import Any, Optional, List
import asyncio
import concurrent.futures
import logging
import re
import uuid


def _sanitize_error_message(msg: str) -> str:
    """Mask potential credentials or authorization headers from error messages."""
    if not msg:
        return ""
    sanitized = re.sub(r'AIza[0-9A-Za-z-_]{20,}', '[REDACTED_API_KEY]', msg)
    sanitized = re.sub(r'sk-[0-9A-Za-z-_]{20,}', '[REDACTED_API_KEY]', sanitized)
    sanitized = re.sub(r'bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer [REDACTED_TOKEN]', sanitized, flags=re.IGNORECASE)
    return sanitized


class GoogleGenAIProvider(LLMInterface):
    """
    Native Google GenAI text generation provider using official google-genai SDK.
    Strictly isolated: exclusively uses GEMINI_API_KEY and never falls back to OPENAI_API_KEY.
    """

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 4096,
        default_generation_max_output_tokens: int = 2048,
        default_generation_temperature: float = 0.2,
        timeout: float = 25.0,
    ):
        if not api_key or not str(api_key).strip():
            raise ValueError("GEMINI_API_KEY must be provided for GoogleGenAIProvider.")

        self.api_key = str(api_key).strip()
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        self.timeout = timeout

        self.generation_model_id = "gemini-3.5-flash"
        self.fallback_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        # Network retry policy: strictly 1 attempt (zero automatic retries)
        self.retry_options = types.HttpRetryOptions(attempts=1)
        self.http_options = types.HttpOptions(
            timeout=int(self.timeout * 1000),  # ms
            retry_options=self.retry_options,
        )

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=self.http_options,
        )

        self.enums = GoogleGenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_fallback_model(self, model_id: str):
        self.fallback_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        if not text or not isinstance(text, str):
            return str(text or "")
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt),
        }

    def _prepare_payload(self, prompt: Any, chat_history: list = None):
        """
        Parses messages and extracts system instructions and conversational contents.
        """
        if isinstance(prompt, list):
            messages = list(prompt)
        else:
            messages = list(chat_history) if chat_history else []
            messages.append(
                self.construct_prompt(prompt=str(prompt or ""), role=GoogleGenAIEnums.USER.value)
            )

        system_parts = []
        conversation_turns: List[types.Content] = []

        for msg in messages:
            role = (msg.get("role") or "").lower()
            raw_content = msg.get("content") or ""
            if not raw_content:
                continue

            if role in (GoogleGenAIEnums.SYSTEM.value, "system"):
                system_parts.append(raw_content)
            elif role in (GoogleGenAIEnums.ASSISTANT.value, GoogleGenAIEnums.MODEL.value, "assistant", "model"):
                conversation_turns.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=raw_content)],
                    )
                )
            else:
                conversation_turns.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=raw_content)],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # Fallback if no user message exists
        if not conversation_turns:
            conversation_turns = [
                types.Content(role="user", parts=[types.Part.from_text(text="")])
            ]

        return system_instruction, conversation_turns

    async def generate_text_async(
        self,
        prompt: Any,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> Optional[str]:
        """
        Asynchronously generates text using client.aio.models.generate_content without blocking the FastAPI event loop.
        """
        if not self.client:
            self.logger.error("GoogleGenAI client was not set.")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for GoogleGenAI was not set.")
            return None

        system_instruction, contents = self._prepare_payload(prompt, chat_history)

        output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        gen_temp = temperature if temperature is not None else self.default_generation_temperature

        # Validated baseline configuration
        thinking_cfg = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=output_tokens,
            temperature=gen_temp,
            candidate_count=1,
            thinking_config=thinking_cfg,
            http_options=self.http_options,
        )

        corr_id = uuid.uuid4().hex[:8]
        try:
            response = await self.client.aio.models.generate_content(
                model=self.generation_model_id,
                contents=contents,
                config=gen_config,
            )

            if not response or not response.candidates or len(response.candidates) == 0:
                self.logger.warning(f"Native Gemini returned no candidates [corr_id={corr_id}].")
                return None

            cand = response.candidates[0]
            finish_reason = str(cand.finish_reason) if cand.finish_reason else ""

            # Truncation check: MAX_TOKENS must fail closed to refund quota
            if "MAX_TOKENS" in finish_reason:
                self.logger.warning(
                    f"Native Gemini response truncated due to MAX_TOKENS [corr_id={corr_id}]. "
                    f"Returning None to trigger controlled quota refund."
                )
                return None

            # Safety check
            if any(r in finish_reason for r in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII")):
                self.logger.warning(
                    f"Native Gemini response blocked by safety filter: {finish_reason} [corr_id={corr_id}]."
                )
                return None

            text = response.text.strip() if response.text else ""
            if not text:
                self.logger.warning(f"Native Gemini returned empty candidate text [corr_id={corr_id}].")
                return None

            usage = getattr(response, "usage_metadata", None)
            p_tok = getattr(usage, "prompt_token_count", None) if usage else None
            c_tok = getattr(usage, "candidates_token_count", None) if usage else None
            th_tok = getattr(usage, "thoughts_token_count", None) if usage else None
            self.logger.info(
                f"Native Gemini generation succeeded [corr_id={corr_id}]: "
                f"model='{self.generation_model_id}', finish_reason='{finish_reason}', "
                f"prompt_tokens={p_tok}, candidate_tokens={c_tok}, thinking_tokens={th_tok}"
            )
            return text

        except Exception as ex:
            sanitized_err = _sanitize_error_message(str(ex))
            status_code = getattr(ex, "code", getattr(ex, "status_code", None))
            self.logger.error(
                f"Error calling Native Gemini API [corr_id={corr_id}] "
                f"(model='{self.generation_model_id}', status={status_code}): {type(ex).__name__} - {sanitized_err}"
            )
            return None

    def generate_text(
        self,
        prompt: Any,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> Optional[str]:
        """
        Synchronous wrapper for generate_text_async to maintain interface compatibility.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(
                        self.generate_text_async(prompt, chat_history, max_output_tokens, temperature)
                    )
                ).result()
        else:
            return asyncio.run(
                self.generate_text_async(prompt, chat_history, max_output_tokens, temperature)
            )

    def embed_text(self, text: str, document_type: str = None):
        raise NotImplementedError(
            "GoogleGenAIProvider does not provide embeddings. "
            "Embeddings are handled exclusively by OpenAIProvider."
        )

    def close(self):
        if self.client:
            self.client.close()

    async def aclose(self):
        if self.client and hasattr(self.client, "aio"):
            await self.client.aio.aclose()
