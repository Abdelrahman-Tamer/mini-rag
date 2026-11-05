from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, GoogleGenAIProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        provider_clean = (provider or "").strip().lower()

        if provider_clean in (LLMEnums.OPENAI.value.lower(), "openai_compatible"):
            timeout = getattr(self.config, "GENERATION_TIMEOUT", 20.0)
            fallback_model_id = getattr(self.config, "FALLBACK_MODEL_ID", None)
            return OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters = self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens = self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature = self.config.GENERATION_DEFAULT_TEMPERATURE,
                timeout = timeout,
                fallback_model_id = fallback_model_id,
            )

        elif provider_clean in (LLMEnums.NATIVE_GEMINI.value.lower(), "native_gemini", "google_genai"):
            gemini_key = getattr(self.config, "GEMINI_API_KEY", None)
            if not gemini_key or not str(gemini_key).strip():
                raise ValueError("GEMINI_API_KEY is required when LLM_GENERATION_PROVIDER is set to 'native_gemini'.")
            timeout = getattr(self.config, "GENERATION_TIMEOUT", 25.0)
            return GoogleGenAIProvider(
                api_key = str(gemini_key).strip(),
                default_input_max_characters = getattr(self.config, "INPUT_DEFAULT_MAX_CHARACTERS", 4096),
                default_generation_max_output_tokens = getattr(self.config, "GENERATION_DEFAULT_MAX_TOKENS", 2048),
                default_generation_temperature = getattr(self.config, "GENERATION_DEFAULT_TEMPERATURE", 0.2),
                timeout = timeout,
            )

        return None
