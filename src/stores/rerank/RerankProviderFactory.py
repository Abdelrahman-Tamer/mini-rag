from .RerankEnums import RerankEnums
from .providers import OllamaRerankProvider, CrossEncoderRerankProvider


class RerankProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == RerankEnums.OLLAMA.value:
            return OllamaRerankProvider(
                api_url=self.config.OPENAI_API_URL,
                api_key=self.config.OPENAI_API_KEY,
            )

        if provider == RerankEnums.CROSS_ENCODER.value:
            return CrossEncoderRerankProvider()

        return None
