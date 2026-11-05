from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    def set_fallback_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass
    async def generate_text_async(self, prompt: str, chat_history: list = None, max_output_tokens: int = None,
                                  temperature: float = None):
        return self.generate_text(prompt=prompt, chat_history=chat_history or [], max_output_tokens=max_output_tokens, temperature=temperature)
