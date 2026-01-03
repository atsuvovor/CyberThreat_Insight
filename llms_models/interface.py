from llms_models.runtime import select_backend
from llms_models.backends import mistral_gguf, tinyllama


class LocalLLM:
    """
    Unified interface for all local LLM backends.
    """

    def __init__(self):
        self.backend = select_backend()

        if self.backend == "mistral-gguf":
            self.model = mistral_gguf.load()
            self.generator = mistral_gguf.generate

        elif self.backend == "tinyllama":
            self.model = tinyllama.load()
            self.generator = tinyllama.generate

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        return self.generator(self.model, prompt, max_tokens).strip()
