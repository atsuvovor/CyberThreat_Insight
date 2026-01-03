from llm.runtime import detect_environment, detect_gpu
from llm.backends.mistral_gguf import load as load_mistral
from llm.backends.tinyllama import load as load_tinyllama


class LocalLLM:
    def __init__(self):
        self.env = detect_environment()
        self.has_gpu = detect_gpu()

        if not self.has_gpu:
            try:
                self.llm = load_mistral()
                self.mode = "mistral-gguf"
            except Exception:
                self.llm = load_tinyllama()
                self.mode = "tinyllama"
        else:
            self.llm = load_tinyllama()
            self.mode = "tinyllama"

    def generate(self, prompt, max_tokens=300):
        if self.mode == "mistral-gguf":
            output = self.llm(prompt, max_tokens=max_tokens)
            return output["choices"][0]["text"].strip()

        result = self.llm(prompt)
        return result[0]["generated_text"].strip()
