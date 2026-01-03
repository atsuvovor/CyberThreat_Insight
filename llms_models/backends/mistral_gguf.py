from llama_cpp import Llama
import os

def load():
    model_path = os.getenv(
        "MISTRAL_GGUF_PATH",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )

    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        verbose=False
    )
