import os

def is_available():
    path = os.getenv(
        "MISTRAL_GGUF_PATH",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    return os.path.exists(path)


def load():
    from llama_cpp import Llama

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


def generate(model, prompt: str, max_tokens: int):
    output = model(prompt, max_tokens=max_tokens)
    return output["choices"][0]["text"]
