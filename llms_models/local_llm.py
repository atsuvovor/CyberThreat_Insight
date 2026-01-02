import os

def load_local_llm():
    """
    Auto-select best local LLM based on environment.
    Priority:
    1. Mistral 7B GGUF (llama.cpp)
    2. TinyLlama (Transformers)
    """

    # ---- Option 1: Mistral GGUF (BEST) ----
    mistral_path = os.getenv("MISTRAL_GGUF_PATH", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

    if os.path.exists(mistral_path):
        from llama_cpp import Llama
        return Llama(
            model_path=mistral_path,
            n_ctx=4096,
            n_threads=8,
            temperature=0.1,
            verbose=False
        ), "mistral-gguf"

    # ---- Option 2: TinyLlama fallback ----
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=False
    )

    return pipe, "tinyllama"
