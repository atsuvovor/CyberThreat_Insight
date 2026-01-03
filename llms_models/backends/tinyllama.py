def is_available():
    try:
        import transformers  # noqa
        return True
    except ImportError:
        return False


def load():
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=False
    )


def generate(model, prompt: str, max_tokens: int):
    result = model(prompt, max_new_tokens=max_tokens)
    return result[0]["generated_text"]
