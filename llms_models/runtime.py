import os
import sys
from llms_models.backends import mistral_gguf, tinyllama

def detect_environment():
    if "google.colab" in sys.modules:
        return "colab"
    if "streamlit" in sys.modules:
        return "streamlit"
    if "ipykernel" in sys.modules:
        return "jupyter"
    return "local"


def detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False



def select_backend():
    """
    Priority-based backend selection.
    """
    if mistral_gguf.is_available():
        return "mistral-gguf"

    if tinyllama.is_available():
        return "tinyllama"

    raise RuntimeError("No compatible local LLM backend found")

