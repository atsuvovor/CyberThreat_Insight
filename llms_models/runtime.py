import os
import sys

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
