from llms_models.interface import LocalLLM
from utils.cleaner import clean_output

class ExecutiveAgent:
    def __init__(self):
        self.llm = LocalLLM()

    def summarize(self, prompt):
        text = self.llm.generate(prompt)
        return clean_output(text)
