#ai_agent/orchestrator.py

class BaseAgent:
    def run(self, data):
        raise NotImplementedError


class ValidationAgent(BaseAgent):
    def run(self, data):
        return "Validation passed"


class ExecutiveInsightAgent(BaseAgent):
    def run(self, data):
        return "Executive summary generated"


class SOCSupportAgent(BaseAgent):
    def run(self, data):
        return "SOC alert dispatched"


class AgentOrchestrator:
    def __init__(self):
        self.agents = [
            ValidationAgent(),
            ExecutiveInsightAgent(),
            SOCSupportAgent()
        ]

    def execute(self, data):
        results = {}
        for agent in self.agents:
            results[agent.__class__.__name__] = agent.run(data)
        return results
