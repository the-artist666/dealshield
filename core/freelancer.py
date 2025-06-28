from .causal_engine import CausalEngine
from transformers import pipeline

class FreelancerAgent:
    def __init__(self):
        self.ce = CausalEngine()
        self.llm = pipeline("text-classification", model="nlpaueb/legal-bert-small")

    def analyze(self, text: str) -> dict:
        clauses = self.llm(text, truncation=True)
        results = []
        for clause in clauses[:3]:
            risk = self.ce.predict_risk(clause['text'], clause['label'])
            results.append({
                "clause": clause['text'][:50] + "...",
                "risk": risk,
                "suggestion": "Consider clarifying terms." if risk > 5 else None
            })
        return {
            "results": results,
            "upgrade_prompt": "Unlock full analysis for $7"
        }
