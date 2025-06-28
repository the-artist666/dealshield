from core.causal_engine import CausalEngine
from transformers import pipeline

class FreelancerAgent:
    def __init__(self):
        self.ce = CausalEngine()
        self.llm = pipeline("text-classification", model="nlpaueb/legal-bert-small")

    def analyze(self, text: str) -> dict:
        # Naively split text into clauses by period
        clauses = [clause.strip() for clause in text.split('.') if clause.strip()]
        results = []
        for clause_text in clauses[:3]:  # analyze first 3 clauses
            classification = self.llm(clause_text, truncation=True)[0]  # returns dict with 'label' and 'score'
            risk = self.ce.predict_risk(clause_text, classification['label'])
            results.append({
                "clause": clause_text[:50] + ("..." if len(clause_text) > 50 else ""),
                "risk": risk,
                "suggestion": "Consider clarifying terms." if risk > 5 else None
            })
        return {
            "results": results,
            "upgrade_prompt": "Unlock full analysis for $7"
        }
