from core.causal_engine import CausalEngine
from transformers import pipeline
import os

class FreelancerAgent:
    def __init__(self):
        self.ce = CausalEngine()
        # Optional token if needed; otherwise None
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.llm = pipeline(
            "text-classification",
            model="nlpaueb/legal-bert-base-uncased",
            use_auth_token=hf_token if hf_token else None
        )

    def analyze(self, text: str) -> dict:
        # Split text into clauses by dot, ignore empty ones
        clauses = [clause.strip() for clause in text.split('.') if clause.strip()]
        results = []
        for clause_text in clauses[:3]:  # Analyze only first 3 clauses
            classification = self.llm(clause_text, truncation=True)[0]
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
