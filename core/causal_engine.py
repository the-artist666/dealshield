import torch
from typing import Literal

class CausalEngine:
    def __init__(self, tier: Literal["freelancer"] = "freelancer"):
        self.graph = self._load_graph(tier)
        self.var_names = ["payment", "liability", "termination", "scope", "dispute"]

    def _load_graph(self, tier):
        if tier == "freelancer":
            return torch.tensor([
                [0, 0.2, 0, 0, 0],
                [0, 0, 0.3, 0.1, 0],
                [0, 0, 0, 0, 0.2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ])

    def predict_risk(self, clause: str, clause_type: str) -> float:
        var_idx = self.var_names.index(clause_type)
        base_risk = min(10, len(clause.split()) / 5)
        for p_idx in torch.where(self.graph[:, var_idx] > 0)[0]:
            base_risk += self.graph[p_idx, var_idx].item() * 2
        return round(base_risk, 1)
