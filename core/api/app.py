from fastapi import FastAPI, Request
from pydantic import BaseModel
from core.freelancer import FreelancerAgent

app = FastAPI()
agent = FreelancerAgent()

class ContractInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_contract(input: ContractInput):
    return agent.analyze(input.text)

@app.get("/health")
async def health_check():
    return {"status": "Freelancer tier active"}
