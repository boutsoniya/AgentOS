from pydantic import BaseModel

class Evidence(BaseModel):
    title: str
    locator: str = ''
    excerpt: str = ''
    score: float = 0.0

class Claim(BaseModel):
    text: str
    support_score: float = 0.0
    evidence_ids: list[str] = []
