from pydantic import BaseModel

class PredictionRequest(BaseModel):
    ticker: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict