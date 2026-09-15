from typing import Dict, List, Optional, Any
from pydantic import BaseModel, EmailStr


class PredictionRequest(BaseModel):
    ticker: str


class XAIFactorContribution(BaseModel):
    feature: str
    group: str
    shap_value: float
    importance: float
    direction: str
    explanation: str


class XAIResponse(BaseModel):
    primary_driver: str
    internal_percentage: float
    external_percentage: float
    top_internal_factors: List[XAIFactorContribution]
    top_external_factors: List[XAIFactorContribution]
    all_feature_contributions: Optional[List[XAIFactorContribution]] = None


class PredictionResponse(BaseModel):
    ticker: str
    company: str
    timestamp: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    xai: XAIResponse
    model_version: Optional[str] = "1.0.0"
    sector: Optional[str] = None
    latest_news: Optional[List[Any]] = None
    feature_snapshot: Optional[Dict[str, float]] = None


# ------------------------
# Authentication Schemas
# ------------------------

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str   


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str