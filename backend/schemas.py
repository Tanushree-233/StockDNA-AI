from pydantic import BaseModel, EmailStr


class PredictionRequest(BaseModel):
    ticker: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


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