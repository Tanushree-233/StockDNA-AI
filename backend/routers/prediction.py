from fastapi import APIRouter

from backend.schemas import PredictionRequest
from backend.predictor import predict

router = APIRouter()


@router.post("/predict")
def predict_stock(request: PredictionRequest):
    """
    Predict BUY / HOLD / SELL for a stock ticker.
    """

    result = predict(request.ticker)

    return result