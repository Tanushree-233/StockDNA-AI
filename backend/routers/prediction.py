from fastapi import APIRouter, Depends

from backend.schemas import PredictionRequest
from backend.predictor import predict
from backend.dependencies import get_current_user

router = APIRouter()


@router.post("/predict")
def predict_stock(
    request: PredictionRequest,
    current_user=Depends(get_current_user)
):
    """
    Predict BUY / HOLD / SELL for a stock ticker.
    """

    result = predict(request.ticker)

    return result