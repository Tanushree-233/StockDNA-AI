from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.schemas import PredictionRequest, PredictionResponse
from backend.predictor import predict
from backend.dependencies import get_current_user
from backend.database.database import SessionLocal
from backend.database.models import User

router = APIRouter(tags=["Prediction"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_id_from_token_payload(payload: Dict[str, Any], db: Session) -> Optional[int]:
    """Resolves database user ID from JWT token payload."""
    email = payload.get("sub")
    if not email:
        return None
    user = db.query(User).filter(User.email == email).first()
    return user.id if user else None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate stock movement prediction and XAI explanation",
    description="""
    Predicts stock price movement for the 5-day trading horizon:
    - **BUY**: Anticipated positive return (>= +2%)
    - **HOLD**: Neutral / consolidation (-2% to +2%)
    - **SELL**: Anticipated negative return (<= -2%)

    Additionally decomposes SHAP attribution into:
    - **INTERNAL Factors**: Company-specific fundamentals, earnings surprises, and event timing.
    - **EXTERNAL Factors**: Market index returns (NIFTY 50), volatility (India VIX), and technical momentum.
    """,
)
def predict_stock_route(
    request: PredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Authenticated endpoint executing production XGBoost inference and SHAP attribution.
    """
    user_id = get_user_id_from_token_payload(current_user, db)
    result = predict(request.ticker, user_id=user_id)
    return result