import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.database import SessionLocal
from backend.database.models import User
from backend.dependencies import get_current_user
from backend.services.history_service import (
    get_prediction_history,
    get_prediction_by_id,
    delete_prediction,
    clear_history,
)

router = APIRouter(prefix="/history", tags=["History"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_id_from_payload(payload: Dict[str, Any], db: Session) -> Optional[int]:
    """Resolve database user ID from token sub (email)."""
    email = payload.get("sub")
    if not email:
        return None
    user = db.query(User).filter(User.email == email).first()
    return user.id if user else None


def format_history_record(item) -> Dict[str, Any]:
    """Helper to serialize a PredictionHistory record."""
    probs = {}
    if item.probabilities:
        try:
            probs = json.loads(item.probabilities)
        except Exception:
            probs = {}

    return {
        "id": item.id,
        "user_id": item.user_id,
        "ticker": item.ticker,
        "company": item.company,
        "prediction": item.prediction,
        "confidence": item.confidence,
        "current_price": item.current_price,
        "probabilities": probs,
        "primary_driver": item.primary_driver,
        "internal_percentage": item.internal_percentage,
        "external_percentage": item.external_percentage,
        "model_version": item.model_version,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


@router.get("/")
def list_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve prediction history for the authenticated user."""
    user_id = get_user_id_from_payload(current_user, db)
    records = get_prediction_history(user_id=user_id)
    return [format_history_record(item) for item in records]


@router.get("/{prediction_id}")
def get_single_prediction(
    prediction_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve details of a specific prediction owned by the authenticated user."""
    user_id = get_user_id_from_payload(current_user, db)
    prediction = get_prediction_by_id(prediction_id, user_id=user_id)

    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction record not found or access denied."
        )

    return format_history_record(prediction)


@router.delete("/{prediction_id}")
def delete_single_prediction(
    prediction_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a single prediction record owned by the authenticated user."""
    user_id = get_user_id_from_payload(current_user, db)
    success = delete_prediction(prediction_id, user_id=user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction record not found or access denied."
        )

    return {"message": "Prediction record deleted successfully."}


@router.delete("/")
def clear_user_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clear all prediction history belonging to the authenticated user."""
    user_id = get_user_id_from_payload(current_user, db)
    count = clear_history(user_id=user_id)

    return {"message": f"Deleted {count} prediction records for current user."}
