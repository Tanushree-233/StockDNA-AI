import json
from typing import Any, Dict, List, Optional
from backend.database.database import SessionLocal
from backend.database.models import PredictionHistory


def save_prediction(result: Dict[str, Any], user_id: Optional[int] = None) -> PredictionHistory:
    """
    Persist a model prediction with user ownership and decision attribution metadata.
    """
    db = SessionLocal()
    try:
        probs = result.get("probabilities", {})
        probs_str = json.dumps(probs) if isinstance(probs, dict) else str(probs)

        prediction = PredictionHistory(
            user_id=user_id,
            ticker=result["ticker"],
            company=result.get("company", ""),
            prediction=result["prediction"],
            confidence=float(result.get("confidence", 0.0)),
            current_price=float(result.get("current_price", 0.0)),
            probabilities=probs_str,
            primary_driver=result.get("primary_driver", "EXTERNAL"),
            internal_percentage=float(result.get("internal_percentage", 50.0)),
            external_percentage=float(result.get("external_percentage", 50.0)),
            model_version=result.get("model_version", "1.0.0"),
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    finally:
        db.close()


def get_prediction_history(user_id: Optional[int] = None) -> List[PredictionHistory]:
    """
    Retrieve prediction history, filtered by user_id if provided.
    """
    db = SessionLocal()
    try:
        query = db.query(PredictionHistory)
        if user_id is not None:
            query = query.filter(PredictionHistory.user_id == user_id)
        history = query.order_by(PredictionHistory.created_at.desc()).all()
        return history
    finally:
        db.close()


def get_prediction_by_id(prediction_id: int, user_id: Optional[int] = None) -> Optional[PredictionHistory]:
    """
    Retrieve a specific prediction by ID, with user ownership check.
    """
    db = SessionLocal()
    try:
        query = db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id)
        if user_id is not None:
            query = query.filter(PredictionHistory.user_id == user_id)
        return query.first()
    finally:
        db.close()


def delete_prediction(prediction_id: int, user_id: Optional[int] = None) -> bool:
    """
    Delete a single prediction record with user ownership check.
    """
    db = SessionLocal()
    try:
        query = db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id)
        if user_id is not None:
            query = query.filter(PredictionHistory.user_id == user_id)
        prediction = query.first()

        if prediction is None:
            return False

        db.delete(prediction)
        db.commit()
        return True
    finally:
        db.close()


def clear_history(user_id: Optional[int] = None) -> int:
    """
    Clear prediction history (for a specific user, or all if user_id is None).
    Returns count of deleted rows.
    """
    db = SessionLocal()
    try:
        query = db.query(PredictionHistory)
        if user_id is not None:
            query = query.filter(PredictionHistory.user_id == user_id)
        count = query.delete(synchronize_session=False)
        db.commit()
        return count
    finally:
        db.close()