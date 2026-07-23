from backend.database.database import SessionLocal
from backend.database.models import PredictionHistory
from backend.database.database import SessionLocal
from backend.database.models import PredictionHistory

def save_prediction(result):
    db = SessionLocal()

    try:
        prediction = PredictionHistory(
            ticker=result["ticker"],
            company=result["company"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            current_price=result["current_price"]
        )

        db.add(prediction)
        db.commit()

    finally:
        db.close()

def get_prediction_history():
    db = SessionLocal()

    try:
        history = db.query(PredictionHistory)\
                    .order_by(PredictionHistory.created_at.desc())\
                    .all()

        return history

    finally:
        db.close()

def delete_prediction(prediction_id):
    db = SessionLocal()

    try:
        prediction = db.query(PredictionHistory).filter(
            PredictionHistory.id == prediction_id
        ).first()

        if prediction is None:
            return False

        db.delete(prediction)
        db.commit()

        return True

    finally:
        db.close()

def clear_history():
    db = SessionLocal()

    try:
        db.query(PredictionHistory).delete()
        db.commit()

    finally:
        db.close()

def get_prediction_by_id(prediction_id):
    db = SessionLocal()

    try:
        return db.query(PredictionHistory).filter(
            PredictionHistory.id == prediction_id
        ).first()

    finally:
        db.close()