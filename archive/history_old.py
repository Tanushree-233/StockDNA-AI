from fastapi import APIRouter
from backend.services.history_service import get_prediction_history
from fastapi import HTTPException
from backend.services.history_service import delete_prediction
from backend.services.history_service import clear_history
from backend.services.history_service import get_prediction_by_id

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/")
def history():

    records = get_prediction_history()

    return [
        {
            "id": item.id,
            "ticker": item.ticker,
            "company": item.company,
            "prediction": item.prediction,
            "confidence": item.confidence,
            "current_price": item.current_price,
            "created_at": item.created_at
        }
        for item in records
    ]

@router.delete("/{prediction_id}")
def delete_history(prediction_id: int):

    success = delete_prediction(prediction_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    return {
        "message": "Prediction deleted successfully."
    }

@router.delete("/")
def delete_all_history():

    clear_history()

    return {
        "message": "All prediction history deleted."
    }

@router.get("/{prediction_id}")
def get_prediction(prediction_id: int):

    prediction = get_prediction_by_id(prediction_id)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    return {
        "id": prediction.id,
        "ticker": prediction.ticker,
        "company": prediction.company,
        "prediction": prediction.prediction,
        "confidence": prediction.confidence,
        "current_price": prediction.current_price,
        "created_at": prediction.created_at
    }