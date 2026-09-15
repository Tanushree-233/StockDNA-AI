from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from data_tool.service import DataToolService
from data_tool.schemas import ProcessedDataBundle, DataType, DataCategory
from data_tool.classifier import classify_text_sentiment
from config.feature_registry import get_feature_group
from config.settings import get_company_universe

router = APIRouter(prefix="/datatool", tags=["Data Tool"])
service = DataToolService(default_source="api_datatool")


class ProcessDataRequest(BaseModel):
    ticker: Optional[str] = None
    data_type: Optional[DataType] = None
    source: Optional[str] = "api_client"
    records: List[Dict[str, Any]] = Field(description="Raw data rows to clean, normalize, and classify")


class SentimentAnalysisRequest(BaseModel):
    text: str


class FactorClassifyRequest(BaseModel):
    feature_name: str


@router.get("/universe", summary="Get Tracked Equities Universe")
def get_tracked_universe():
    """Expose the authoritative tracked equity universe."""
    df = get_company_universe()
    return df.to_dict(orient="records")


@router.post("/process", response_model=Dict[str, Any], summary="Process Raw Data with Provenance")
def process_data(request: ProcessDataRequest):
    """
    Ingest, clean, normalize, deduplicate, and classify raw multi-source data
    into a standardized ProcessedDataBundle with full provenance metadata.
    Operates completely independently from ML prediction models.
    """
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data records provided."
        )

    try:
        bundle = service.process(
            data=request.records,
            ticker=request.ticker,
            data_type=request.data_type,
            source=request.source,
        )
        if hasattr(bundle, "model_dump"):
            return bundle.model_dump()
        return bundle.dict()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data Tool processing error: {str(e)}"
        )


@router.post("/sentiment", summary="Financial Sentiment Classification")
def analyze_sentiment(request: SentimentAnalysisRequest):
    """Classify financial text sentiment into POSITIVE, NEGATIVE, or NEUTRAL with score."""
    cat, score = classify_text_sentiment(request.text)
    return {
        "text": request.text,
        "sentiment": cat.value,
        "score": score,
        "factor_category": "INTERNAL",
    }


@router.post("/factors/classify", summary="Classify Factor Category (POST)")
def classify_factor_post(request: FactorClassifyRequest):
    """Classify whether a feature is an INTERNAL (company) or EXTERNAL (market) factor."""
    group = get_feature_group(request.feature_name)
    return {
        "feature_name": request.feature_name,
        "category": group,
        "group": group,
    }


@router.get("/factors/classify", summary="Classify Factor Category (GET)")
def classify_factor_get(feature_name: str):
    """Classify whether a feature is an INTERNAL (company) or EXTERNAL (market) factor via query param."""
    group = get_feature_group(feature_name)
    return {
        "feature_name": feature_name,
        "category": group,
        "group": group,
    }
