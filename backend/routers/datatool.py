from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from pydantic import BaseModel, Field

from data_tool.service import DataToolService
from data_tool.schemas import ProcessedDataBundle, DataType, DataCategory
from data_tool.classifier import classify_text_sentiment
from data_tool.aggregator_engine import process_uploaded_dataset, DataAggregatorError
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


@router.post("/aggregate/upload", summary="Upload and Aggregate Unstructured Dataset")
async def upload_and_aggregate_dataset(file: UploadFile = File(...)):
    """
    Upload an unstructured/semi-structured dataset file (.csv, .tsv, .json, .txt).
    Inspects, cleans, deduplicates, generates labels (domain taxonomy or unsupervised clusters),
    aggregates related records, and produces structured output.
    """
    try:
        max_bytes = 10 * 1024 * 1024  # 10 MB limit
        content_buf = bytearray()
        chunk_size = 1024 * 1024  # 1 MB chunk

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content_buf.extend(chunk)
            if len(content_buf) > max_bytes:
                raise DataAggregatorError("File size exceeds maximum allowed limit of 10 MB.")

        content = bytes(content_buf)
        filename = file.filename or "uploaded_dataset.csv"
        result = process_uploaded_dataset(content, filename)
        return result
    except DataAggregatorError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aggregation error: {str(e)}"
        )
    finally:
        await file.close()


@router.get("/aggregate/sample", summary="Get Sample Unstructured Datasets")
def get_sample_datasets():
    """Provides sample unstructured datasets for instant testing."""
    return {
        "financial_sample": (
            "Date,Close,Volume,Ticker\n"
            "2024-01-15,3850.5,1250000,TCS\n"
            "2024-01-15,3850.5,1250000,TCS\n"  # Duplicate
            "2024-01-16,3890.0,980000,TCS\n"
            "2024-01-17,3910.0,1150000,TCS\n"
            "2024-01-18,3945.2,1420000,TCS\n"
            "2024-01-19,3920.0,1050000,TCS\n"
            "2024-01-22,3980.5,1600000,TCS\n"
        ),
        "non_financial_sample": (
            "ID,Customer_Review,Channel\n"
            "1,The package arrived damaged and delivery was delayed by three days.,Shipping\n"
            "2,Outstanding customer support resolved my billing issue in minutes.,Support\n"
            "3,Terrible product broke within a week and customer service ignored me.,Support\n"
            "4,Super fast shipping and great build quality would buy again.,Shipping\n"
            "5,Delivery courier was polite and package arrived in pristine condition.,Shipping\n"
            "6,Refund request processed quickly great communication from staff.,Support\n"
        ),
    }

