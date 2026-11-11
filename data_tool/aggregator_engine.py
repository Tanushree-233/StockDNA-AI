"""
Data Aggregator Engine for StockDNA-AI.

Provides an end-to-end, decoupled pipeline for:
1. Ingesting and parsing unstructured/semi-structured files (CSV, TSV, JSON, TXT).
2. Auditing data quality (duplicates, missing values, anomalies, type validity).
3. Dual-mode classification:
   - Domain-Specific Taxonomy: Leverages existing data_tool modules for financial feeds.
   - Unsupervised Automatic Label Generation: For arbitrary non-financial datasets
     using TF-IDF + K-Means clustering, top-keyword extraction, and statistical binning.
4. Grouping and aggregating related records.
5. Emitting standardized structured results suitable for UI rendering and CSV/JSON downloads.
"""

import io
import re
import csv
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_tool.schemas import DataType, DataCategory
from data_tool.classifier import (
    infer_data_type,
    classify_factor_category,
    classify_market_regime,
    classify_earnings_surprise,
    classify_text_sentiment,
)
from data_tool.cleaner import clean_data, clean_market_data, clean_earnings_data, clean_news_data
from data_tool.deduplicator import remove_duplicates


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".json", ".txt"}


class DataAggregatorError(Exception):
    """Custom error for data aggregation failures."""
    pass


def parse_raw_file(
    content: bytes, filename: str
) -> Tuple[pd.DataFrame, str]:
    """
    Parses raw bytes into a pandas DataFrame based on file extension.
    Returns (DataFrame, detected_format).
    """
    if len(content) == 0:
        raise DataAggregatorError("Uploaded file is empty (0 bytes).")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DataAggregatorError(
            f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # Determine extension
    ext = ""
    dot_idx = filename.rfind(".")
    if dot_idx != -1:
        ext = filename[dot_idx:].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise DataAggregatorError(
            f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # Decode text with fallback
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as e:
            raise DataAggregatorError(f"Could not decode text encoding: {str(e)}")

    if not text.strip():
        raise DataAggregatorError("File contains only whitespace or empty text.")

    # 1. JSON Parsing
    if ext == ".json":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                if not parsed:
                    raise DataAggregatorError("JSON array is empty.")
                df = pd.DataFrame(parsed)
            elif isinstance(parsed, dict):
                # Check if it has a records/data key
                nested_found = False
                for key in ["data", "records", "items", "rows", "results"]:
                    if key in parsed and isinstance(parsed[key], list) and len(parsed[key]) > 0:
                        df = pd.DataFrame(parsed[key])
                        nested_found = True
                        break
                if not nested_found:
                    df = pd.DataFrame([parsed])
            else:
                raise DataAggregatorError("JSON root must be an object or an array of objects.")
            return df, "json"
        except json.JSONDecodeError as e:
            raise DataAggregatorError(f"Malformed JSON syntax: {str(e)}")

    # 2. CSV / TSV Parsing
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else None
        try:
            # Sniff delimiter if CSV
            if sep is None:
                sample_lines = "\n".join(text.strip().splitlines()[:10])
                try:
                    dialect = csv.Sniffer().sniff(sample_lines, delimiters=",\t;|")
                    sep = dialect.delimiter
                except Exception:
                    sep = ","

            df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
            if df.empty:
                raise DataAggregatorError("Parsed table contains zero records.")
            return df, ext.replace(".", "")
        except Exception as e:
            raise DataAggregatorError(f"Failed to parse tabular data: {str(e)}")

    # 3. Unstructured TXT Parsing
    if ext == ".txt":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise DataAggregatorError("Text file contains no valid non-empty lines.")

        # Check if text lines look like delimited data
        first_few = lines[:min(5, len(lines))]
        has_tabs = all("\t" in l for l in first_few)
        has_commas = all("," in l for l in first_few)

        if has_tabs or has_commas:
            try:
                sep = "\t" if has_tabs else ","
                df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
                if len(df) > 0 and len(df.columns) > 1:
                    return df, "delimited_txt"
            except Exception:
                pass

        # Treat as unstructured text feed (e.g. news, reviews, logs, notes)
        records = []
        for i, line in enumerate(lines, 1):
            records.append({
                "Record_ID": i,
                "Text_Content": line,
                "Length_Chars": len(line),
                "Word_Count": len(line.split()),
            })
        df = pd.DataFrame(records)
        return df, "unstructured_txt"

    raise DataAggregatorError(f"Unsupported format '{ext}'.")


def audit_and_clean_dataset(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans inconsistencies, removes duplicates, and generates a data-quality audit report.
    """
    raw_rows = len(df)
    raw_cols = len(df.columns)

    if raw_rows == 0:
        raise DataAggregatorError("Dataset has 0 rows.")

    # 1. Clean Column Names
    data = df.copy()
    data.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in data.columns]

    # Drop entirely empty rows and columns
    data = data.dropna(how="all").dropna(axis=1, how="all")
    non_empty_rows = len(data)
    empty_rows_dropped = raw_rows - non_empty_rows

    # 2. Count Missing Cells
    total_cells = max(1, len(data) * len(data.columns))
    missing_cells = int(data.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100.0, 2)

    # 3. Detect and Remove Duplicates
    # Check duplicates across all columns
    duplicates_exact = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    clean_rows = len(data)

    # 4. Standardize Data Types
    for col in data.columns:
        # If column is object, check if it can be coerced to numeric cleanly
        if data[col].dtype == object:
            # Check for numeric strings with commas/currency
            sample = data[col].dropna().astype(str).head(20)
            if sample.str.match(r"^[\$€£₹]?\s*-?[\d,]+(\.\d+)?$").all() and len(sample) > 0:
                cleaned_numeric = (
                    data[col]
                    .astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                )
                data[col] = pd.to_numeric(cleaned_numeric, errors="coerce")

    # 5. Compute Quality Scores
    completeness_score = round(max(0.0, 1.0 - (missing_cells / total_cells)) * 100.0, 2)
    uniqueness_score = round((clean_rows / max(1, raw_rows)) * 100.0, 2)
    validity_score = round(100.0 - min(50.0, missing_pct * 0.5), 2)
    overall_quality_score = round(
        (completeness_score * 0.4) + (uniqueness_score * 0.4) + (validity_score * 0.2), 2
    )

    audit_summary = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "raw_columns": raw_cols,
        "clean_columns": len(data.columns),
        "empty_rows_dropped": empty_rows_dropped,
        "duplicates_removed": duplicates_exact,
        "missing_cells_count": missing_cells,
        "missing_cells_pct": missing_pct,
        "completeness_score_pct": completeness_score,
        "uniqueness_score_pct": uniqueness_score,
        "validity_score_pct": validity_score,
        "overall_quality_score_pct": overall_quality_score,
    }

    return data, audit_summary


def detect_and_classify_dataset(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies either Domain-Specific Financial Taxonomy or Unsupervised Automatic
    Label Generation to classify records.
    """
    data = df.copy()

    # 1. Check if the dataset matches a recognized financial modality
    inferred_type = infer_data_type(data)
    is_financial = inferred_type != DataType.GENERIC

    # Also detect market prices if close and (ticker/volume/open/date) are present
    if not is_financial:
        cols_lower = {str(c).lower().strip() for c in data.columns}
        if "close" in cols_lower and any(k in cols_lower for k in ["ticker", "volume", "open", "symbol"]):
            inferred_type = DataType.MARKET_PRICE
            is_financial = True

    if is_financial:
        # MODE 1: DOMAIN-SPECIFIC TAXONOMY REUSE
        labeling_mode = "domain_taxonomy"
        modality_name = inferred_type.value

        # Apply specific domain pipelines
        if inferred_type == DataType.MARKET_PRICE:
            cleaned_market = clean_market_data(data)
            if not cleaned_market.empty and len(cleaned_market) >= 5:
                data = classify_market_regime(cleaned_market)
            data["Category"] = "EXTERNAL"
            data["Primary_Label"] = data.get("Trend_Regime", "Market Price")

        elif inferred_type == DataType.EARNINGS_EVENT:
            data = clean_earnings_data(data)
            data = classify_earnings_surprise(data)
            data["Category"] = "INTERNAL"
            data["Primary_Label"] = data.get("Earnings_Surprise_Category", "Earnings Event")

        elif inferred_type == DataType.NEWS_SENTIMENT:
            data = clean_news_data(data)
            text_col = "Headline" if "Headline" in data.columns else data.columns[0]
            sentiments = data[text_col].astype(str).apply(classify_text_sentiment)
            data["Sentiment"] = sentiments.apply(lambda x: x[0].value)
            data["Sentiment_Score"] = sentiments.apply(lambda x: x[1])
            data["Category"] = "INTERNAL"
            data["Primary_Label"] = data["Sentiment"]

        else:
            data["Category"] = "INTERNAL" if inferred_type in [DataType.FUNDAMENTAL_STATEMENT] else "EXTERNAL"
            data["Primary_Label"] = inferred_type.value

        # Build label descriptors
        label_counts = data["Primary_Label"].value_counts().to_dict()
        generated_labels = [
            {
                "label": str(lbl),
                "count": int(cnt),
                "percentage": round(cnt / len(data) * 100.0, 2),
                "type": "Domain Taxonomy",
                "criteria": f"Identified via {modality_name} rules",
            }
            for lbl, cnt in label_counts.items()
        ]

        metrics = {
            "silhouette_score": None,
            "clustering_cohesion": "N/A (Predefined Taxonomy)",
        }

    else:
        # MODE 2: UNSUPERVISED AUTOMATIC LABEL GENERATION
        labeling_mode = "unsupervised_clustering"
        modality_name = "Generic / Semi-Structured"

        # Separate text and numeric columns
        text_cols = [
            c for c in data.columns
            if data[c].dtype == object and data[c].nunique() > 1
        ]
        numeric_cols = [
            c for c in data.columns
            if pd.api.types.is_numeric_dtype(data[c])
        ]

        # Case 2A: Dataset has text columns -> TF-IDF + K-Means Clustering + Topic Extraction
        if text_cols and len(data) >= 3:
            # Combine text columns into a single representation
            combined_text = data[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

            # Vectorize
            try:
                tfidf = TfidfVectorizer(
                    max_features=100,
                    stop_words="english",
                    min_df=1,
                    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
                )
                X_tfidf = tfidf.fit_transform(combined_text)
                vocab = np.array(tfidf.get_feature_names_out())

                # Determine cluster count k
                k = min(4, max(2, len(data) // 4))
                if X_tfidf.shape[0] < k:
                    k = max(1, X_tfidf.shape[0])

                if k > 1 and X_tfidf.shape[1] > 0:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    cluster_ids = km.fit_predict(X_tfidf)

                    # Extract top keywords per cluster to form semantic labels
                    cluster_labels_map = {}
                    for c_idx in range(k):
                        center = km.cluster_centers_[c_idx]
                        top_indices = center.argsort()[::-1][:3]
                        if len(top_indices) > 0 and len(vocab) > 0:
                            top_words = [vocab[idx] for idx in top_indices if center[idx] > 0]
                            if top_words:
                                label_name = f"Topic: {', '.join(top_words)}"
                            else:
                                label_name = f"Group {c_idx + 1}"
                        else:
                            label_name = f"Group {c_idx + 1}"
                        cluster_labels_map[c_idx] = label_name

                    data["Primary_Label"] = [cluster_labels_map.get(cid, "Uncategorized") for cid in cluster_ids]
                    data["Category"] = [f"Cluster_{cid + 1}" for cid in cluster_ids]

                    # Silhouette Score
                    try:
                        sil = float(silhouette_score(X_tfidf, cluster_ids))
                        sil_score = round(sil, 4)
                    except Exception:
                        sil_score = None
                else:
                    data["Primary_Label"] = "General Record"
                    data["Category"] = "Uncategorized"
                    sil_score = None

            except Exception:
                data["Primary_Label"] = "General Record"
                data["Category"] = "Uncategorized"
                sil_score = None

            # Add NLP Sentiment on text
            primary_text_col = text_cols[0]
            sentiments = data[primary_text_col].astype(str).apply(classify_text_sentiment)
            data["Sentiment"] = sentiments.apply(lambda x: x[0].value)

            metrics = {
                "silhouette_score": sil_score,
                "clustering_cohesion": "High" if sil_score and sil_score > 0.25 else "Moderate",
            }

        # Case 2B: Purely numeric dataset -> Quantile binning & Anomaly labeling
        elif numeric_cols:
            lead_num = numeric_cols[0]
            # Quantile binning
            try:
                data["Primary_Label"] = pd.qcut(
                    data[lead_num].rank(method="first"),
                    q=min(3, data[lead_num].nunique()),
                    labels=["Low Value", "Medium Value", "High Value"][:min(3, data[lead_num].nunique())]
                ).astype(str)
                data["Category"] = data["Primary_Label"]
            except Exception:
                data["Primary_Label"] = "Standard Record"
                data["Category"] = "General"

            metrics = {
                "silhouette_score": None,
                "clustering_cohesion": "Numeric Quantile Binning",
            }

        else:
            data["Primary_Label"] = "Record"
            data["Category"] = "General"
            metrics = {
                "silhouette_score": None,
                "clustering_cohesion": "Single Class",
            }

        # Check for numeric anomalies via IQR
        for nc in numeric_cols[:2]:
            q1 = data[nc].quantile(0.25)
            q3 = data[nc].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                is_outlier = (data[nc] < (q1 - 1.5 * iqr)) | (data[nc] > (q3 + 1.5 * iqr))
                if is_outlier.any():
                    data["Anomaly_Status"] = np.where(is_outlier, "Outlier", "Normal")
                    break

        label_counts = data["Primary_Label"].value_counts().to_dict()
        generated_labels = [
            {
                "label": str(lbl),
                "count": int(cnt),
                "percentage": round(cnt / len(data) * 100.0, 2),
                "type": "Unsupervised Clustering",
                "criteria": "Dynamically synthesized via TF-IDF & statistical distributions",
            }
            for lbl, cnt in label_counts.items()
        ]

    classification_metadata = {
        "is_financial": is_financial,
        "inferred_modality": modality_name,
        "labeling_mode": labeling_mode,
        "generated_labels": generated_labels,
        "unsupervised_metrics": metrics,
    }

    return data, classification_metadata


def aggregate_dataset_groups(
    df: pd.DataFrame, group_col: str = "Primary_Label"
) -> List[Dict[str, Any]]:
    """
    Groups and aggregates related records by primary category.
    """
    if group_col not in df.columns or df.empty:
        return []

    groups = []
    for grp_val, group_df in df.groupby(group_col, sort=False):
        cnt = len(group_df)
        pct = round(cnt / len(df) * 100.0, 2)

        # Extract sample record
        sample_dict = group_df.head(1).to_dict(orient="records")[0]
        # Clean NaNs in sample dict
        cleaned_sample = {
            k: (None if pd.isna(v) else v)
            for k, v in sample_dict.items()
        }

        # Summary of numeric columns in group
        num_summary = {}
        for col in group_df.select_dtypes(include=[np.number]).columns[:3]:
            mean_val = float(group_df[col].mean())
            num_summary[col] = round(mean_val, 4)

        groups.append({
            "group_name": str(grp_val),
            "record_count": cnt,
            "percentage": pct,
            "numeric_averages": num_summary,
            "sample_record": cleaned_sample,
        })

    return groups


def process_uploaded_dataset(
    content: bytes, filename: str
) -> Dict[str, Any]:
    """
    End-to-end execution pipeline for user-uploaded datasets.
    """
    # 1. Parse File
    df, detected_format = parse_raw_file(content, filename)

    # 2. Clean, Deduplicate & Audit
    cleaned_df, audit_summary = audit_and_clean_dataset(df)

    # 3. Classify (Domain vs Unsupervised)
    classified_df, classification_meta = detect_and_classify_dataset(cleaned_df)

    # 4. Group and Aggregate Related Records
    aggregated_groups = aggregate_dataset_groups(classified_df, group_col="Primary_Label")

    # 5. Format Records for JSON Serialization (convert NaNs to None)
    records = classified_df.replace({np.nan: None}).to_dict(orient="records")

    # Column Schema Metadata
    column_meta = []
    for col in classified_df.columns:
        column_meta.append({
            "name": col,
            "dtype": str(classified_df[col].dtype),
            "non_null_count": int(classified_df[col].notna().sum()),
        })

    return {
        "status": "success",
        "filename": filename,
        "detected_format": detected_format,
        "quality_audit": audit_summary,
        "classification": classification_meta,
        "aggregated_groups": aggregated_groups,
        "column_schema": column_meta,
        "total_records": len(records),
        "records": records,
    }
