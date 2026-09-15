from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    UniqueConstraint,
    ForeignKey
)
from sqlalchemy.orm import relationship

from backend.database.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
        index=True
    )

    ticker = Column(
        String,
        nullable=False
    )

    company = Column(
        String
    )

    prediction = Column(
        String,
        nullable=False
    )

    confidence = Column(
        Float
    )

    current_price = Column(
        Float
    )

    probabilities = Column(
        String,
        nullable=True
    )

    primary_driver = Column(
        String,
        nullable=True
    )

    internal_percentage = Column(
        Float,
        nullable=True
    )

    external_percentage = Column(
        Float,
        nullable=True
    )

    model_version = Column(
        String,
        default="1.0.0",
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    user = relationship("User", back_populates="predictions")


class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    username = Column(
        String,
        unique=True,
        nullable=False
    )

    email = Column(
        String,
        unique=True,
        nullable=False
    )

    hashed_password = Column(
        String,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    predictions = relationship("PredictionHistory", back_populates="user", cascade="all, delete-orphan")


class StockData(Base):

    __tablename__ = "stock_data"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # Identification
    Date = Column(
        Date,
        nullable=False
    )

    Ticker = Column(
        String,
        nullable=False,
        index=True
    )

    # Market data
    Close = Column(Float)
    High = Column(Float)
    Low = Column(Float)
    Open = Column(Float)
    Volume = Column(Integer)

    # Return / trend features
    Daily_Return = Column(Float)

    SMA20 = Column(Float)
    SMA50 = Column(Float)
    SMA100 = Column(Float)
    SMA200 = Column(Float)

    EMA20 = Column(Float)
    EMA50 = Column(Float)
    EMA100 = Column(Float)

    # Momentum indicators
    RSI = Column(Float)

    MACD = Column(Float)
    MACD_Signal = Column(Float)
    MACD_Histogram = Column(Float)

    # Bollinger Bands
    BB_High = Column(Float)
    BB_Low = Column(Float)
    BB_Middle = Column(Float)

    # Volatility
    ATR = Column(Float)
    Volatility = Column(Float)

    # Lag features
    Lag_Close_1 = Column(Float)
    Lag_Close_3 = Column(Float)
    Lag_Close_5 = Column(Float)

    Lag_Volume_1 = Column(Float)

    # Momentum
    Momentum_5 = Column(Float)
    Momentum_10 = Column(Float)
    Momentum_20 = Column(Float)

    ROC = Column(Float)

    # Volume features
    Volume_Change = Column(Float)
    Volume_MA20 = Column(Float)

    # Price spread features
    High_Low_Spread = Column(Float)
    Open_Close_Spread = Column(Float)

    # Target
    Target_Return = Column(Float)

    Target = Column(
        String,
        index=True
    )

    # One stock record per ticker per trading date
    __table_args__ = (
        UniqueConstraint(
            "Ticker",
            "Date",
            name="uq_stockdata_ticker_date"
        ),
    )