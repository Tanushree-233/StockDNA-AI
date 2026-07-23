from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

from backend.database.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    ticker = Column(String)

    company = Column(String)

    prediction = Column(String)

    confidence = Column(Float)

    current_price = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)