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

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, nullable=False)

    email = Column(String, unique=True, nullable=False)

    hashed_password = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)