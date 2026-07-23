from fastapi import FastAPI

from backend.routers.prediction import router
from backend.database.database import engine
from backend.database.models import Base
from backend.routes.history import router as history_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="StockDNA AI",
    version="1.0.0"
)

app.include_router(router)
app.include_router(history_router)