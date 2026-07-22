from fastapi import FastAPI

from backend.routers.prediction import router

app = FastAPI(
    title="StockDNA AI",
    version="1.0.0"
)

app.include_router(router)