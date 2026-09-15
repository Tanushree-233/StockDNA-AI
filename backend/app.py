import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.database import engine
from backend.database.models import Base
from backend.routers.auth import router as auth_router
from backend.routers.prediction import router as prediction_router
from backend.routers.history import router as history_router
from backend.routers.datatool import router as datatool_router
from backend.routers.research import router as research_router

# Ensure tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="StockDNA AI",
    description="Decoupled Three-Layer Financial Intelligence Platform: Data Tool, Prediction Pipeline, and Explainable AI.",
    version="2.0.0"
)

# Enable CORS for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(history_router)
app.include_router(datatool_router)
app.include_router(research_router)

# Mount frontend production build if available
dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")