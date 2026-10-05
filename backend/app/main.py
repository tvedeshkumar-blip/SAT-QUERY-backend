import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("satquery.main")

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Multimodal Remote Sensing Vision-Language Assistant API for ISRO Problem Statement 26167",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,*")
allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import & register route modules
from app.api.routes_health import router as health_router
from app.api.routes_models import router as models_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_reports import router as reports_router
from app.api.routes_chat import router as chat_router

app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(models_router, prefix="/api/v1", tags=["Models"])
app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["Evaluation"])
app.include_router(reports_router, prefix="/api/v1", tags=["Reports"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

@app.get("/")
def root():
    return {
        "title": "SatQuery AI Multimodal Remote Sensing API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SatQuery AI Backend",
        "version": "1.0.0"
    }

