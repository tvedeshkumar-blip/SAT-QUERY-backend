
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env when available.
# Render environment variables will also be available through os.getenv().
load_dotenv()

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("satquery.main")

# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="SatQuery AI Backend",
    description=(
        "Agentic Multimodal Remote Sensing Vision-Language Assistant API "
        "for ISRO Problem Statement 26167"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# -------------------------------------------------------------------
# CORS configuration
# -------------------------------------------------------------------

default_origins = (
    "http://localhost:3000,"
    "http://localhost:5173,"
    "*"
)

origins_raw = os.getenv("CORS_ORIGINS", default_origins)

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in origins_raw.split(",")
    if origin.strip()
]

# Allow Vercel preview and production subdomains.
# Add your exact custom frontend domain through CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Import and register API routers
# -------------------------------------------------------------------

from app.api.routes_health import router as health_router
from app.api.routes_models import router as models_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_reports import router as reports_router
from app.api.routes_chat import router as chat_router
from app.api.routes_acquisition import router as acquisition_router

API_PREFIX = "/api/v1"

app.include_router(
    health_router,
    prefix=API_PREFIX,
    tags=["Health"],
)

app.include_router(
    models_router,
    prefix=API_PREFIX,
    tags=["Models"],
)

app.include_router(
    analysis_router,
    prefix=API_PREFIX,
    tags=["Analysis"],
)

app.include_router(
    evaluation_router,
    prefix=API_PREFIX,
    tags=["Evaluation"],
)

app.include_router(
    reports_router,
    prefix=API_PREFIX,
    tags=["Reports"],
)

app.include_router(
    chat_router,
    prefix=API_PREFIX,
    tags=["Chat"],
)

app.include_router(
    acquisition_router,
    prefix=API_PREFIX,
    tags=["Acquisition"],
)

logger.info("SatQuery AI routes registered successfully.")

# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "title": "SatQuery AI Multimodal Remote Sensing API",
        "status": "online",
        "docs": "/docs",
        "version": app.version,
    }

# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "SatQuery AI Backend",
        "version": app.version,
    }

# -------------------------------------------------------------------
# Startup event
# -------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("SatQuery AI Backend starting up.")
    logger.info("API documentation available at /docs")
