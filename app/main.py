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
# Note: When allow_credentials=True, CORS specs forbid "*" in allow_origins.
# We include known frontend origins and use allow_origin_regex for Vercel preview deployments.
default_origins = [
    "https://sat-query-gamma.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
origins_env = os.getenv("CORS_ORIGINS", "")
custom_origins = [o.strip() for o in origins_env.split(",") if o.strip() and o.strip() != "*"]
allowed_origins = list(dict.fromkeys(default_origins + custom_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
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
from app.api.routes_acquisition import router as acquisition_router

app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(models_router, prefix="/api/v1", tags=["Models"])
app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["Evaluation"])
app.include_router(reports_router, prefix="/api/v1", tags=["Reports"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(acquisition_router, prefix="/api/v1", tags=["Acquisition"])

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
