import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

from app.evaluation.schemas import BenchmarkSuiteResponse, SyntheticValidationResponse
from app.evaluation.persistence import load_persisted_benchmark_results
from app.evaluation.synthetic_validator import SyntheticPipelineValidator

logger = logging.getLogger("satquery.routes_evaluation")
router = APIRouter()

@router.get("/evaluation", response_model=BenchmarkSuiteResponse)
def get_benchmark_evaluations():
    """
    Returns benchmark evaluations.
    CRITICAL POLICY: Never returns fabricated numbers. Only returns results from
    verified, completed benchmark runs loaded from disk.
    """
    persisted_results = load_persisted_benchmark_results()
    
    if not persisted_results:
        return BenchmarkSuiteResponse(
            status="not_evaluated",
            results=[],
            message="Benchmark results will appear after actual evaluation on mounted datasets."
        )

    return BenchmarkSuiteResponse(
        status="completed",
        results=persisted_results,
        message="Verified benchmark evaluation results loaded from persisted runs."
    )

@router.post("/evaluation/synthetic-validation")
def run_synthetic_validation(payload: Optional[Dict[str, Any]] = None):
    """
    Executes local synthetic pipeline validation on deterministic test patterns.
    Explicitly labeled: NOT official ISRO evaluation.
    """
    seed = 42
    if payload and "seed" in payload:
        try:
            seed = int(payload["seed"])
        except (ValueError, TypeError):
            seed = 42

    result = SyntheticPipelineValidator.run_validation(seed=seed)
    return result

@router.post("/evaluation/isro-run")
def legacy_isro_run_alias(payload: Optional[Dict[str, Any]] = None):
    """
    Backward-compatibility route mapped to synthetic pipeline validation with explicit disclaimers.
    """
    return run_synthetic_validation(payload)

@router.post("/evaluation/official-run")
def run_official_evaluation(payload: Optional[Dict[str, Any]] = None):
    """
    Endpoint for official ISRO / SAC evaluation when authorized ground truth imagery is mounted.
    Fails safely and honestly when official dataset is not mounted.
    """
    return {
        "status": "not_mounted",
        "is_official_evaluation": True,
        "message": (
            "Official ISRO / SAC evaluation requires mounted Cartosat-2S and RISAT-1A "
            "radiometrically calibrated rasters with authorized ground truth shapefiles. "
            "No official evaluation package is currently mounted in data/official_isro/."
        )
    }
