from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class EvaluationStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BenchmarkResultItem(BaseModel):
    dataset: str
    split: str = "test"
    task: str
    model: str
    metric: str
    score: Optional[float] = None  # Numerical score only when status is COMPLETED
    score_display: Optional[str] = None
    timestamp: str
    commit_hash: Optional[str] = "HEAD"
    configuration: Dict[str, Any] = Field(default_factory=dict)
    evaluation_status: EvaluationStatus = EvaluationStatus.NOT_EVALUATED
    notes: Optional[str] = None

class BenchmarkSuiteResponse(BaseModel):
    status: str
    results: List[BenchmarkResultItem]
    message: str

class SyntheticValidationRequest(BaseModel):
    sample_size: int = 1
    seed: Optional[int] = 42

class SyntheticValidationResponse(BaseModel):
    pipeline_type: str = "synthetic_pipeline_validation"
    is_official_evaluation: bool = False
    disclaimer: str = (
        "SYNTHETIC VALIDATION ONLY. Generated on procedural in-memory rasters to verify "
        "pipeline arithmetic (IoU, NCC correlation, difference masks). Does not represent official "
        "ISRO / SAC Cartosat or RISAT performance scores."
    )
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
