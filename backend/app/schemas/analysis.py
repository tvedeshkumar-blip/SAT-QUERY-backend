from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class ImageInput(BaseModel):
    data: str = Field(..., description="Base64 encoded image string or URL")
    mimeType: str = Field(default="image/png", description="MIME type e.g. image/tiff, image/png")
    filename: Optional[str] = Field(default="image.png", description="Filename if provided")
    role: Optional[str] = Field(default="primary", description="Role e.g. primary, secondary, optical, sar")

class AnalysisRequest(BaseModel):
    images: List[ImageInput] = Field(..., description="List of input satellite images")
    query: str = Field(..., description="Natural language prompt / question")
    mode: str = Field(default="auto", description="Operational mode: auto, single, optical_sar, bitemporal")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BoundingBoxSchema(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: Optional[str] = None
    confidence: Optional[float] = None

class VisualEvidenceSchema(BaseModel):
    id: str
    type: str  # original, processed, overlay, mask, boxes, change_map, fused
    title: str
    description: Optional[str] = None
    artifact_url: Optional[str] = None
    data_base64: Optional[str] = None
    boxes: Optional[List[BoundingBoxSchema]] = None
    statistics: Optional[Dict[str, Any]] = None

class TraceStepSchema(BaseModel):
    step: str
    timestamp: str
    detail: str
    status: str  # pending, active, completed, failed

class ExecutionTraceSchema(BaseModel):
    task: str
    models_selected: List[str]
    steps: List[TraceStepSchema]
    parameters: Dict[str, Any]
    execution_time_ms: float

class ConfidenceBreakdownSchema(BaseModel):
    model_confidence: Optional[float] = Field(default=None, description="Direct model prediction confidence / logit probability")
    evidence_confidence: Optional[float] = Field(default=None, description="Confidence derived from observational data completeness")
    system_confidence: Optional[float] = Field(default=None, description="Overall calibrated system confidence score")
    evidence_quality: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic evidence quality indicators")

class ConflictInfoSchema(BaseModel):
    conflict_detected: bool = Field(default=False, description="Whether cross-evidence contradiction was detected")
    conflict_type: Optional[str] = Field(default=None, description="Type identifier of the detected contradiction")
    conflict_details: Optional[str] = Field(default=None, description="Explanation of contradictory observations")
    reanalysis_performed: bool = Field(default=False, description="Whether targeted reanalysis was triggered and completed")
    reanalysis_tool: Optional[str] = Field(default=None, description="Identifier of the fallback or specialized reanalysis tool")

class AnalysisResponseSchema(BaseModel):
    id: str
    task: str
    mode: str
    answer: str
    confidence: Optional[float] = None
    confidence_label: str
    models: List[str]
    implementation_status: str = Field(default="baseline", description="Status: baseline, pretrained_model, demo, etc.")
    primary_model: Optional[str] = Field(default=None, description="Primary intended specialist model")
    actual_model_used: Optional[str] = Field(default=None, description="Model or baseline that generated the inference")
    fallback_used: bool = Field(default=False, description="Whether fallback model was activated due to unavailable weights")
    model_status: Optional[str] = Field(default=None, description="Model status: loaded, checkpoint_not_found, baseline, etc.")
    model_provenance: Optional[Dict[str, Any]] = Field(default=None, description="Detailed model provenance metadata")
    evidence: List[VisualEvidenceSchema]
    trace: ExecutionTraceSchema
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: Optional[ConfidenceBreakdownSchema] = None
    conflict_info: Optional[ConflictInfoSchema] = None
    execution_time_ms: float
    created_at: str

class ModelHealthItem(BaseModel):
    name: str
    status: str  # baseline, loaded, unavailable, demo
    task: Optional[str] = None

class HealthResponseSchema(BaseModel):
    status: str
    version: str
    models_loaded: List[str] = Field(default_factory=list)
    models: List[ModelHealthItem] = Field(default_factory=list)
    device: str
