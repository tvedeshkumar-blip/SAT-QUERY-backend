from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from app.schemas.verification import VerificationResponseSchema
from app.schemas.report import ScientificReportSchema

class ImageInput(BaseModel):
    data: str = Field(..., description="Base64 encoded image string or URL")
    mimeType: str = Field(default="image/png", description="MIME type e.g. image/tiff, image/png")
    filename: Optional[str] = Field(default="image.png", description="Filename if provided")
    role: Optional[str] = Field(default="primary", description="Role e.g. primary, secondary, optical, sar")
    asset_category: Optional[str] = Field(
        default=None, 
        description="STAC asset category e.g. visual_preview, rgb_visual_raster, single_band_spectral"
    )
    provenance: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="STAC scene and asset provenance metadata"
    )
    scientific_limitations: Optional[List[str]] = Field(
        default_factory=list, 
        description="Scientific caveats associated with this asset"
    )

class AnalysisRequest(BaseModel):
    images: List[ImageInput] = Field(default_factory=list, description="List of input satellite images")
    query: str = Field(..., description="Natural language prompt / question")
    mode: str = Field(default="auto", description="Operational mode: auto, single, optical_sar, bitemporal, orchestration, temporal")
    aoi_bbox: Optional[List[float]] = Field(default=None, description="Optional AOI bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326")
    start_date: Optional[str] = Field(default=None, description="Optional start date (YYYY-MM-DD or ISO 8601)")
    end_date: Optional[str] = Field(default=None, description="Optional end date (YYYY-MM-DD or ISO 8601)")
    t1_start_date: Optional[str] = Field(default=None, description="Pre-event (T1) start date (YYYY-MM-DD)")
    t1_end_date: Optional[str] = Field(default=None, description="Pre-event (T1) end date (YYYY-MM-DD)")
    t2_start_date: Optional[str] = Field(default=None, description="Post-event (T2) start date (YYYY-MM-DD)")
    t2_end_date: Optional[str] = Field(default=None, description="Post-event (T2) end date (YYYY-MM-DD)")
    max_cloud_cover: Optional[float] = Field(default=20.0, description="Maximum allowable optical cloud cover percentage (0-100)")
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
    inspector_data: Optional[Dict[str, Any]] = None

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
    execution_steps: Optional[List[Dict[str, Any]]] = None
    cumulative_transfer_bytes: Optional[int] = None
    transfer_budget_bytes: Optional[int] = None

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
    classification_result: Optional[Dict[str, Any]] = Field(default=None, description="Detailed multi-label classification outputs")
    trace: ExecutionTraceSchema
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: Optional[ConfidenceBreakdownSchema] = None
    conflict_info: Optional[ConflictInfoSchema] = None
    verification: Optional["VerificationResponseSchema"] = Field(default=None, description="Stage 7 Evidence-Grounded Scientific Verification")
    reproducible_report: Optional[ScientificReportSchema] = Field(default=None, description="Stage 8 Structured Reproducible Scientific Analysis Report")
    is_clarification_required: bool = Field(default=False, description="Whether query is ambiguous and requires user clarification")
    missing_parameters: Optional[List[str]] = Field(default=None, description="Parameters that require user clarification")
    suggested_examples: Optional[List[str]] = Field(default=None, description="Suggested valid queries or templates")
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
