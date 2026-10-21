"""
Pydantic schemas for Stage 7 Evidence-Grounded LLM Scientific Verification.
Provides strictly typed, validated contracts for evidence packages, claim evaluations,
and verification responses.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class PhysicalMeasurementItem(BaseModel):
    name: str = Field(..., description="Physical measurement variable name, e.g. surface_reflectance_b04")
    value: Union[float, int, str] = Field(..., description="Numerical value of the measurement")
    unit: str = Field(..., description="Physical unit, e.g. 'unitless', 'linear power (m^2/m^2)', 'dB'")
    statistic_type: str = Field(..., description="Statistic calculated, e.g. 'mean', 'median', 'p95'")
    mask_applied: str = Field(..., description="Specific mask applied, e.g. 'joint_valid_mask', 'valid_pixels'")
    calibration_quantity: Optional[str] = Field(None, description="Calibration quantity, e.g. 'boa_surface_reflectance', 'gamma0'")


class DisplayStatisticItem(BaseModel):
    name: str = Field(..., description="Display statistic name, e.g. optical_albedo_display_mean")
    value: Union[float, int, str] = Field(..., description="Display value, e.g. 92.8")
    scale: str = Field(default="8-bit normalized [0, 255]", description="Representation scale")
    purpose: str = Field(default="visualization_only", description="Must be visualization_only")


class HeuristicIndicatorItem(BaseModel):
    name: str = Field(..., description="Heuristic indicator name, e.g. heuristic_surface_water_proxy")
    percentage: float = Field(..., description="Preliminary percentage computed from heuristic slicing")
    pixel_count: int = Field(..., description="Number of pixels matching the heuristic threshold")
    definition: str = Field(..., description="Definition / threshold used")
    is_certified_classification: bool = Field(default=False, description="Must ALWAYS be False; not ground truth")
    limitations: str = Field(..., description="Scientific limitations of this proxy")


class EvidenceArtifactItem(BaseModel):
    artifact_id: str = Field(..., description="Unique artifact identifier, e.g. 'optical_input', 'sar_input'")
    artifact_type: str = Field(..., description="Artifact category, e.g. 'original', 'processed', 'mask', 'fused'")
    title: str = Field(..., description="Human-readable title")
    description: Optional[str] = Field(None, description="Detailed description")
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Artifact-associated statistics")


class ProviderInfoItem(BaseModel):
    provider_name: str = Field(default="unknown")
    actual_model_used: Optional[str] = None
    fallback_used: bool = False
    is_trained_model: bool = False
    model_status: Optional[str] = None


class EvidencePackageSchema(BaseModel):
    task_id: str
    query: str
    sensor_identities: List[str] = Field(default_factory=list)
    acquisition_timestamps: List[str] = Field(default_factory=list)
    source_assets: List[Dict[str, Any]] = Field(default_factory=list)
    crs: Optional[str] = None
    resolution: Optional[Union[float, str]] = None
    dimensions: Optional[str] = None
    registration_status: Dict[str, Any] = Field(default_factory=dict)
    valid_pixel_count: Optional[int] = None
    total_pixel_count: Optional[int] = None
    valid_pixel_percentage: Optional[float] = None
    physical_measurements: List[PhysicalMeasurementItem] = Field(default_factory=list)
    display_statistics: List[DisplayStatisticItem] = Field(default_factory=list)
    heuristic_indicators: List[HeuristicIndicatorItem] = Field(default_factory=list)
    evidence_artifacts: List[EvidenceArtifactItem] = Field(default_factory=list)
    provider_info: ProviderInfoItem = Field(default_factory=ProviderInfoItem)
    quality_screening: Optional[Dict[str, Any]] = None
    validation_warnings: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)


class ClaimVerificationItem(BaseModel):
    claim_text: str = Field(..., description="The specific claim being evaluated")
    claim_type: str = Field(..., description="Type of claim: 'measurement', 'display', 'provenance', 'registration', 'model', 'heuristic', 'general'")
    status: str = Field(..., description="Verification status of this claim: 'supported', 'partially_supported', 'unsupported', 'contradicted'")
    cited_artifact_ids: List[str] = Field(default_factory=list, description="Artifact IDs cited in support of this claim")
    evidence_found: Optional[str] = Field(None, description="Actual evidence located in the package")
    reason: Optional[str] = Field(None, description="Detailed scientific justification or contradiction reason")


class VerificationResponseSchema(BaseModel):
    verification_status: str = Field(..., description="'supported', 'partially_supported', 'unsupported', or 'not_run'")
    deterministic_passed: bool = Field(..., description="Whether all deterministic scientific integrity checks passed")
    deterministic_failures: List[str] = Field(default_factory=list, description="List of deterministic rule violations if any")
    claims: List[ClaimVerificationItem] = Field(default_factory=list, description="Individual evaluated claims")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions between claims and source evidence")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims lacking supporting evidence")
    missing_evidence: List[str] = Field(default_factory=list, description="Evidence gaps or unverified metadata")
    scientific_limitations: List[str] = Field(default_factory=list, description="Scientific limitations and caveats")
    recommended_checks: List[str] = Field(default_factory=list, description="Recommended follow-up verification actions")
    summary_explanation: str = Field(..., description="Concise evidence-grounded scientific explanation")
    is_interpretive_only: bool = Field(default=True, description="Always True: verifier is strictly an interpretive layer")
    interpretive_statement: str = Field(
        default="The LLM verifier is strictly an interpretive and explanatory layer based solely on computed evidence. It does not calculate raster data, alter numerical measurements, or replace physical validation.",
        description="Explicit notice regarding LLM scope"
    )
    llm_model_used: Optional[str] = None
    llm_run: bool = Field(default=False, description="Whether LLM inference was successfully executed")
    execution_time_ms: float = 0.0


class VerificationRequestSchema(BaseModel):
    evidence_package: Optional[EvidencePackageSchema] = None
    analysis_response: Optional[Dict[str, Any]] = None
    custom_claims: Optional[List[str]] = None
