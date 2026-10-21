"""
Pydantic schemas for Stage 8: Scientific Analysis Quality and Evidence Reporting.
Provides strictly typed, validated models for reproducible analysis reports,
analysis provenance, physical measurement tables, evidence interpretation,
and reproducibility tracking.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from app.schemas.verification import (
    ClaimVerificationItem,
    DisplayStatisticItem,
    HeuristicIndicatorItem,
)


class PhysicalMeasurementReportItem(BaseModel):
    name: str = Field(..., description="Measurement variable name, e.g. optical_mean_boa_reflectance")
    value: Union[float, int, str] = Field(..., description="Numerical value of the measurement")
    unit: str = Field(..., description="Explicit physical unit, e.g. 'unitless ratio [0, 1]', 'dB'")
    statistic_type: str = Field(..., description="Statistic calculated, e.g. 'mean', 'mean linear power'")
    calibration_quantity: Optional[str] = Field(None, description="Calibration quantity, e.g. 'boa_surface_reflectance', 'gamma0'")
    mask_applied: str = Field(..., description="Specific mask applied, e.g. 'joint_valid_mask'")
    source_artifact_ids: List[str] = Field(default_factory=list, description="IDs of artifacts providing evidence for this measurement")


class ReportEvidenceArtifactItem(BaseModel):
    artifact_id: str = Field(..., description="Unique artifact identifier, e.g. 'optical_input', 'sar_input'")
    artifact_type: str = Field(..., description="Artifact category, e.g. 'original', 'processed', 'mask', 'fused'")
    title: str = Field(..., description="Human-readable title")
    description: Optional[str] = Field(None, description="Detailed description")
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Artifact-associated statistics")


class ReportProvenanceSchema(BaseModel):
    analysis_id: str = Field(..., description="Unique analysis execution identifier")
    task: str = Field(..., description="Classified task, e.g. optical_sar, change_detection, vqa")
    query: str = Field(..., description="User query or analysis question")
    mode: str = Field(default="auto", description="Operational mode")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    sensor_identities: List[str] = Field(default_factory=list, description="Observed sensor identities")
    source_asset_ids: List[str] = Field(default_factory=list, description="IDs of source imagery assets")
    acquisition_timestamps: List[str] = Field(default_factory=list, description="Acquisition timestamps of source scenes")
    crs: Optional[str] = Field(None, description="Coordinate Reference System, e.g. EPSG:32643")
    spatial_resolution_m: Optional[Union[float, str]] = Field(None, description="Pixel resolution in meters")
    dimensions: Optional[str] = Field(None, description="Raster dimensions (width x height)")
    registration_status: Dict[str, Any] = Field(default_factory=dict, description="Geometric co-registration metadata")
    valid_pixel_count: Optional[int] = Field(None, description="Number of valid intersecting pixels")
    total_pixel_count: Optional[int] = Field(None, description="Total pixels in scene bounding box")
    valid_pixel_percentage: Optional[float] = Field(None, description="Valid pixel coverage percentage")
    quality_screening: Optional[Dict[str, Any]] = Field(default=None, description="Radiometric quality screening provenance (e.g. SCL classification, SAR validity)")
    observed_pixel_count: Optional[int] = Field(None, description="Number of observed pixels prior to quality masking")
    excluded_quality_pixel_count: Optional[int] = Field(None, description="Number of pixels excluded due to clouds/shadow/invalid radiometric quality")
    quality_clearance_percentage: Optional[float] = Field(None, description="Percentage of valid data with clear quality")
    provider_name: str = Field(default="unknown", description="Analysis provider name")
    actual_model_used: Optional[str] = Field(None, description="Actual model or algorithm executed")
    fallback_used: bool = Field(default=False, description="Whether fallback baseline was activated")
    is_trained_model: bool = Field(default=False, description="Whether a trained neural model was executed")
    model_status: Optional[str] = Field(None, description="Status of the provider/model")
    ndvi_mean: Optional[float] = Field(None, description="Calibrated Sentinel-2 B04+B08 mean NDVI")
    ndvi_p90: Optional[float] = Field(None, description="Calibrated Sentinel-2 B04+B08 90th percentile NDVI")
    vegetation_coverage_pct: Optional[float] = Field(None, description="Percentage of valid pixels with NDVI >= vegetation threshold")
    ndvi_vegetation_threshold: Optional[float] = Field(None, description="Explicit NDVI vegetation threshold (default 0.40)")
    reflectance_scale_factor: Optional[float] = Field(None, description="Surface reflectance scaling factor (e.g. 10000.0)")
    scl_masking_policy: Optional[str] = Field(None, description="Applied SCL masking policy summary")
    b04_asset_id: Optional[str] = Field(None, description="Source asset ID or href for B04 (Red)")
    b08_asset_id: Optional[str] = Field(None, description="Source asset ID or href for B08 (NIR)")
    scl_asset_id: Optional[str] = Field(None, description="Source asset ID or href for SCL mask")
    ndvi_status: Optional[str] = Field(None, description="NDVI calculation status (e.g. 'success' or 'unsupported_zero_valid_pixels')")
    dprvi_mean: Optional[float] = Field(None, description="Calibrated Sentinel-1 RTC VV/VH mean DpRVI")
    dprvi_p90: Optional[float] = Field(None, description="Calibrated Sentinel-1 RTC VV/VH 90th percentile DpRVI")
    cross_pol_ratio_linear: Optional[float] = Field(None, description="Calibrated mean cross-polarization ratio q = VH/VV (linear)")
    cross_pol_ratio_db: Optional[float] = Field(None, description="Calibrated mean cross-polarization ratio 10*log10(q) (dB)")
    sar_vv_mean_db: Optional[float] = Field(None, description="Sentinel-1 RTC mean VV backscatter power converted to dB (10*log10(mean linear))")
    sar_vh_mean_db: Optional[float] = Field(None, description="Sentinel-1 RTC mean VH backscatter power converted to dB (10*log10(mean linear))")
    sar_polarization_mode: Optional[str] = Field(None, description="Polarization configuration (e.g. 'DUAL_VV_VH')")
    sar_vegetation_proxy_coverage_pct: Optional[float] = Field(None, description="Percentage of valid radar pixels with DpRVI >= 0.45")
    dprvi_vegetation_threshold: Optional[float] = Field(None, description="Explicit DpRVI volume scattering canopy threshold (default 0.45)")
    vv_asset_id: Optional[str] = Field(None, description="Source asset ID or href for VV polarization")
    vh_asset_id: Optional[str] = Field(None, description="Source asset ID or href for VH polarization")
    dprvi_status: Optional[str] = Field(None, description="DpRVI calculation status (e.g. 'success' or 'unsupported_zero_valid_pixels')")
    sar_terrain_mask_status: Optional[str] = Field(None, description="Status of terrain layover/shadow mask (e.g. 'unavailable_from_stac_provider')")
    delta_ndvi_mean: Optional[float] = Field(None, description="Calibrated Sentinel-2 bi-temporal mean Delta-NDVI (T2 - T1)")
    vegetation_loss_pct: Optional[float] = Field(None, description="Percentage of joint clear pixels exhibiting vegetation loss (Delta-NDVI <= -0.15)")
    vegetation_gain_pct: Optional[float] = Field(None, description="Percentage of joint clear pixels exhibiting vegetation gain (Delta-NDVI >= +0.15)")
    delta_dprvi_mean: Optional[float] = Field(None, description="Calibrated Sentinel-1 RTC bi-temporal mean Delta-DpRVI (T2 - T1)")
    radar_canopy_loss_pct: Optional[float] = Field(None, description="Percentage of joint valid pixels exhibiting radar canopy loss (Delta-DpRVI <= -0.15)")
    radar_canopy_gain_pct: Optional[float] = Field(None, description="Percentage of joint valid pixels exhibiting radar canopy gain (Delta-DpRVI >= +0.15)")
    temporal_t1_datetime: Optional[str] = Field(None, description="Pre-event acquisition timestamp (T1)")
    temporal_t2_datetime: Optional[str] = Field(None, description="Post-event acquisition timestamp (T2)")
    joint_valid_pixel_count: Optional[int] = Field(None, description="Number of joint valid screened pixels across temporal pair")
    dprvi_assumption_violations_count: Optional[int] = Field(None, description="Count of pixels where q > 1.0 violating co-pol dominance assumption")
    aoi_containment_status: Optional[str] = Field(None, description="AOI containment status: 'contained', 'out_of_bounds_clamped', or None")
    actual_fixture_bounds_wgs84: Optional[List[float]] = Field(None, description="Actual geographic bounds [min_lon, min_lat, max_lon, max_lat] in WGS84")


class ReportVerificationSummarySchema(BaseModel):
    verification_status: str = Field(..., description="'supported', 'partially_supported', 'unsupported', or 'not_run'")
    deterministic_passed: bool = Field(..., description="Whether mandatory deterministic checks passed")
    deterministic_failures: List[str] = Field(default_factory=list, description="List of deterministic rule violations")
    claims: List[ClaimVerificationItem] = Field(default_factory=list, description="Evaluated scientific claims")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions between claims and source evidence")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims lacking supporting evidence")
    missing_evidence: List[str] = Field(default_factory=list, description="Identified evidence gaps")
    scientific_limitations: List[str] = Field(default_factory=list, description="Known scientific caveats and limitations")
    recommended_checks: List[str] = Field(default_factory=list, description="Recommended verification follow-ups")
    is_interpretive_only: bool = Field(default=True, description="Always True: verifier is strictly interpretive")
    interpretive_statement: str = Field(
        default="The interpretive layer provides evidence-grounded explanation based strictly on computed results. It does not alter numerical measurements or replace physical validation.",
        description="Notice on scope of interpretive synthesis"
    )
    summary_explanation: str = Field(..., description="Concise summary of verification outcome")


class ReportReproducibilitySchema(BaseModel):
    software_version: str = Field(
        default="SatQuery-AI v1.0.0 (ISRO PS 26167 Stage 8 Scientific Reporting)",
        description="Software system and pipeline version"
    )
    provider_config: Dict[str, Any] = Field(default_factory=dict, description="Safe configuration parameters (no secrets)")
    pipeline_steps: List[str] = Field(default_factory=list, description="Sequential processing pipeline steps executed")
    unavailable_fields: List[str] = Field(default_factory=list, description="List of fields where metadata was unprovided or unobserved")


class ScientificReportSchema(BaseModel):
    report_id: str = Field(..., description="Unique report identifier, e.g. report_task_...")
    report_version: str = Field(default="1.0.0", description="Report specification schema version")
    generated_at: str = Field(..., description="ISO 8601 timestamp when report was compiled")
    provenance: ReportProvenanceSchema = Field(..., description="Comprehensive provenance and metadata")
    physical_measurements: List[PhysicalMeasurementReportItem] = Field(
        default_factory=list,
        description="Physical scientific measurements with explicit units and masks"
    )
    display_statistics: List[DisplayStatisticItem] = Field(
        default_factory=list,
        description="8-bit normalized display statistics (visualization only)"
    )
    heuristic_indicators: List[HeuristicIndicatorItem] = Field(
        default_factory=list,
        description="Preliminary heuristic proxies (NOT certified classifications)"
    )
    evidence_artifacts: List[ReportEvidenceArtifactItem] = Field(
        default_factory=list,
        description="Visual and data evidence artifacts with statistics"
    )
    verification_summary: Optional[ReportVerificationSummarySchema] = Field(
        default=None,
        description="Stage 7/7B Scientific verification summary and claim evaluations"
    )
    reproducibility: ReportReproducibilitySchema = Field(
        ...,
        description="Reproducibility metadata, pipeline steps, and unavailable fields"
    )
    executive_summary: str = Field(..., description="Synthesized agent findings and answer text")
    integrity_notice: str = Field(
        default="This report strictly separates calibrated physical measurements from 8-bit display visualizations and preliminary heuristic indicators. The interpretive layer does not alter numerical measurements or replace physical validation.",
        description="Mandatory scientific integrity statement"
    )

    @model_validator(mode="after")
    def validate_scientific_integrity(self) -> "ScientificReportSchema":
        # 1. Enforce that heuristic indicators are never certified classifications
        for hi in self.heuristic_indicators:
            if hi.is_certified_classification:
                raise ValueError(
                    f"Scientific integrity violation: Heuristic indicator '{hi.name}' cannot be marked as a certified classification."
                )

        # 2. Reject unknown or fabricated artifact references
        if self.evidence_artifacts:
            known_ids = {a.artifact_id for a in self.evidence_artifacts}
            for pm in self.physical_measurements:
                for src_id in pm.source_artifact_ids:
                    if src_id not in known_ids:
                        raise ValueError(
                            f"Unknown or fabricated artifact reference '{src_id}' in measurement '{pm.name}'. Known artifacts: {sorted(known_ids)}"
                        )
            if self.verification_summary:
                for claim in self.verification_summary.claims:
                    for cited_id in claim.cited_artifact_ids:
                        if cited_id not in known_ids:
                            raise ValueError(
                                f"Unknown or fabricated artifact reference '{cited_id}' cited in claim '{claim.claim_text}'. Known artifacts: {sorted(known_ids)}"
                            )

        # 3. Ensure distinct nomenclature for Gamma0 linear power vs dB
        for pm in self.physical_measurements:
            stat_lower = pm.statistic_type.lower()
            unit_lower = pm.unit.lower()
            if "linear" in stat_lower and "db" in unit_lower:
                raise ValueError(
                    f"Unit contradiction in measurement '{pm.name}': linear power statistic cannot have unit 'dB'."
                )
            if "db" in stat_lower and unit_lower not in ["db", "decibels", "decibel"]:
                raise ValueError(
                    f"Unit contradiction in measurement '{pm.name}': dB statistic must have unit 'dB'."
                )

        return self

