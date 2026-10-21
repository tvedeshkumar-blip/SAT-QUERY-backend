import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("satquery.conflict")

class ConflictCheckResult(BaseModel):
    conflict_detected: bool = False
    conflict_type: Optional[str] = None
    conflicting_outputs: List[Dict[str, Any]] = []
    reanalysis_tool: Optional[str] = None
    reanalysis_reason: Optional[str] = None
    status: str = "evaluated"  # "evaluated", "unsupported", "insufficient_evidence", "no_conflict"
    uncertainty_reason: Optional[str] = None

class ConflictDetector:
    """
    Deterministic cross-evidence consistency evaluator.
    Identifies observational contradictions between primary model inferences,
    spectral indices, visual grounding components, bi-temporal change metrics,
    and multimodal Optical-SAR cross-sensor observations.
    """

    @classmethod
    def evaluate(
        cls, 
        task: str, 
        query: str, 
        primary_result: Dict[str, Any], 
        metadata: Dict[str, Any],
        evidence_package: Optional[Any] = None,
        candidate_claims: Optional[List[Any]] = None,
    ) -> ConflictCheckResult:
        q_lower = (query or "").lower()
        ans_lower = (primary_result.get("answer") or "").lower()
        evidence_list = primary_result.get("evidence", [])

        # Check 1: Grounding Conflict (Feature requested vs 0 candidate boxes detected)
        if task == "grounding":
            box_count = 0
            for ev in evidence_list:
                if ev.get("boxes"):
                    box_count += len(ev.get("boxes", []))
            
            if box_count == 0 and any(w in q_lower for w in ["building", "urban", "water", "lake", "structure"]):
                return ConflictCheckResult(
                    conflict_detected=True,
                    conflict_type="GROUNDING_ZERO_DETECTIONS",
                    conflicting_outputs=[
                        {"source": "query_prompt", "target_feature": q_lower},
                        {"source": "grounding_model", "detected_bounding_boxes": 0}
                    ],
                    reanalysis_tool="adaptive_spectral_thresholding",
                    reanalysis_reason="Primary grounding threshold yielded zero detections for target feature. Requesting reanalysis with relaxed adaptive threshold."
                )

        # Check 2: Water Presence Contradiction in VQA
        if task == "vqa":
            claims_water = any(w in ans_lower or w in q_lower for w in ["water body", "lake", "reservoir", "river", "open water"])
            provenance = metadata.get("provenance", {})
            if claims_water and provenance.get("detected_modality") == "SAR":
                mean_intensity = primary_result.get("mean_intensity") or metadata.get("mean_intensity")
                if mean_intensity is not None and mean_intensity > 180:
                    return ConflictCheckResult(
                        conflict_detected=True,
                        conflict_type="SAR_SPECULAR_WATER_CONTRADICTION",
                        conflicting_outputs=[
                            {"source": "vqa_assertion", "asserted_feature": "water body"},
                            {"source": "sar_backscatter", "mean_intensity": mean_intensity}
                        ],
                        reanalysis_tool="polarimetric_decomposition",
                        reanalysis_reason="VQA claim asserts water body, but SAR backscatter shows high intensity inconsistent with calm water specular reflection."
                    )

        # Check 3: Bi-Temporal Change Detection Query Contradiction
        if task in ("change_detection", "change_vqa", "temporal_optical_change", "temporal_sar_change"):
            change_pct = primary_result.get("change_area_percentage", 0.0)
            if change_pct is None:
                change_pct = 0.0
                
            # If user explicitly asks about dramatic expansion/increase, but 0.0% change was detected
            if any(w in q_lower for w in ["expansion", "growth", "new construction"]) and change_pct < 0.1:
                return ConflictCheckResult(
                    conflict_detected=True,
                    conflict_type="CHANGE_ZERO_DELTA",
                    conflicting_outputs=[
                        {"source": "query_expectation", "expected_change": "expansion/growth"},
                        {"source": "change_detector", "observed_change_percent": change_pct}
                    ],
                    reanalysis_tool="multi_threshold_spectral_difference",
                    reanalysis_reason="Query prompted for expansion but primary threshold detected negligible change (< 0.1%). Requesting reanalysis with sensitive delta window."
                )

        # Check 4: Multimodal Optical-SAR Cross-Sensor Physical Consistency
        # Optical and SAR measure different physical properties (reflectance vs microwave scattering).
        # We only evaluate contradictions where evidence and metadata support a meaningful comparison.
        if (
            task in ("optical_sar", "optical_sar_fusion", "optical_sar_joint")
            or "optical_sar_fusion" in metadata
            or "registration_info" in primary_result
        ):
            fusion_meta = metadata.get("optical_sar_fusion", {}) if isinstance(metadata.get("optical_sar_fusion"), dict) else {}
            reg_info = primary_result.get("registration_info") or fusion_meta.get("registration_info") or metadata.get("registration_info", {})

            # Propagate unsupported status if underlying observation is explicitly unsupported
            if primary_result.get("status") == "unsupported" or fusion_meta.get("status") == "unsupported":
                reason = primary_result.get("answer") or fusion_meta.get("answer") or "Optical + SAR analysis is unsupported."
                return ConflictCheckResult(
                    conflict_detected=False,
                    status="unsupported",
                    uncertainty_reason=f"Underlying observation is unsupported: {reason}"
                )

            # 4a. Insufficient Evidence: Display-Only / Unaligned Rasters
            # Do NOT infer a contradiction on unaligned or display-only rasters
            is_display_only = reg_info.get("is_display_only", False)
            is_co_registered = reg_info.get("co_registered", True)
            alignment_valid = reg_info.get("alignment_valid", True)
            if is_display_only or not is_co_registered or not alignment_valid:
                return ConflictCheckResult(
                    conflict_detected=False,
                    status="unsupported",
                    uncertainty_reason="Rasters lack authenticated geospatial co-registration (display-only/unaligned). Cross-sensor physical contradiction analysis unsupported."
                )

            # 4b. Insufficient Evidence: Zero Valid Pixels / Cloud Obstruction
            valid_px = primary_result.get("valid_pixel_count")
            if valid_px is None:
                valid_px = fusion_meta.get("valid_pixel_count")
            if valid_px is None and evidence_package is not None:
                valid_px = getattr(evidence_package, "valid_pixel_count", None)

            if valid_px is not None and valid_px == 0:
                return ConflictCheckResult(
                    conflict_detected=False,
                    status="insufficient_evidence",
                    uncertainty_reason="Zero valid clear-sky pixels remain across scenes. Physical observations cannot be evaluated for contradictions."
                )

            # 4c. Specular Water Signature Contradiction
            # Calm open water exhibits specular reflection: low optical reflectance AND low SAR backscatter (< 45 display or < -18 dB)
            water_pct = primary_result.get("water_pct") if primary_result.get("water_pct") is not None else fusion_meta.get("water_pct")
            mean_sar = primary_result.get("mean_sar_display") or fusion_meta.get("mean_sar_display")
            sar_db = primary_result.get("sar_db_mean") or fusion_meta.get("sar_db_mean")
            claims_water = any(w in q_lower or w in ans_lower for w in ["water body", "lake", "reservoir", "river", "open water", "water surface"])

            if claims_water:
                # Contradiction: asserted water body, but co-registered SAR exhibits high diffuse/double-bounce scattering and 0% specular water
                if (water_pct is not None and water_pct == 0.0) and ((mean_sar is not None and mean_sar > 180) or (sar_db is not None and sar_db > -5.0)):
                    return ConflictCheckResult(
                        conflict_detected=True,
                        conflict_type="OPTICAL_SAR_SPECULAR_WATER_CONTRADICTION",
                        conflicting_outputs=[
                            {"source": "claim_or_query", "asserted_feature": "open water"},
                            {"source": "sar_backscatter", "mean_display_intensity": mean_sar, "specular_water_percentage": water_pct}
                        ],
                        reanalysis_tool="cross_sensor_polarimetric_decomposition",
                        reanalysis_reason="Optical claim asserts open water presence, but co-registered SAR evidence indicates high radar backscatter inconsistent with specular reflection on water."
                    )

            # 4d. Urban Double-Bounce vs Specular Water Contradiction
            claims_urban = any(w in q_lower or w in ans_lower for w in ["urban", "city", "dense buildings", "construction", "commercial center"])
            if claims_urban and water_pct is not None and water_pct > 75.0:
                return ConflictCheckResult(
                    conflict_detected=True,
                    conflict_type="OPTICAL_SAR_URBAN_BACKSCATTER_CONTRADICTION",
                    conflicting_outputs=[
                        {"source": "claim_or_query", "asserted_feature": "urban/buildings"},
                        {"source": "multimodal_fusion", "water_coverage_percentage": water_pct}
                    ],
                    reanalysis_tool="urban_coherence_analysis",
                    reanalysis_reason="Claim asserts dense urban construction, but co-registered SAR and optical evidence show low-backscatter specular reflection consistent with water."
                )

            # 4e. Physical Modality Non-Interchangeability Violation
            if any(term in ans_lower for term in [
                "optical reflectance equals sar",
                "sar backscatter indicates optical color",
                "reflectance is interchangeable with backscatter"
            ]):
                return ConflictCheckResult(
                    conflict_detected=True,
                    conflict_type="MODALITY_NON_INTERCHANGEABILITY_VIOLATION",
                    conflicting_outputs=[
                        {"source": "claim_assertion", "invalid_equivalence": "optical_reflectance == sar_backscatter"}
                    ],
                    reanalysis_tool="physical_modality_separation",
                    reanalysis_reason="Claim incorrectly treats optical surface reflectance and radar microwave backscatter as interchangeable physical observables."
                )

        # Check 5: Direct Candidate Claims Contradicted by Evidence
        if candidate_claims:
            for c in candidate_claims:
                c_status = getattr(c, "status", None) or (c.get("status") if isinstance(c, dict) else None)
                if c_status == "contradicted":
                    c_text = getattr(c, "claim_text", "") or (c.get("claim_text", "") if isinstance(c, dict) else str(c))
                    c_reason = getattr(c, "reason", None) or (c.get("reason") if isinstance(c, dict) else "Claim contradicted by deterministic evidence.")
                    return ConflictCheckResult(
                        conflict_detected=True,
                        conflict_type="CLAIM_CONTRADICTED_BY_EVIDENCE",
                        conflicting_outputs=[
                            {"source": "claim", "text": c_text},
                            {"source": "evidence_package", "status": "contradicted"}
                        ],
                        reanalysis_tool="deterministic_evidence_reconciliation",
                        reanalysis_reason=c_reason
                    )

        return ConflictCheckResult(conflict_detected=False, status="no_conflict")
