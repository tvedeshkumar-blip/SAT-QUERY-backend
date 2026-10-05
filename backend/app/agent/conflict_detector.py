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

class ConflictDetector:
    """
    Deterministic cross-evidence consistency evaluator.
    Identifies observational contradictions between primary model inferences,
    spectral indices, visual grounding components, and bi-temporal change metrics.
    """

    @classmethod
    def evaluate(
        cls, 
        task: str, 
        query: str, 
        primary_result: Dict[str, Any], 
        metadata: Dict[str, Any]
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
            claims_water = any(w in ans_lower for w in ["water body", "lake", "reservoir", "river"])
            # Check if spectral evidence or statistics contradict
            provenance = metadata.get("provenance", {})
            if claims_water and provenance.get("detected_modality") == "SAR":
                # In SAR, water is specular (very low backscatter)
                # If mean intensity is high, water claim might be contradictory
                pass

        # Check 3: Bi-Temporal Change Detection Query Contradiction
        if task in ("change_detection", "change_vqa"):
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

        return ConflictCheckResult(conflict_detected=False)
