import logging
from typing import Dict, Any, Optional, List
import numpy as np
from app.models.base import BaseModel
from app.models.change_detection.change_detector import ChangeDetectionProvider

logger = logging.getLogger("satquery.change_vqa")


class ChangeVQA(BaseModel):
    """
    Evidence-grounded Bi-Temporal Change VQA Adapter.
    Uses ChangeDetectionProvider and reports the actual underlying model.
    """

    def __init__(self):
        super().__init__(
            model_name="EvidenceGroundedChangeVQA",
            task_type="change_vqa",
            implementation_status="baseline",
            is_trained=False,
            is_remote_sensing_adapted=False
        )
        self.change_detector = ChangeDetectionProvider()

    def predict(
        self,
        images: List[np.ndarray],
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if len(images) < 2:
            raise ValueError("Change VQA requires 2 image inputs (Earlier & Later).")

        # Run the actual change-detection provider.
        cd_result = self.change_detector.predict(images, query, metadata)

        q = (query or "What changed between these two dates?").strip()
        change_pct = cd_result.get("changed_area_percent", 0.0)
        evidence = cd_result.get("evidence", [])
        reg_info = cd_result.get("registration_info", {})

        cluster_count = 0
        if len(evidence) > 1 and "boxes" in evidence[1]:
            cluster_count = len(evidence[1]["boxes"])

        q_lower = q.lower()
        semantic_query = any(k in q_lower for k in [
            "building", "urban", "vegetation", "forest", "crop",
            "road", "water", "flood", "loss", "gain"
        ])

        if not reg_info.get("georeferenced", True):
            co_reg_note = "Inputs lack valid geospatial referencing (CRS unavailable); genuine bi-temporal remote sensing change detection requires georeferenced GeoTIFFs."
        elif reg_info.get("co_registered"):
            co_reg_note = "Inputs are verified co-registered."
        else:
            co_reg_note = "Inputs lack verified spatial co-registration; spatial shift may affect pixel differences."

        if semantic_query:
            attribution_text = (
                "Spectral change detected; semantic class attribution is not available "
                "from the change-detection model alone."
            )
        else:
            attribution_text = (
                f"Detected notable reflectance variance across "
                f"{cluster_count} distinct spatial clusters."
            )

        answer = (
            f"Bi-temporal change analysis: {change_pct:.2f}% of the scene exhibited "
            f"measurable spectral difference. {co_reg_note} {attribution_text}"
        )

        # Preserve the actual provider/model telemetry.
        actual_model = cd_result.get(
            "actual_model_used",
            cd_result.get("model_name", self.change_detector.model_name)
        )
        primary_model = cd_result.get(
            "primary_model",
            self.change_detector.model_name
        )
        fallback_used = cd_result.get("fallback_used", False)
        implementation_status = cd_result.get(
            "implementation_status",
            "baseline"
        )

        return {
            "answer": answer,
            "changed_area_percent": change_pct,
            "cluster_count": cluster_count,

            # Actual execution information
            "model_type": actual_model,
            "implementation_status": implementation_status,
            "actual_model_used": actual_model,
            "primary_model": primary_model,
            "fallback_used": fallback_used,

            "semantic_attribution_available": False,
            "evidence_based": True,
            "confidence": None,
            "confidence_label": (
                "Real model execution"
                if implementation_status == "real_model"
                else "Not available (Evidence-Grounded Baseline)"
            ),

            "models": [
                self.model_name,
                primary_model,
                actual_model
            ],

            "registration_info": reg_info,
            "evidence": evidence
        }
