import numpy as np
import logging
from typing import Dict, Any, Optional, List
from app.evaluation.metrics import (
    compute_mask_iou,
    compute_box_iou,
    compute_cross_modal_correlation,
    compute_f1_score
)

logger = logging.getLogger("satquery.evaluation.synthetic")

class SyntheticPipelineValidator:
    """
    Local Pipeline Arithmetic Validator.
    Clearly designated as SYNTHETIC LOCAL VALIDATION for verifying spatial math,
    raster cross-correlation, and bounding box IoU routines without external datasets.
    Must NEVER be portrayed as official ISRO benchmark evaluation.
    """

    @classmethod
    def run_validation(
        cls,
        opt_arr: Optional[np.ndarray] = None,
        sar_arr: Optional[np.ndarray] = None,
        pred_boxes: Optional[List[Dict[str, Any]]] = None,
        gt_boxes: Optional[List[Dict[str, Any]]] = None,
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes local pipeline validation. If images are not provided,
        deterministic geometric test patterns are used (strictly labeled synthetic).
        """
        np.random.seed(seed)
        h, w = 256, 256

        is_procedural = False
        if opt_arr is None or sar_arr is None:
            is_procedural = True
            # Deterministic geometric test pattern
            opt_arr = np.zeros((h, w, 3), dtype=np.uint8)
            opt_arr[30:130, 30:130] = [180, 180, 180] # Urban square
            opt_arr[150:220, 150:220] = [30, 80, 160] # Water square

            sar_arr = np.zeros((h, w), dtype=np.uint8)
            sar_arr[30:130, 30:130] = 220 # High corner reflector
            sar_arr[150:220, 150:220] = 20  # Low backscatter water

        if pred_boxes is None or gt_boxes is None:
            pred_boxes = [{"xmin": 30.0, "ymin": 30.0, "xmax": 130.0, "ymax": 130.0, "label": "Urban Cluster"}]
            gt_boxes = [{"xmin": 32.0, "ymin": 28.0, "xmax": 128.0, "ymax": 132.0, "label": "Urban Ground Truth"}]

        if pred_mask is None or gt_mask is None:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            pred_mask[30:130, 30:130] = 255
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            gt_mask[32:128, 28:132] = 255

        # Compute real metrics using the verified metric library
        mask_iou = round(compute_mask_iou(pred_mask, gt_mask), 4)
        mask_f1 = round(compute_f1_score(pred_mask, gt_mask), 4)
        ncc = round(compute_cross_modal_correlation(opt_arr, sar_arr), 4)

        box_scores = []
        for pb in pred_boxes:
            p_coords = [pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]]
            best_iou = 0.0
            for gb in gt_boxes:
                g_coords = [gb["xmin"], gb["ymin"], gb["xmax"], gb["ymax"]]
                iou = compute_box_iou(p_coords, g_coords)
                if iou > best_iou:
                    best_iou = iou
            box_scores.append(best_iou)

        mean_box_iou = round(float(np.mean(box_scores)), 4) if box_scores else None

        return {
            "validation_mode": "procedural_synthetic" if is_procedural else "user_provided_pair",
            "is_official_isro_evaluation": False,
            "validation_status": "completed",
            "disclaimer": (
                "SYNTHETIC LOCAL PIPELINE VALIDATION ONLY. Computed on test patterns "
                "to verify pipeline arithmetic (Mask IoU, Bounding Box IoU, Cross-Modal NCC). "
                "This does NOT represent official ISRO / SAC Cartosat-2S or RISAT-1A evaluation scores."
            ),
            "metrics": {
                "spatial_ncc_correlation": ncc,
                "mask_iou": mask_iou,
                "mask_f1_score": mask_f1,
                "bounding_box_mean_iou": mean_box_iou,
                "evaluated_pixel_count": int(h * w)
            },
            "test_pattern_metadata": {
                "source": "procedural_test_geometry" if is_procedural else "external_evaluation_pair",
                "raster_dimensions": [w, h],
                "crs_applied": None,
                "is_synthetic_demonstration": True
            }
        }
