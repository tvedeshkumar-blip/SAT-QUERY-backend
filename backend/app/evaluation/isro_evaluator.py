import numpy as np
from typing import Dict, Any, List, Optional
from app.evaluation.metrics import (
    compute_mask_iou,
    compute_box_iou,
    compute_cross_modal_correlation,
    compute_f1_score
)

class ISROEvaluator:
    """
    Evaluation Engine for ISRO / SAC Problem Statement 26167.
    Evaluates Cartosat-2S Optical + RISAT-1A SAR co-registered pairs,
    computing real spatial mask IoU, bounding box overlap, and cross-modal correlation.
    """

    compute_mask_iou = staticmethod(compute_mask_iou)
    compute_box_iou = staticmethod(compute_box_iou)
    compute_cross_modal_correlation = staticmethod(compute_cross_modal_correlation)

    @classmethod
    def evaluate_test_pair(
        cls,
        opt_arr: np.ndarray,
        sar_arr: np.ndarray,
        pred_boxes: Optional[List[Dict[str, Any]]] = None,
        gt_boxes: Optional[List[Dict[str, Any]]] = None,
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes comprehensive evaluation of an Optical + SAR evaluation pair.
        """
        ncc = cls.compute_cross_modal_correlation(opt_arr, sar_arr)

        mask_iou = None
        mask_f1 = None
        if pred_mask is not None and gt_mask is not None:
            mask_iou = round(cls.compute_mask_iou(pred_mask, gt_mask), 4)
            mask_f1 = round(compute_f1_score(pred_mask, gt_mask), 4)

        box_scores = []
        if pred_boxes and gt_boxes:
            for pb in pred_boxes:
                p_coords = [pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]]
                best_iou = 0.0
                for gb in gt_boxes:
                    g_coords = [gb["xmin"], gb["ymin"], gb["xmax"], gb["ymax"]]
                    iou = cls.compute_box_iou(p_coords, g_coords)
                    if iou > best_iou:
                        best_iou = iou
                box_scores.append(best_iou)

        mean_box_iou = round(float(np.mean(box_scores)), 4) if box_scores else None

        return {
            "spatial_ncc_correlation": round(ncc, 4),
            "mask_iou": mask_iou,
            "mask_f1_score": mask_f1,
            "bounding_box_mean_iou": mean_box_iou,
            "evaluation_status": "completed",
            "isro_pipeline_readiness": True
        }
