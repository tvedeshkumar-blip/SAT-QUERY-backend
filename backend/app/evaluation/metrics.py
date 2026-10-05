import numpy as np
from typing import List, Dict, Any, Optional

def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between binary masks."""
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    intersection = np.logical_and(pred_bin, gt_bin)
    union = np.logical_or(pred_bin, gt_bin)
    union_count = np.count_nonzero(union)
    if union_count == 0:
        return 1.0 if np.count_nonzero(pred_bin) == 0 else 0.0
    return float(np.count_nonzero(intersection) / union_count)

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes IoU between two bounding boxes: [xmin, ymin, xmax, ymax]
    """
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

def compute_cross_modal_correlation(opt_arr: np.ndarray, sar_arr: np.ndarray) -> float:
    """
    Computes Normalized Cross-Correlation (NCC) between optical and SAR rasters.
    """
    if opt_arr.ndim == 3 and opt_arr.shape[2] >= 3:
        opt_gray = 0.299 * opt_arr[:, :, 0] + 0.587 * opt_arr[:, :, 1] + 0.114 * opt_arr[:, :, 2]
    else:
        opt_gray = opt_arr.astype(float) if opt_arr.ndim == 2 else opt_arr[:, :, 0].astype(float)

    sar_gray = sar_arr.astype(float) if sar_arr.ndim == 2 else sar_arr[:, :, 0].astype(float)

    min_h = min(opt_gray.shape[0], sar_gray.shape[0])
    min_w = min(opt_gray.shape[1], sar_gray.shape[1])
    opt_gray = opt_gray[:min_h, :min_w]
    sar_gray = sar_gray[:min_h, :min_w]

    opt_f = opt_gray.flatten()
    sar_f = sar_gray.flatten()

    opt_norm = opt_f - np.mean(opt_f)
    sar_norm = sar_f - np.mean(sar_f)

    denom = np.linalg.norm(opt_norm) * np.linalg.norm(sar_norm)
    if denom == 0:
        return 0.0
    return float(np.dot(opt_norm, sar_norm) / denom)

def compute_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes F1-score (Dice coefficient) on binary masks."""
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    tp = np.count_nonzero(np.logical_and(pred_bin, gt_bin))
    fp = np.count_nonzero(np.logical_and(pred_bin, ~gt_bin))
    fn = np.count_nonzero(np.logical_and(~pred_bin, gt_bin))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0 if (tp + fp + fn) == 0 else 0.0
    return float((2.0 * tp) / denom)
