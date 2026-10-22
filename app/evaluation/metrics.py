import numpy as np
from typing import List, Dict, Any, Optional

def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between binary masks.
    
    Edge cases:
    - Both masks empty (all zeros): returns 1.0 (perfect agreement on negative background).
    - One mask empty, one non-empty: returns 0.0.
    - Disjoint masks: returns 0.0.
    - Identical non-empty masks: returns 1.0.
    - Shape mismatch: rasters are aligned to common minimum dimensions.
    """
    if pred_mask is None or gt_mask is None or pred_mask.size == 0 or gt_mask.size == 0:
        return 0.0

    min_h = min(pred_mask.shape[0], gt_mask.shape[0])
    min_w = min(pred_mask.shape[1], gt_mask.shape[1])
    p_sub = pred_mask[:min_h, :min_w]
    g_sub = gt_mask[:min_h, :min_w]

    pred_bin = p_sub > 0
    gt_bin = g_sub > 0

    p_count = np.count_nonzero(pred_bin)
    g_count = np.count_nonzero(gt_bin)

    if p_count == 0 and g_count == 0:
        return 1.0
    if p_count == 0 or g_count == 0:
        return 0.0

    intersection = np.logical_and(pred_bin, gt_bin)
    union = np.logical_or(pred_bin, gt_bin)
    union_count = np.count_nonzero(union)
    if union_count == 0:
        return 1.0

    return float(np.count_nonzero(intersection) / union_count)

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes IoU between two bounding boxes: [xmin, ymin, xmax, ymax].
    
    Edge cases:
    - Inverted coordinates (xmin > xmax or ymin > ymax): automatically normalized.
    - Degenerate boxes (zero width or height): returns 0.0.
    - Disjoint boxes: returns 0.0.
    - Identical non-degenerate boxes: returns 1.0.
    - Invalid box format (len != 4): returns 0.0.
    """
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    # Normalize inverted coordinates
    b1_x1, b1_x2 = min(float(box1[0]), float(box1[2])), max(float(box1[0]), float(box1[2]))
    b1_y1, b1_y2 = min(float(box1[1]), float(box1[3])), max(float(box1[1]), float(box1[3]))

    b2_x1, b2_x2 = min(float(box2[0]), float(box2[2])), max(float(box2[0]), float(box2[2]))
    b2_y1, b2_y2 = min(float(box2[1]), float(box2[3])), max(float(box2[1]), float(box2[3]))

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    x_min = max(b1_x1, b2_x1)
    y_min = max(b1_y1, b2_y1)
    x_max = min(b1_x2, b2_x2)
    y_max = min(b1_y2, b2_y2)

    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    inter_area = inter_w * inter_h

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)

def compute_cross_modal_correlation(opt_arr: np.ndarray, sar_arr: np.ndarray) -> float:
    """
    Computes Normalized Cross-Correlation (NCC) between optical and SAR rasters.
    
    Edge cases:
    - Empty or 0-sized arrays: returns 0.0.
    - Constant/flat arrays (zero variance in either image): returns 0.0.
    - Dimension mismatch: cropped to common overlap.
    - NaN/Inf values: sanitized to finite values before computation.
    - Identical patterns: returns 1.0.
    - Perfectly inverted patterns: returns -1.0.
    """
    if opt_arr is None or sar_arr is None or opt_arr.size == 0 or sar_arr.size == 0:
        return 0.0

    if opt_arr.ndim == 3 and opt_arr.shape[2] >= 3:
        opt_gray = 0.299 * opt_arr[:, :, 0] + 0.587 * opt_arr[:, :, 1] + 0.114 * opt_arr[:, :, 2]
    else:
        opt_gray = opt_arr.astype(float) if opt_arr.ndim == 2 else opt_arr[:, :, 0].astype(float)

    sar_gray = sar_arr.astype(float) if sar_arr.ndim == 2 else sar_arr[:, :, 0].astype(float)

    min_h = min(opt_gray.shape[0], sar_gray.shape[0])
    min_w = min(opt_gray.shape[1], sar_gray.shape[1])
    if min_h <= 0 or min_w <= 0:
        return 0.0

    opt_gray = np.nan_to_num(opt_gray[:min_h, :min_w], nan=0.0, posinf=255.0, neginf=0.0)
    sar_gray = np.nan_to_num(sar_gray[:min_h, :min_w], nan=0.0, posinf=255.0, neginf=0.0)

    opt_f = opt_gray.flatten()
    sar_f = sar_gray.flatten()

    std_opt = float(np.std(opt_f))
    std_sar = float(np.std(sar_f))
    if std_opt <= 1e-7 or std_sar <= 1e-7:
        return 0.0

    opt_norm = opt_f - np.mean(opt_f)
    sar_norm = sar_f - np.mean(sar_f)

    norm_opt = np.linalg.norm(opt_norm)
    norm_sar = np.linalg.norm(sar_norm)

    denom = norm_opt * norm_sar
    if denom <= 1e-12:
        return 0.0

    corr = float(np.dot(opt_norm, sar_norm) / denom)
    return float(np.clip(corr, -1.0, 1.0))

def compute_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes F1-score (Dice coefficient) on binary masks: 2*TP / (2*TP + FP + FN).
    
    Edge cases:
    - Both masks empty (all zeros): returns 1.0 (empty predicted mask matches empty ground truth).
    - One empty, other non-empty: returns 0.0.
    - Disjoint masks: returns 0.0.
    - Identical non-empty masks: returns 1.0.
    - Shape mismatch: aligned to common minimum dimensions.
    """
    if pred_mask is None or gt_mask is None or pred_mask.size == 0 or gt_mask.size == 0:
        return 0.0

    min_h = min(pred_mask.shape[0], gt_mask.shape[0])
    min_w = min(pred_mask.shape[1], gt_mask.shape[1])
    p_sub = pred_mask[:min_h, :min_w]
    g_sub = gt_mask[:min_h, :min_w]

    pred_bin = p_sub > 0
    gt_bin = g_sub > 0

    p_count = np.count_nonzero(pred_bin)
    g_count = np.count_nonzero(gt_bin)

    if p_count == 0 and g_count == 0:
        return 1.0
    if p_count == 0 or g_count == 0:
        return 0.0

    tp = np.count_nonzero(np.logical_and(pred_bin, gt_bin))
    fp = np.count_nonzero(np.logical_and(pred_bin, ~gt_bin))
    fn = np.count_nonzero(np.logical_and(~pred_bin, gt_bin))

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2.0 * tp) / denom)
