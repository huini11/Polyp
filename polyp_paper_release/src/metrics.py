"""
Evaluation metrics for polyp segmentation.

Conventions (aligned with PraNet / Polyp-PVT and boundary-quality papers):
  - Predictions are sigmoid probabilities resized to the ORIGINAL image size
    before thresholding (done by the caller / eval.py).
  - mDice / mIoU: per-image Dice and IoU on the binary prediction, averaged
    over the dataset.
  - BF score (Borji et al. 2019 "Boundary F1"): boundary F-measure between
    predicted and GT boundaries with a pixel tolerance; we sweep thresholds
    0.1..0.9 and report the max (maxBF), matching common usage.
  - Boundary IoU (Cheng et al. 2021): IoU between boundary bands of the
    prediction and GT, where a boundary band is dilate(x, t) - erode(x, t)
    with tolerance t pixels (default 2 px at the evaluation resolution).

All functions take numpy arrays: pred_prob (H,W) float in [0,1],
gt_mask (H,W) uint8 {0,1}.
"""

import cv2
import numpy as np

__all__ = ["dice_score", "iou_score", "extract_boundary", "bf_score",
           "boundary_iou", "evaluate_image"]


# ---------------------------------------------------------------------------
# Region metrics
# ---------------------------------------------------------------------------

def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float((inter + eps) / (union + eps))


# ---------------------------------------------------------------------------
# Boundary utilities
# ---------------------------------------------------------------------------

def _disk_kernel(radius: int) -> np.ndarray:
    """Circular structuring element of given radius (cv2 morphology)."""
    size = 2 * radius + 1
    k = np.zeros((size, size), np.uint8)
    cv2.circle(k, (radius, radius), radius, 1, -1)
    return k


def extract_boundary(mask_bin: np.ndarray, tolerance: int = 2) -> np.ndarray:
    """
    Boundary band of a binary mask: dilate(x, t) - erode(x, t).
    mask_bin: (H,W) {0,1} uint8/bool. Returns (H,W) {0,1} uint8.
    """
    m = (mask_bin.astype(np.uint8)) * 255
    if tolerance <= 0:
        return (m > 0).astype(np.uint8)
    kernel = _disk_kernel(tolerance)
    dil = cv2.dilate(m, kernel)
    ero = cv2.erode(m, kernel)
    return ((dil > 0) & ~(ero > 0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------------

def _fscore(pred_b: np.ndarray, gt_b: np.ndarray, tolerance: int = 2,
            eps: float = 1e-6):
    """
    Precision / Recall of the predicted boundary against the GT boundary with
    pixel tolerance: a predicted boundary pixel counts as matched if it falls
    within `tolerance` px of a GT boundary pixel (implemented by dilating the
    counterpart boundary before counting).
    """
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0, 1.0, 1.0  # both empty -> perfect match by convention
    kernel = _disk_kernel(tolerance)
    gt_d = cv2.dilate(gt_b * 255, kernel) > 0
    pr_d = cv2.dilate(pred_b * 255, kernel) > 0
    precision = float((pred_b > 0)[gt_d].sum() / (max((pred_b > 0).sum(), eps)))
    recall = float((gt_b > 0)[pr_d].sum() / (max((gt_b > 0).sum(), eps)))
    f = 2 * precision * recall / max(precision + recall, eps)
    return precision, recall, f


def bf_score(pred_prob: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2,
             thresholds=None) -> float:
    """
    max BF score over probability thresholds (default 0.1..0.9 step 0.1).
    pred_prob: (H,W) float in [0,1]; gt_bin: (H,W) {0,1}.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.1)
    gt_b = extract_boundary(gt_bin, tolerance)
    best = 0.0
    for t in thresholds:
        pred_b = extract_boundary((pred_prob >= t).astype(np.uint8), tolerance)
        _, _, f = _fscore(pred_b, gt_b, tolerance)
        best = max(best, f)
    return float(best)


def boundary_iou(pred_prob: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2,
                 threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Boundary IoU at a fixed probability threshold (default 0.5) and boundary
    band tolerance (default 2 px). Empty-vs-empty counts as 1.0.
    """
    pred_b = extract_boundary((pred_prob >= threshold).astype(np.uint8), tolerance)
    gt_b = extract_boundary(gt_bin, tolerance)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float((inter + eps) / (union + eps))


# ---------------------------------------------------------------------------
# One-call per-image evaluation
# ---------------------------------------------------------------------------

def evaluate_image(pred_prob: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> dict:
    """
    Compute all paper metrics for one image.
    pred_prob/gt_bin must already be at the ORIGINAL image resolution.
    Returns {"dice", "iou", "bf", "boundary_iou"}.
    """
    gt_bin = (gt_bin > 0).astype(np.uint8)
    pred_bin = (pred_prob >= 0.5).astype(np.uint8)
    return {
        "dice": dice_score(pred_bin, gt_bin),
        "iou": iou_score(pred_bin, gt_bin),
        "bf": bf_score(pred_prob, gt_bin, tolerance),
        "boundary_iou": boundary_iou(pred_prob, gt_bin, tolerance),
    }
