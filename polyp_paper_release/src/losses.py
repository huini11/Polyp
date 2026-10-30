"""
Composite training loss for the boundary-aware polyp segmentation network.

Main segmentation head : BCE + Soft IoU (weighted 1:1 by default).
Boundary head          : BCE + Dice on a boundary ground truth generated on
                         the fly from the mask via a morphological gradient
                         (dilate - erode, implemented with pooling ops so no
                         extra dependency is needed).
Deep supervision       : the same BCE+IoU loss applied to auxiliary coarse
                         seg maps against downsampled masks; the weight is
                         halved at each coarser level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["make_boundary_gt", "bce_loss", "soft_iou_loss", "dice_loss",
           "CompoundLoss"]


def make_boundary_gt(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Boundary GT from a binary mask via morphological gradient.
    mask: (B,1,H,W) in {0,1}. Returns (B,1,H,W) in {0,1}.
    Uses max-pool as dilation and -max-pool(-x) as erosion (no cv2 needed,
    works on GPU and in autograd-free contexts).
    """
    pad = k // 2
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable BCE-with-logits averaged over batch and pixels."""
    return F.binary_cross_entropy_with_logits(pred, target)


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft IoU (Jaccard) loss on sigmoid probabilities; pred is a logit map."""
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss; pred is a logit map."""
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    """
    Total loss:
        L = w_bce * BCE(seg) + w_iou * SoftIoU(seg)
          + w_boundary * [BCE(bnd) + Dice(bnd)]           (boundary head)
          + w_aux * sum_l 0.5^l * [BCE + SoftIoU](aux_l)  (deep supervision)

    aux targets are the masks downsampled to each aux map's resolution.
    """

    def __init__(self, w_bce=1.0, w_iou=1.0, w_boundary=0.5, w_aux=0.4,
                 boundary_k=3):
        super().__init__()
        self.w_bce = w_bce
        self.w_iou = w_iou
        self.w_boundary = w_boundary
        self.w_aux = w_aux
        self.boundary_k = boundary_k

    def _seg_loss(self, pred, target):
        return bce_loss(pred, target) + soft_iou_loss(pred, target)

    def forward(self, outputs: dict, mask: torch.Tensor) -> dict:
        """
        outputs: dict from the model {"seg", "boundary"?, "aux": [a2, a3]}
        mask   : (B,1,H,W) in {0,1}
        Returns {"total", "seg", "boundary", "aux"} for logging.
        """
        seg = outputs["seg"]
        losses = {
            "seg": self._seg_loss(seg, mask),
        }
        total = self.w_bce * bce_loss(seg, mask) + self.w_iou * soft_iou_loss(seg, mask)

        # Boundary supervision
        if "boundary" in outputs and self.w_boundary > 0:
            bgt = make_boundary_gt(mask, self.boundary_k)
            bnd = outputs["boundary"]
            losses["boundary"] = bce_loss(bnd, bgt) + dice_loss(bnd, bgt)
            total = total + self.w_boundary * losses["boundary"]

        # Deep supervision on aux maps
        if "aux" in outputs and self.w_aux > 0:
            aux_total = 0.0
            w = self.w_aux
            for aux in outputs["aux"]:
                tgt = F.interpolate(mask, size=aux.shape[-2:], mode="nearest")
                aux_total = aux_total + w * self._seg_loss(aux, tgt)
                w *= 0.5
            losses["aux"] = aux_total
            total = total + aux_total

        losses["total"] = total
        return losses
