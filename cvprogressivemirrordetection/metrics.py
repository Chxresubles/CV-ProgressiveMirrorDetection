import torch
from torch import Tensor
from cvprogressivemirrordetection.constants import EPSILON


def iou(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    intersection = torch.sum(pred_masks * gt_masks, dim=(2, 3))
    union = (
        torch.sum(pred_masks, dim=(2, 3))
        + torch.sum(gt_masks, dim=(2, 3))
        - intersection
    )
    return torch.mean(intersection / (union + EPSILON))


def f_score(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    intersection = torch.sum(pred_masks * gt_masks, dim=(2, 3))
    precision = intersection / (torch.sum(pred_masks, dim=(2, 3)) + EPSILON)
    recall = intersection / (torch.sum(gt_masks, dim=(2, 3)) + EPSILON)
    return torch.mean(2 * precision * recall / (precision + recall + EPSILON))


def mae(pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred_masks - gt_masks))
