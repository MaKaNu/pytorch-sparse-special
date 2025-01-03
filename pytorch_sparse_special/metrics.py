import torch

from pytorch_sparse_special.special.sparse_mask import SparseMasksTensor
from pytorch_sparse_special.utils import area_of_bbox


def iou_sparse_masks_bbox(sparse_masks: SparseMasksTensor, bbox: torch.Tensor) -> torch.Tensor:
    """Calculates the Intersection over Union for SparseMasksTensor and a bbox

    Args:
        sparse_masks (SparseMasksTensor): Multiple sparse depictions of a class valued. [WxHxN]
        bbox (torch.Tensor): bbox representation in from [xmin, ymin, xmax, ymax].

    Returns:
        torch.Tensor: iou of all masks against the bbox
    """
    iou = sparse_masks.area_per_mask_inside(bbox) / (
        area_of_bbox(bbox) + sparse_masks.area_per_mask() - sparse_masks.area_per_mask_inside(bbox)
    )
    return iou
