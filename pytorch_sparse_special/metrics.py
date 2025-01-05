"""
Metrics module: provides the metric functions for pytorch_sparse_special.
    Copyright (C) 2025  MaKaNu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
