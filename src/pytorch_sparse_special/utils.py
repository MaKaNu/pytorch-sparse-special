"""
Utils module: provides the utils functions for pytorch_sparse_special.
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


def area_of_bbox(bbox: torch.Tensor) -> torch.Tensor:
    """Calculate the area of a given bbox

    Args:
        bbox (torch.Tensor): bbox in form [xmin, ymin, xmax, ymax]

    Returns:
        torch.Tensor: Area of bbox.
    """
    xmin, ymin, xmax, ymax = bbox
    return ((xmax - xmin) * (ymax - ymin)).clone().detach()
