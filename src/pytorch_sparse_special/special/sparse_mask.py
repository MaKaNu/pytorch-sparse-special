"""
sparse_mask module: provides the SparseMasksTensor class.
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

from ..errors import SizeValueError


class SparseMasksTensor:
    """A 3D Sparse Matrix which represents a stack of binary class masks."""

    def __init__(self, indices: torch.Tensor, values: torch.Tensor, size: tuple[int]) -> None:
        """initilaize a SparseMaskTensor instance.
        The actual tensor is a property of the class object.
        For further information about the class arguments refer:
        [sparse_coo_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html)

        Args:
            indices (torch.Tensor): [DxP] The coordinates for the values of the Matrix. D equals 3.
            values (torch.Tensor): [1xP] The values of the masks.
            size (tuple[int]): Size of the Matrix. Has to be three values. [NxHxW]
                N = Number of masks
                H = Height of image
                W = Width of image

        Raises:
            SizeValueError: If Size or indices doesn't match 3D.
        """
        if len(size) != 3 or indices.shape[0] != 3:
            raise SizeValueError(self)
        self.sparse_tensor: torch.Tensor = torch.sparse_coo_tensor(indices, values, size, is_coalesced=True)
        self.n_total: int = size[0]

    def extract_sparse_region(self, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract non-zero elements within a bounding box from a sparse tensor.

        Args:
            bbox (torch.Tensor): The BBox, which inhouses the pixels

        Returns:
            tuple[torch.Tensor, torch.Tensor]: filtered indices and values which are inside the bbox.
        """

        x_min, y_min, x_max, y_max = bbox
        indices = self.sparse_tensor.indices()
        values = self.sparse_tensor.values()

        # Mask for indices within the bounding box
        mask_x = (indices[1] >= x_min) & (indices[1] < x_max)
        mask_y = (indices[2] >= y_min) & (indices[2] < y_max)
        mask = mask_x & mask_y

        # Extract the relevant indices and values
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]

        return filtered_indices, filtered_values

    def _get_full_count(self, indices: torch.Tensor) -> torch.Tensor:
        """Creates for all masks count of pixels based on indices.

        Args:
            indices (torch.Tensor): Tensor with indices.

        Returns:
            torch.Tensor: Tensor of shape [N, 1] with the number of pixels based on indices.
        """
        # count the values on the N axis
        unique_index, count = indices[0, :].unique(return_counts=True)

        # Create Tensor with the range of all mask
        # necessary, if mask not inside bbox and we want to keep the actual shape
        num_masks = torch.arange(self.n_total)

        # Final variable which has the shape matching all masks
        full_count = torch.zeros(num_masks.shape, dtype=torch.long)

        # The unique_index correlates with the mask layer index
        # which enables infusing the count into full_count
        full_count[unique_index] = count
        return full_count

    def pixel_per_mask(self) -> torch.Tensor:
        """Count the number of pixels per masks from the sparse matrix.

        Returns:
            Tensor: Number of unique values on z axis.
        """
        indices = self.sparse_tensor.indices()
        # only need to count all unique values on the N axis
        count: torch.Tensor = indices[0, :].unique(return_counts=True)[1]
        return count

    def pixel_per_mask_inside(self, bbox: torch.Tensor) -> torch.Tensor:
        """Count the number of pixels per mask inside the given bbox from the sparse matrix.

        Args:
            bbox (Tensor): holds the bbox information (xmin, ymin, xmax, ymax)

        Returns:
            Tensor: Number of unique values on z axis inside bbox
        """
        inside_indices, _ = self.extract_sparse_region(bbox)
        return self._get_full_count(inside_indices)
