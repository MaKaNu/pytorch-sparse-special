import torch


class SparseMasksTensor:
    def __init__(self, indices, values, size):
        self.sparse_tensor = torch.sparse_coo_tensor(indices, values, size, is_coalesced=True)
        self.x_total = size[0]
        self.y_total = size[1]
        self.z_total = size[2]
        self.norm_pixel_area = (1 / self.x_total) * (1 / self.y_total)

    def extract_sparse_region(self, bbox: torch.Tensor):
        """
        Extract non-zero elements within a bounding box from a sparse tensor.
        sparse_tensor: torch.sparse_coo_tensor
        bbox: [x_min, y_min, x_max, y_max]
        """

        x_min, y_min, x_max, y_max = bbox
        indices = self.sparse_tensor.indices()
        values = self.sparse_tensor.values()

        # Mask for indices within the bounding box
        mask_x = (indices[1] >= x_min * self.x_total) & (indices[1] < x_max * self.y_total)
        mask_y = (indices[0] >= y_min * self.x_total) & (indices[0] < y_max * self.y_total)
        mask = mask_x & mask_y

        # Extract the relevant indices and values
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]

        return filtered_indices, filtered_values

    def pixel_per_mask(self) -> torch.Tensor:
        """Count the number of pixels per masks from the sparse matrix.

        Returns:
            Tensor: Number of unique values on z axis.
        """
        indices = self.sparse_tensor.indices()
        # only need to count all unique values on the z axis
        _, count = indices[2, :].unique(return_counts=True)
        return count

    def pixel_per_mask_inside(self, bbox: torch.Tensor):
        """Count the number of pixels per mask inside the given bbox from the sparse matrix.

        Args:
            bbox (Tensor): holds the bbox information (xmin, ymin, xmax, ymax)

        Returns:
            Tensor: Number of unique values on z axis inside bbox
        """
        inside_indices, _ = self.extract_sparse_region(bbox)
        # count the values on the z axis
        unique_index, count = inside_indices[2, :].unique(return_counts=True)

        # Create Tensor with the range of all mask
        # necessary, if mask not inside bbox and we want to keep the actual shape
        num_masks = torch.arange(self.z_total)

        # Final variable which has the shape matching all masks
        full_count = torch.zeros(num_masks.shape, dtype=torch.long)

        # The unique_index correlates with the mask layer index
        # which enables infusing the count into full_count
        full_count[unique_index] = count
        return full_count

    def area_per_mask(self):
        return self.norm_pixel_area * self.pixel_per_mask()

    def area_per_mask_inside(self, bbox: torch.Tensor):
        return self.norm_pixel_area * self.pixel_per_mask_inside(bbox)
