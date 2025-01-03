import torch


def area_of_bbox(bbox: torch.Tensor) -> torch.Tensor:
    """Calculate the area of a given bbox

    Args:
        bbox (torch.Tensor): bbox in form [xmin, ymin, xmax, ymax]

    Returns:
        torch.Tensor: Area of bbox.
    """
    xmin, ymin, xmax, ymax = bbox
    return torch.tensor((xmax - xmin) * (ymax - ymin))
