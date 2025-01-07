import pytest
import torch

from pytorch_sparse_special.metrics import iou_sparse_masks_bbox
from pytorch_sparse_special.special.sparse_mask import SparseMasksTensor


@pytest.mark.parametrize(
    "scenarios_sparse, expected_iou",
    [
        (1, torch.tensor([3 / 15, 5 / 9, 4 / 12])),
        (2, torch.tensor([4 / 18, 6 / 12, 6 / 15])),
        (5, torch.tensor([5 / 9, 0 / 14])),
        (6, torch.tensor([5 / 9, 0, 0 / 14])),
    ],
    ids=["scenario1", "scenario2", "scenario5", "scenario6"],
    indirect=["scenarios_sparse"],
)
def test_iou_sparse_masks_bbox(scenarios_sparse, expected_iou):
    indices, values, size = scenarios_sparse["sparse"]
    bbox = scenarios_sparse["bbox"]
    instance = SparseMasksTensor(indices, values, size)
    iou = iou_sparse_masks_bbox(instance, bbox)
    assert torch.isclose(iou, expected_iou).all()
