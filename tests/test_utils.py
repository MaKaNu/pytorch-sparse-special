import pytest
import torch

from pytorch_sparse_special.utils import area_of_bbox


@pytest.mark.parametrize(
    "scenarios_sparse, expected_pixel_count",
    [
        (1, torch.tensor(9.0)),
        (2, torch.tensor(12.0)),
        (5, torch.tensor(9.0)),
    ],
    ids=["scenario1", "scenario2", "scenario5"],
    indirect=["scenarios_sparse"],
)
def test_area_of_bbox(scenarios_sparse, expected_pixel_count):
    area = area_of_bbox(scenarios_sparse["bbox"])
    assert torch.equal(area, expected_pixel_count)
