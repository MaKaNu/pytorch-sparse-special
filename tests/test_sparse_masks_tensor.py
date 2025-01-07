import pytest
import torch

from pytorch_sparse_special.errors import SizeValueError
from pytorch_sparse_special.special.sparse_mask import SparseMasksTensor


@pytest.fixture
def create_instance(scenarios_sparse):
    indices, values, size = scenarios_sparse["sparse"]
    return SparseMasksTensor(indices, values, size)


@pytest.mark.parametrize(
    "scenarios_sparse",
    [1, 2],
    ids=["scenario1", "scenario2"],
    indirect=True,
)
def test_create(create_instance):
    assert isinstance(create_instance, SparseMasksTensor)


@pytest.mark.parametrize(
    "scenarios_sparse_fails, expect_errors",
    ([3, SizeValueError], [4, SizeValueError]),
    ids=["scenario3", "scenario4"],
    indirect=["scenarios_sparse_fails"],
)
def test_create_fails(scenarios_sparse_fails, expect_errors):
    indices, values, size = scenarios_sparse_fails["sparse"]
    with pytest.raises(expect_errors):
        SparseMasksTensor(indices, values, size)


@pytest.mark.parametrize(
    "scenarios_sparse, expected",
    [
        (1, {"shape_idx": torch.Size([3, 12]), "shape_val": torch.Size([12])}),
        (2, {"shape_idx": torch.Size([3, 16]), "shape_val": torch.Size([16])}),
        (5, {"shape_idx": torch.Size([3, 5]), "shape_val": torch.Size([5])}),
        (6, {"shape_idx": torch.Size([3, 5]), "shape_val": torch.Size([5])}),
    ],
    ids=["scenario1", "scenario2", "scenario5", "scenario6"],
    indirect=["scenarios_sparse"],
)
def test_extract_sparse_region(scenarios_sparse, expected):
    indices, values, size = scenarios_sparse["sparse"]
    bbox = scenarios_sparse["bbox"]
    instance = SparseMasksTensor(indices, values, size)
    filtered_idx, filtered_val = instance.extract_sparse_region(bbox)
    assert filtered_idx.shape == expected["shape_idx"]
    assert filtered_val.shape == expected["shape_val"]


@pytest.mark.parametrize(
    "scenarios_sparse, expected_pixel_count",
    [
        (1, torch.tensor([9, 5, 7])),
        (2, torch.tensor([10, 6, 9])),
        (5, torch.tensor([5, 5])),
        (6, torch.tensor([5, 0, 5])),
    ],
    ids=["scenario1", "scenario2", "scenario5", "scenario6"],
    indirect=["scenarios_sparse"],
)
def test_pixel_per_mask(create_instance, expected_pixel_count):
    pixel_count = create_instance.pixel_per_mask()
    assert torch.equal(pixel_count, expected_pixel_count)


@pytest.mark.parametrize(
    "scenarios_sparse, expected_pixel_count",
    [
        (1, torch.tensor([3, 5, 4])),
        (2, torch.tensor([4, 6, 6])),
        (5, torch.tensor([5, 0])),
        (6, torch.tensor([5, 0, 0])),
    ],
    ids=["scenario1", "scenario2", "scenario5", "scenario6"],
    indirect=["scenarios_sparse"],
)
def test_pixel_per_mask_inside(scenarios_sparse, expected_pixel_count):
    indices, values, size = scenarios_sparse["sparse"]
    bbox = scenarios_sparse["bbox"]
    instance = SparseMasksTensor(indices, values, size)
    pixel_count = instance.pixel_per_mask_inside(bbox)
    assert torch.equal(pixel_count, expected_pixel_count)
