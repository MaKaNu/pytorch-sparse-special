import pytest
import torch

"""
Helper
"""


def create_sparse(indices, values, size, bbox=None):
    return {"sparse": (indices, values, size), "bbox": bbox}


"""
Scenario1: 3 Masks in Square Image
"""


@pytest.fixture
def sparse_scenario1():
    indices = torch.tensor([
        # <-------T-Shape-------->  <---Cross--->  <------Rects------>
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],  # N
        [0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 1, 2, 3, 2, 0, 1, 0, 1, 2, 1, 2],  # H
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 2, 2, 3, 0, 0, 1, 1, 1, 2, 2],  # W
    ])
    values = torch.tensor([1] * 9 + [2] * 5 + [3] * 7, dtype=torch.float32)
    size = (3, 5, 5)
    bbox = torch.tensor([1, 1, 4, 4], dtype=torch.float32)
    return create_sparse(indices, values, size, bbox)


"""
Scenario2: 3 Masks in Rectangle Image
"""


@pytest.fixture
def sparse_scenario2():
    indices = torch.tensor([
        # <--------T-Shape---------->  <-----Cross---->  <---------Rects--------->
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 1, 2, 1, 2, 1, 2],
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 2, 2, 3, 4, 0, 0, 1, 1, 1, 2, 2, 3, 3],
    ])
    values = torch.tensor([1] * 10 + [2] * 6 + [3] * 9, dtype=torch.float32)
    size = (3, 6, 5)
    bbox = torch.tensor([1, 1, 4, 5], dtype=torch.float32)
    return create_sparse(indices, values, size, bbox)


"""
Scenario3: Raises Error because size is not 3D
"""


@pytest.fixture
def sparse_scenario3():
    indices = torch.tensor([
        [1],
        [1],
        [1],
    ])
    values = torch.tensor([1], dtype=torch.float32)
    size = (2, 2)
    return create_sparse(indices, values, size)


"""
Scenario4: Raises Error because indices is not 3D
"""


@pytest.fixture
def sparse_scenario4():
    indices = torch.tensor([
        [1],
        [1],
    ])
    values = torch.tensor([1], dtype=torch.float32)
    size = (2, 2, 2)
    return create_sparse(indices, values, size)


"""
Scenario5: Mask OutSide
"""


@pytest.fixture
def sparse_scenario5():
    indices = torch.tensor([
        # <---Cross-->  <--L-Shape-->
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # N
        [2, 1, 2, 3, 2, 2, 3, 4, 4, 4],  # H
        [1, 2, 2, 2, 3, 4, 4, 4, 3, 2],  # W
    ])
    values = torch.tensor([2] * 5 + [4] * 5, dtype=torch.float32)
    size = (2, 5, 5)
    bbox = torch.tensor([1, 1, 4, 4], dtype=torch.float32)
    return create_sparse(indices, values, size, bbox)


"""
Scenario6: Empty Masks
"""


@pytest.fixture
def sparse_scenario6():
    indices = torch.tensor([
        # <---Cross-->  <--L-Shape-->
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],  # N
        [2, 1, 2, 3, 2, 2, 3, 4, 4, 4],  # H
        [1, 2, 2, 2, 3, 4, 4, 4, 3, 2],  # W
    ])
    values = torch.tensor([2] * 5 + [4] * 5, dtype=torch.float32)
    size = (3, 5, 5)
    bbox = torch.tensor([1, 1, 4, 4], dtype=torch.float32)
    return create_sparse(indices, values, size, bbox)


"""
Combined Scenarios
"""


@pytest.fixture
def scenarios_sparse(request, sparse_scenario1, sparse_scenario2, sparse_scenario5, sparse_scenario6):
    scenarios = {
        1: sparse_scenario1,
        2: sparse_scenario2,
        5: sparse_scenario5,
        6: sparse_scenario6,
    }
    if request.param not in scenarios:
        raise NotImplementedError(f"No such scenario: {request.param!r}")
    return scenarios[request.param]


@pytest.fixture()
def scenarios_sparse_fails(request, sparse_scenario3, sparse_scenario4):
    scenarios = {
        3: sparse_scenario3,
        4: sparse_scenario4,
    }
    if request.param not in scenarios:
        raise NotImplementedError(f"No such scenario: {request.param!r}")
    return scenarios[request.param]
