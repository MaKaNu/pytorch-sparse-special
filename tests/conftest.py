import pytest
import torch


@pytest.fixture
def indices_scenario1():
    return torch.tensor([
        # <-------T-Shape-------->  <---Cross--->  <------Rects------>
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 2, 2, 3, 0, 0, 1, 1, 1, 2, 2],  # X
        [0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 1, 2, 3, 2, 0, 1, 0, 1, 2, 1, 2],  # Y
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],  # Z
    ])


@pytest.fixture
def values_scenario1():
    return torch.tensor(
        # <-------T-Shape-------->  <---Cross--->  <------Rects------>
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        dtype=torch.float32,
    )


@pytest.fixture
def size_scenario1():
    return (5, 5, 3)


@pytest.fixture
def sparse_scenario1(indices_scenario1, values_scenario1, size_scenario1):
    return indices_scenario1, values_scenario1, size_scenario1


@pytest.fixture
def bbox_scenario1():
    return torch.tensor(
        [
            0.2,  # xmin
            0.2,  # ymin
            0.8,  # xmax
            0.8,  # ymax
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def indices_scenario2():
    return torch.tensor([
        # <--------T-Shape---------->  <-----Cross---->  <---------Rects--------->
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 2, 2, 3, 4, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        [0, 1, 2, 3, 4, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 1, 2, 1, 2, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])


@pytest.fixture
def values_scenario2():
    return torch.tensor(
        # <--------T-Shape---------->  <-----Cross---->  <---------Rects--------->
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        dtype=torch.float32,
    )


@pytest.fixture
def size_scenario2():
    return (5, 6, 3)


@pytest.fixture
def sparse_scenario2(indices_scenario2, values_scenario2, size_scenario2):
    return indices_scenario2, values_scenario2, size_scenario2


@pytest.fixture
def bbox_scenario2():
    return torch.tensor(
        [
            0.2,  # xmin
            1 / 6,  # ymin
            0.8,  # xmax
            5 / 6,  # ymax
        ],
        dtype=torch.float32,
    )


@pytest.fixture()
def scenarios_sparse(request, sparse_scenario1, sparse_scenario2):
    if request.param == 1:
        return sparse_scenario1
    elif request.param == 2:
        return sparse_scenario2
    raise NotImplementedError(f"No such scenario: {request.param!r}")


@pytest.fixture()
def scenarios_bbox(request, bbox_scenario1, bbox_scenario2):
    if request.param == 1:
        return bbox_scenario1
    elif request.param == 2:
        return bbox_scenario2
    raise NotImplementedError(f"No such scenario: {request.param!r}")


@pytest.fixture()
def scenarios_sparse_bbox(request, sparse_scenario1, bbox_scenario1, sparse_scenario2, bbox_scenario2):
    if request.param == 1:
        return {
            "sparse": sparse_scenario1,
            "bbox": bbox_scenario1,
        }
    elif request.param == 2:
        return {
            "sparse": sparse_scenario2,
            "bbox": bbox_scenario2,
        }
    raise NotImplementedError(f"No such scenario: {request.param!r}")
