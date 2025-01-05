# Examples

This Page gives examples and explanations how this library can be used.

## Mask Tensor

A mask tensor is a mono value index matrix. It describes at which pixels in an image the pixel is assigned to class index.

A mask Tensor is an 3D Tensor with following dimensions:

$$Shape = [W, H, N]$$

with $W$ as the width of the image, $H$ as the height of the image and $N$ as the number of masks.

To create a Mask Tensor use the [MaskSparseTensor](modules.md/#pytorch_sparse_special.special.sparse_mask.SparseMasksTensor) class with an index and a value tensor as additional attributes to the size:

```python
import torch

from pytorch-sparse-special import MaskSparseTensor

indices = torch.tensor(
    [
      [0, 1, 1, 2, 0, 0, 2, 2, 1],
      [1, 0, 2, 1, 0, 2, 0, 2, 1],
      [0, 0, 0, 0, 1, 1, 1, 1, 2],
    ],
)

values = torch.tensor(
  [1,1,1,1,3,3,3,3,2]
)

size = (3,3,3)

tensor = MaskSparseTensor(indices, values, size)

```

**indices** define the position of the pixel in the tensor. They have to match the 3D space.

**values** define the class index for each pixel. Shape has to match the Number of pixels of indices.

For additional information about the methods implemented to the class refer [modules](modules.md/#pytorch_sparse_special.special.sparse_mask.SparseMasksTensor.area_per_mask) section.
