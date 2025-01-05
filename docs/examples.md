# Examples

This Page gives examples and explanations how this library can be used.

## Mask Tensor

A mask tensor is a mono value index matrix. It describes at which pixels in an image the pixel is assigned to a class index.

A mask tensor is an 3D tensor with following dimensions:

$$Shape = [N, H, W]$$

with $N$ as the number of masks, $H$ as the height of the image and $W$ as the width of the image, following the order of the dimensions like Pytorch.

As an Example we use this small example of these three 3 by 3 masks.
All three masks are displayed overlapping and since in this example only the zeros overlap we are able to see all three of them, with the $blue$ mask on layer 0,
the $red$ mask on layer 1 and the $orange$ mask on layer 2.

![sparse_example](./assets/sparse_example.png)

with $idx_{blue} = 1$ , $idx_{red} = 3$ and $idx_{orange} = 2$.

So the blue pixel in the upper middle has the indices $N=0$, $H=0$, $W=1$ and the $value=1$.
This is also the most left entry in the code example below.
Each mask in the code example is ordered from left to right blue -> red -> orange.

To create a Mask Tensor use the [MaskSparseTensor](modules.md/#pytorch_sparse_special.special.sparse_mask.SparseMasksTensor) class with an index and a value tensor as additional attributes to the size:

```{.py .copy}
import torch

from pytorch_sparse_special.special.sparse_mask import MaskSparseTensor

indices = torch.tensor(
    [
#      < blue   >  < red    >  < orange>
      [0, 0, 0, 0, 1, 1, 1, 1, 2],
      [0, 1, 1, 2, 0, 0, 2, 2, 1],
      [1, 0, 2, 1, 0, 2, 0, 2, 1],
    ],
)

values = torch.tensor(
#  < blue   >  < red    >  < orange>
  [1, 1, 1, 1, 3, 3, 3, 3, 2]
)

size = (3,3,3)

tensor = MaskSparseTensor(indices, values, size)

```

**indices** define the position of the pixel in the tensor. They have to match the 3D space.

**values** define the class index for each pixel. Shape has to match the Number of pixels of indices.

For additional information about the methods implemented to the class refer [modules](modules.md/#pytorch_sparse_special.special.sparse_mask.SparseMasksTensor.area_per_mask) section.
