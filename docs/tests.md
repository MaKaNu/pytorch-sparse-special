# Comparison of sparse iou calculation against normal matrices iou calculation

The calculation for sparse is looking like the following:

1. determine the indices of the sparse which are located inside the bbox.
2. calculate area of the bbox normalized to pixels
3. Intersection is equal to the sum of pixels inside
   1. Calculate the normalized area for the pixels inside.
   2. area of single $pixel = (1 / total)^2$
   3. sum up area
4. Union equals sum of all normalized pixels plus area of bbox minus Intersection.

**Formulas**

$$area_{total} = pixelsize^2 \cdot count_{total}$$

$$area_{inside} = pixelsize^2 \cdot count_{inside}$$

$$area_{bbox} = (xmax -xmin) \cdot (ymax - ymin)$$

$$iou =  {area_{inside}\over area_{bbox} + area_{total} - area_{inside}}$$

## Scenario 1: Simple Square

![class1](./assets/class1.png)
![class2](./assets/class2.png)
![class3](./assets/class3.png)

The three example masks with given bbox are defined as the following:

- total image size: 5 x 5

|                  | Pixels T-shape | Pixels Cross | Pixels Rects |
| ---------------- | -------------- | ------------ | ------------ |
| $count_{total}$  | 9              | 5            | 7            |
| $count_{inside}$ | 3              | 5            | 4            |
| $area_{total}$   | 0.36           | 0.2          | 0.28         |
| $area_{inside}$  | 0.12           | 0.2          | 0.16         |
| $iou$            | 0.2            | ${5\over9}$  | ${1\over3}$  |

|                | bbox |
| -------------- | ---- |
| $xmin$         | 0.2  |
| $ymin$         | 0.2  |
| $xmax$         | 0.8  |
| $ymax$         | 0.8  |
| $area\_{bbox}$ | 0.36 |

## Scenario 2: Rectangle unequal ratio

![class1_v2](./assets/class1_v2.png)
![class2_v2](./assets/class2_v2.png)
![class3_v2](./assets/class3_v2.png)

The three example masks with given bbox are defined as the following:

- total image size: 5 x 6

|                  | Pixels T-shape                       | Pixels Cross                         | Pixels Rects                                |
| ---------------- | ------------------------------------ | ------------------------------------ | ------------------------------------------- |
| $count_{total}$  | 10                                   | 6                                    | 9                                           |
| $count_{inside}$ | 4                                    | 6                                    | 6                                           |
| $area_{total}$   | 0.4                                  | 0.24                                 | 0.36                                        |
| $area_{inside}$  | 0.16                                 | 0.24                                 | 0.24                                        |
| $iou$            | ${0.16\over0.4 + 0.4 - 0.16} = 0.25$ | ${0.24\over0.4 + 0.24 - 0.24} = 0.6$ | ${0.24\over0.4 + 0.36 - 0.24} = {6\over13}$ |

|               | bbox                                    |
| ------------- | --------------------------------------- |
| $xmin$        | ${1\over5}$                             |
| $ymin$        | ${1\over6}$                             |
| $xmax$        | ${4\over5}$                             |
| $ymax$        | ${5\over6}$                             |
| $area_{bbox}$ | ${3\over5} \cdot {4\over6} = {2\over5}$ |

### Comparison to simple pixel count method

In the simple pixel count method we just count the area based on pixel.
Since our bounding box exactly closes on pixel we are able to do so.

**Formula IoU (Pixel)**

$$IoU = {count_{inside}\over count_{total} + count_{bbox} -  count_{inside}}$$

#### Scenario 1

|        | Pixels T-shape             | Pixels Cross               | Pixels Rects               | Pixels bbox |
| ------ | -------------------------- | -------------------------- | -------------------------- | ----------- |
| total  | 9                          | 5                          | 7                          | 9           |
| inside | 3                          | 5                          | 4                          | -           |
| IoU    | ${3\over 9 + 9 - 3} = 0.2$ | ${5\over 9 + 5 - 5} = 5/9$ | ${4\over 9 + 7 - 4} = 1/3$ | -           |

Scenario 1 results are equal, so for squared objects this approach is correct.

#### Scenario 2

|        | Pixels T-shape                 | Pixels Cross                | Pixels Rects                | Pixels bbox |
| ------ | ------------------------------ | --------------------------- | --------------------------- | ----------- |
| total  | 10                             | 6                           | 9                           | 12          |
| inside | 4                              | 6                           | 6                           | -           |
| IoU    | ${4\over 12 + 10 - 4} = 2 / 9$ | ${6\over 12 + 6 - 6} = 0.5$ | ${6\over 12 + 9 - 6} = 0.4$ | -           |

Comparing the results with above we can see they are not equal.
The reason could only be inside the normalisation.
The issue is probably in the calculation of the normalised size of the bbox or the pixels.

##### Validate the Pixels

Instead of calculating the pixels even in both directions lets calculate the $pixelsize$ depended on the direction:

$$ pixelsize^2 = pixel*{height} \cdot pixel*{width}$$

with $pixel_{height} = {1\over6}$ and $pixel\_{width} = {1\over5}$

|                  | Pixels T-shape                                              | Pixels Cross                      | Pixels Rects                      |
| ---------------- | ----------------------------------------------------------- | --------------------------------- | --------------------------------- |
| $count_{total}$  | 10                                                          | 6                                 | 9                                 |
| $count_{inside}$ | 4                                                           | 6                                 | 6                                 |
| $area_{total}$   | ${1\over3}$                                                 | 0.2                               | 0.3                               |
| $area_{inside}$  | ${2\over15}$                                                | 0.2                               | 0.2                               |
| $iou$            | ${{2\over15}\over0.4 + {1\over3} - {2\over15}} = {2\over9}$ | ${0.2\over0.4 + 0.2 - 0.2} = 0.5$ | ${0.2\over0.4 + 0.3 - 0.2} = 0.4$ |

We might be able to solve the same issue from the bbox side, but since we validated, that the normalisation was the issue we can now fix this issue.

## Scenario 3:

#TODO

## Scenario 4:

#TODO

## Scenario 5: Mask Outside

![class2](./assets/class2.png)
![class3](./assets/class4.png)

The two example masks with given bbox are defined as the following:

- total image size: 5 x 5

|                  | Pixels Cross | Pixels L-Shape |
| ---------------- | ------------ | -------------- |
| $count_{total}$  | $5$          | $5$            |
| $count_{inside}$ | $5$          | $0$            |
| $iou$            | ${5\over9}$  | ${0\over14}$   |

|                | bbox |
| -------------- | ---- |
| $xmin$         | $1$  |
| $ymin$         | $1$  |
| $xmax$         | $4$  |
| $ymax$         | $4$  |
| $area\_{bbox}$ | $9$  |

The test should show that, even so the calculation for the second example provides zero iou the value is also created.
