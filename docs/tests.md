# Comparison of sparse iou calculation against normal matrices iou calculation

The calculation for sparse is looking like the following:

1. determine the indices of the sparse which are located inside the bbox.
2. calculate area of the bbox.
3. Intersection is equal to the sum of mask pixels inside.
4. Union is equal the sum of all mask pixels plus area of bbox minus Intersection.

**Formulas**

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
| $count_{total}$  | $9$            | $5$          | $7$          |
| $count_{inside}$ | $3$            | $5$          | $4$          |
| $iou$            | ${3\over15}$   | ${5\over9}$  | ${4\over12}$ |

|                | bbox |
| -------------- | ---- |
| $xmin$         | $1$  |
| $ymin$         | $1$  |
| $xmax$         | $4$  |
| $ymax$         | $4$  |
| $area\_{bbox}$ | $9$  |

## Scenario 2: Rectangle unequal ratio

![class1_v2](./assets/class1_v2.png)
![class2_v2](./assets/class2_v2.png)
![class3_v2](./assets/class3_v2.png)

The three example masks with given bbox are defined as the following:

- total image size: 5 x 6

|                  | Pixels T-shape | Pixels Cross | Pixels Rects |
| ---------------- | -------------- | ------------ | ------------ |
| $count_{total}$  | $10$           | $6$          | $9$          |
| $count_{inside}$ | $4$            | $6$          | $6$          |
| $iou$            | ${4\over18}$   | ${6\over12}$ | ${6\over15}$ |

|               | bbox |
| ------------- | ---- |
| $xmin$        | $1$  |
| $ymin$        | $1$  |
| $xmax$        | $4$  |
| $ymax$        | $5$  |
| $area_{bbox}$ | $12$ |

## Scenario 3: invalid single point, invalid size

A single Pixel.

Variable `indices` is valid, but the size of image is 2D.

Should raise a SizeValueError.

## Scenario 4: valid single point, valid size

A single Pixel.

Variables `indices` is invalid (two axis), size of image is 3D.

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
