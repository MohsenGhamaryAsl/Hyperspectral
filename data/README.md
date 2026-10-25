# Data

| File | Content |
|---|---|
| `Indian_Pine.mat` | MATLAB file: `Pixels` (145, 145, 200) uint16 cube, `Trained_Data` (145, 145) uint8 ground truth, `Classes` (17, 1) names |
| `Indian_Pine.npz` | the same arrays in NumPy format: `Image_DataCube`, `Ground_Truth`, `Classes_List`, `Image_Dataset` |

Both files hold the same scene, so you can keep only one of them.

## Loading

```python
from hyperspectral import load_hypercube

cube = load_hypercube("data/Indian_Pine.mat")
print(cube.shape)          # (145, 145, 200)
print(cube.label_counts()) # {1: 20, 2: 26, ...}
```

## Using your own cube

`load_hypercube` expects a `(rows, columns, bands)` array and a
`(rows, columns)` label map. Save them with the usual names:

```python
import numpy as np
np.savez_compressed("my_scene.npz", Pixels=cube, Ground_Truth=labels)
cube = load_hypercube("my_scene.npz")
```

You can also build the object directly:

```python
from hyperspectral import Hypercube
cube = Hypercube(pixels=my_cube, ground_truth=my_labels, class_names=["Background", "A", "B"])
```

## About the Indian Pines scene

Recorded by AVIRIS over an agricultural test site in Indiana (USA); a standard
benchmark for hyperspectral classification research. The ground truth contains
16 classes (`0` = unlabelled). The data is included for reproducibility of the
examples; please respect the terms of use of the original data provider
(Purdue University; AVIRIS data courtesy of NASA JPL).
