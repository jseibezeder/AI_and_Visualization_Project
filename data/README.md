# BloodMNIST

`bloodmnist.npz` is the **unmodified original 28 x 28 RGB** BloodMNIST archive
from [MedMNIST's Zenodo release](https://zenodo.org/records/10519652). It contains
17,092 images across eight blood-cell classes and is 35,461,855 bytes (~33.8 MiB),
small enough for regular Git. Larger MedMNIST+ variants are not included.

| Split | Images | Image array | Label array |
| --- | ---: | --- | --- |
| Train | 11,959 | `train_images` | `train_labels` |
| Validation | 1,712 | `val_images` | `val_labels` |
| Test | 3,421 | `test_images` | `test_labels` |

Images are `uint8` arrays of shape `(N, 28, 28, 3)`, with values in `[0, 255]`.
Labels have shape `(N, 1)`. Preserve the official splits; use validation for
model selection and reserve test data for final evaluation.

| Label | Class |
| ---: | --- |
| 0 | basophil |
| 1 | eosinophil |
| 2 | erythroblast |
| 3 | immature granulocytes (myelocytes, metamyelocytes and promyelocytes) |
| 4 | lymphocyte |
| 5 | monocyte |
| 6 | neutrophil |
| 7 | platelet |

The exact upstream names, counts, and source checksum are in `metadata.json`.
The supplied CNN and ViT use this same label order.

```python
import numpy as np

with np.load("data/bloodmnist.npz", allow_pickle=False) as dataset:
    images = dataset["train_images"]  # Decompress each array once, not per sample.
    labels = dataset["train_labels"].reshape(-1)
```

Upstream MD5: `7053d0359d879ad8a5505303e11de1dc`.

## Attribution and license

BloodMNIST is distributed by the MedMNIST team under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see `LICENSE`.
Credit Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke,
Hanspeter Pfister, and Bingbing Ni. The archive is redistributed unchanged.

- [MedMNIST repository](https://github.com/MedMNIST/MedMNIST)
- [Dataset release, DOI: 10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652)
- Yang et al., *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification*, Scientific Data (2023),
  [DOI: 10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8).

This is a research/teaching benchmark, not a clinical diagnostic dataset or tool.
