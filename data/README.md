# BloodMNIST

The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic
or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. \[from [original paper](https://www.nature.com/articles/s41597-022-01721-8)\]

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
