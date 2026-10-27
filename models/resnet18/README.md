# Official MedMNIST ResNet-18 (28 x 28)

- **Checkpoint:** `resnet18_28_1.pth`, official BloodMNIST run 1, unchanged.
- **Source:** `resnet18_28_1.pth` inside `weights_bloodmnist.zip` from
  [Zenodo record 7782114](https://zenodo.org/records/7782114).
- **Architecture:** the small-image ResNet-18 from
  [MedMNIST/experiments](https://github.com/MedMNIST/experiments), not
  `torchvision.models.resnet18`. `models.py` is an unmodified upstream copy at
  revision `70b6b3a7ad7afddff1df2a3b735235830fbdb142`.
- **Input:** RGB, `(N, 3, 28, 28)`, scaled from uint8 to `[0, 1]`, then
  normalized with mean `0.5` and standard deviation `0.5` per channel.
- **Output:** eight logits in the order documented in [`../../data/README.md`](../../data/README.md).
- **Storage:** ~42.7 MiB; tracked using Git LFS.

From the repository root:

```python
import numpy as np
import torch
from models.resnet18.models import ResNet18

model = ResNet18(in_channels=3, num_classes=8)
checkpoint = torch.load(
    "models/resnet18/resnet18_28_1.pth", map_location="cpu", weights_only=True
)
model.load_state_dict(checkpoint["net"], strict=True)
model.eval()

with np.load("data/bloodmnist.npz", allow_pickle=False) as dataset:
    images = dataset["test_images"][:8]
x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float()
x = (x / 255.0 - 0.5) / 0.5
with torch.inference_mode():
    probabilities = model(x).softmax(dim=1)
```

## Attribution and licenses

Checkpoint: MedMNIST team, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
per the [weights release](https://doi.org/10.5281/zenodo.7782114). Redistributed
unchanged. Credit Jiancheng Yang et al.; see [`../../data/README.md`](../../data/README.md)
for the MedMNIST paper and full author list.

Architecture code: MedMNIST Team, Apache-2.0 (full text in `LICENSE`); upstream
notes that it was adapted from `kuangliu/pytorch-cifar`. This code license does
not replace the checkpoint's CC BY 4.0 license.

The archive member was extracted using HTTP ranges and its ZIP CRC was checked;
other models and the full 1.46 GB archive are deliberately not included.
SHA-256 and source details are in [`../../assets.json`](../../assets.json).
