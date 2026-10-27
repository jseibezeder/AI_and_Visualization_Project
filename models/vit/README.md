# BloodMNIST Vision Transformer

Local, unmodified inference files from
[KiViDrag/ViT_bloodmnist_std_15](https://huggingface.co/KiViDrag/ViT_bloodmnist_std_15),
pinned to revision `8e5f52d1316b936012909bfac13bc8e3d6429a2b`.

- **Architecture:** `ViTForImageClassification`, fine-tuned from
  `google/vit-base-patch16-224`, with eight BloodMNIST output classes.
- **Files:** `model.safetensors`, `config.json`, `preprocessor_config.json`,
  and the original model card in `UPSTREAM_README.md`. Training logs and
  pickled training arguments are not required for inference and are omitted.
- **Input:** RGB images; the supplied processor resizes to 224 x 224 using
  bilinear interpolation, rescales by `1/255`, then normalizes with mean and
  standard deviation `0.5` per channel.
- **Storage:** ~327.3 MiB; weights are tracked using Git LFS.
- **License:** Apache-2.0 as declared by the publisher; full text in `LICENSE`.
  Attribution: KiViDrag; base model by Google. No model files were modified.

From the repository root (no Hugging Face network access required):

```python
import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("models/vit", local_files_only=True)
model = ViTForImageClassification.from_pretrained("models/vit", local_files_only=True)
model.eval()
with np.load("data/bloodmnist.npz", allow_pickle=False) as dataset:
    images = [Image.fromarray(image) for image in dataset["test_images"][:8]]
inputs = processor(images=images, return_tensors="pt")
with torch.inference_mode():
    probabilities = model(**inputs).logits.softmax(dim=1)
```

Pass raw uint8/PIL images to the processor, not already normalized tensors.
Its class mapping matches [`../../data/metadata.json`](../../data/metadata.json).
The published validation accuracy (~96.99%) is self-reported, not a locally
reproduced result. The model card does not fully specify training-data
preprocessing or the original MedMNIST resolution; using our 28 x 28 archive
upsampled to 224 x 224 is supported, but does not establish reproduction of
that metric or a like-for-like benchmark against the CNN.

SHA-256 and source URLs are in [`../../assets.json`](../../assets.json).
For research/teaching only, not clinical use.
