# AI_and_Visualization_Project

Project github for the course AI_and_Visualization.

Initial assets for visualizing BloodMNIST and comparing a pretrained CNN with a
Vision Transformer. The original dataset and both models are included, not just
links to external downloads.

## Contents

```text
data/
  bloodmnist.npz              # Original 28 x 28 RGB dataset (~33.8 MiB)
  metadata.json              # Official splits and eight class labels
  README.md, LICENSE
models/
  resnet18/
    resnet18_28_1.pth         # Official MedMNIST CNN, run 1 (~42.7 MiB, LFS)
    models.py                # Matching upstream architecture
    README.md, LICENSE
  vit/
    model.safetensors        # KiViDrag BloodMNIST ViT (~327.3 MiB, LFS)
    config.json
    preprocessor_config.json
    UPSTREAM_README.md
    README.md, LICENSE
requirements.txt
```

## Get started

Install [Git LFS](https://git-lfs.com/) before cloning:

```bash
git lfs install
git clone https://github.com/jseibezeder/AI_and_Visualization_Project.git
cd AI_and_Visualization_Project
git lfs pull
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell (use this instead)
.venv\Scripts\Activate.ps1
```

Then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Use Python 3.10+ with a supported PyTorch build. For GPU acceleration, install
the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for
your platform. Loading and preprocessing examples for both models are linked
below; they use local files without additional model downloads.

Both checkpoints use **Git LFS** because the ViT exceeds GitHub's 100 MiB normal
Git file limit. The dataset is ordinary Git data. A full checkout downloads
about 404 MiB of dataset and model weights. If a checkpoint is only a tiny text
pointer, run `git lfs pull`. GitHub ZIP downloads may contain pointers rather
than weights; cloning with Git LFS is recommended. LFS storage/bandwidth counts
against the repository owner's quota.

## Dataset and models

- [BloodMNIST data, labels, splits and attribution](data/README.md): 17,092 images,
  eight classes, fixed train/validation/test splits.
- [MedMNIST ResNet-18](models/resnet18/README.md): official 28 x 28 checkpoint and
  matching small-image architecture, with loading/preprocessing example.
- [BloodMNIST ViT](models/vit/README.md): complete local Hugging Face inference
  files and loading/preprocessing example; input is resized to 224 x 224.

Both models share the official class order. Their preprocessing differs; use
the documented pipeline for each. The ViT model card leaves some training-data
preprocessing details unspecified, so its reported validation score should not
be treated as reproduced on this checkout.

## Collaboration

The initial assets are set up directly on `main` for this setup only. Subsequent
work should use branches, including `HeuVi` and `TheJak`, with changes merged
back through pull requests.

## Sources and licenses

- [MedMNIST dataset](https://zenodo.org/records/10519652) and
  [project](https://github.com/MedMNIST/MedMNIST): BloodMNIST, CC BY 4.0.
- [Official experiment code](https://github.com/MedMNIST/experiments): Apache-2.0;
  [pretrained weights](https://zenodo.org/records/7782114): CC BY 4.0.
- [KiViDrag/ViT_bloodmnist_std_15](https://huggingface.co/KiViDrag/ViT_bloodmnist_std_15):
  Apache-2.0 as declared by its publisher.

Upstream files are unmodified. Attribution and applicable license notices are
kept with each asset; no single replacement license is imposed on them.
This project is for research and teaching, not clinical diagnosis.
