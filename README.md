# AI_and_Visualization_Project

Project github for the course AI_and_Visualization.

Visualizing BloodMNIST and comparing frozen ResNet18, supplied ResNet50 and
Vision Transformer pipelines. The dataset and local checkpoints are included;
all reported analysis uses only the 3,421 TEST images.

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

Large checkpoints use **Git LFS**; the ViT exceeds GitHub's 100 MiB normal
Git file limit. The dataset and prediction caches are ordinary Git data. If a checkpoint is only a tiny text
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

## Analysis: four connected stories

Open `confidence_analysis.ipynb` in JupyterLab (`jupyter lab`) for:

1. **Where do models disagree?** Shared/exclusive errors, image galleries, class
   rates/substitutions and focused immature-granulocyte/neutrophil competition.
2. **Can we trust confidence?** Reliability and confidently wrong images paired
   one-to-one with same-class correct images within 1 confidence percentage point.
3. **Which changes destabilize predictions?** Selected tiny-change flips with all
   eight probabilities, interpolation/noise controls and an offline transition explorer.
   `stability_analysis.ipynb` supplies the full-test, common-scale label-change/accuracy grids.
4. **What might drive differences?** Saturation destinations, tightly matched
   appearance/occlusion with paired-image bootstrap intervals, and small balanced
   approximate ROI color probes with visually reviewed core/expanded masks.

The main notebooks execute from caches, without loading models or repeating inference.
`archive.ipynb` preserves **complete original and intermediate notebooks**, outputs,
removed entropy/PCA analyses and the reproducible targeted-experiment driver.
Its replay cell is optional and does run actual inference. The targeted results are
in `predictions/targeted_followup.npz`; existing full-test prediction files are unchanged.

**Provenance:** ResNet18 uses `weights_bloodmnist/resnet18_224_1.pth` with nearest
224 resizing, ResNet50 uses `weights_bloodmnist/resnet50_28_1.pth` at native 28,
and ViT uses `models/vit/` with its local bilinear-224 processor. These are not
interchangeable with the original 28px ResNet18 asset listed above. Matching,
ROI and interpolation cohort sizes/selection and uncertainty units are explicit
in the notebooks; comparisons concern checkpoints **plus preprocessing**, not
architecture alone. Approximate ROIs are not expert segmentations and do not
cleanly separate anatomical foreground/background or equalize intervention dose.

### Live image sandbox — actual checkpoint inference

From the repository root, with the virtual environment activated:

```bash
python image_sandbox.py
```

Open **http://127.0.0.1:8765** (or pass `--port 8766`). Startup verifies the frozen
checkpoint hashes and baseline probabilities; models remain loaded, inference is
serialized and CUDA is used when available, otherwise CPU. No model downloads,
training, public network binding or external image uploads are performed.

- Load a random TEST image, a confident error, an IG/neutrophil error, or a TEST ID.
- Draw, gray-mask, restore original pixels, or reset the 28×28 editing canvas.
- Apply brightness/saturation/channel changes, translation, rotation, blur, contrast
  or deterministic noise, and inspect the actual transformed input.
- See all eight probabilities for **all three actual models**, with original cached
  probabilities in gray and edited probabilities in color; edits trigger debounced inference.

The offline `visualizations/transformation_explorer.html` only explores cached
cohort transitions and needs no Python server. The live sandbox requires its local
server and model dependencies; it is not an offline prediction lookup. Edited images
can leave the data distribution or invalidate the original label. Stop the local
server with Ctrl+C when launched in a terminal; all analysis is research/teaching only.

## Collaboration

The initial assets are set up directly on `main` for this setup only. Subsequent
work should use branches, including `HeuVik` and `TheJak`, with changes merged
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
