# BloodMNIST model behavior

This course project compares frozen ResNet18, supplied ResNet50 and ViT models on
3,421 TEST images. It measures predictions, confidence and responses to input
changes. It does not inspect internal model representations or establish
biological causes of errors.

## Read and run

Run the notebooks from the repository root in this order:

1. [01_baseline.ipynb](01_baseline.ipynb) checks saved TEST predictions, labels,
   logits and checkpoint hashes. Optional checkpoint replay is off by default.
2. [02_model_behavior.ipynb](02_model_behavior.ipynb) reports recall and error counts
   for every class across all TEST predictions. It examines disagreement,
   confidence, input sensitivity and matched errors, then compares IG and
   neutrophil misclassifications. Full error galleries and the optional
   approximate-region study follow the conclusions as supplements.
3. [03_robustness.ipynb](03_robustness.ipynb) shows label-change rates and
   accuracy across the full TEST set under each transformation.

[RESULTS.md](RESULTS.md) summarizes findings and limitations. The notebooks read
saved binary results without training or new inference. The
[offline transformation explorer](visualizations/transformation_explorer.html)
needs no model server.

Install Git LFS and fetch the active model files if needed:

```bash
git lfs pull --include="models/vit/*,weights_bloodmnist/*"
python -m venv .venv
```

Activate `.venv` with `source .venv/bin/activate` on Linux or macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the dependencies
and open JupyterLab:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Select the environment's Python kernel. Use Python 3.10 or later with a supported
PyTorch build. CUDA is optional, and the default notebook runs do not use it.
You do not need the archived checkpoints to run the current notebooks.

## Repository layout

```text
01_baseline.ipynb
02_model_behavior.ipynb
03_robustness.ipynb
RESULTS.md
visualizations/                 # Offline transformation explorer
predictions/                    # Eight NPZ caches for the current notebooks
weights_bloodmnist/              # Two active CNN checkpoints
models/                         # Shared model code, local ViT and attribution
data/                           # Original dataset, class order and licenses
agent/                          # Agent notes and Git-ignored local working files
archived/
  analysis_history.ipynb         # Earlier code, plots and interpretations
  legacy_visualization/          # Former visualization/ entry point
  predictions/                   # Older runs, train/val exports, CSV copies and PCA
  checkpoints/                   # Unused runs and the original 28px reference
  workbench/                     # Paused editor, UI and development notes
  requirements.txt               # Extra dependencies for archived code
  outputs/                       # Local intermediate caches, ignored by Git
```

Only `visualizations/` is active. The old singular `visualization/` folder used a
different ResNet50 result and is archived. [archived/README.md](archived/README.md)
explains the moved files and paused workbench. Do not Run All in the historical
notebook.

## Models and preprocessing

- ResNet18 uses `weights_bloodmnist/resnet18_224_1.pth` and nearest-neighbor
  resizing to 224 pixels.
- ResNet50 uses `weights_bloodmnist/resnet50_28_1.pth` at native 28 pixels and the
  upstream `ResNet50` class in `models/resnet18/models.py`.
- ViT uses the local weights and configuration in `models/vit/` with bilinear
  resizing to 224 pixels.

Do not substitute historical checkpoints or preprocessing when reproducing these
results. Data, active weights and shared model code remain outside `archived/`
because the current notebooks and optional replay need them. The archived
workbench reuses these files. Its own source, UI and extra dependencies are archived.

## Sources and licenses

- [BloodMNIST and MedMNIST](data/README.md) documents dataset attribution,
  official splits, the eight labels and CC BY 4.0 terms.
- [MedMNIST model code](models/resnet18/README.md) documents upstream Apache-2.0
  code and checkpoint provenance. See the
  [weights release](https://zenodo.org/records/7782114).
- [Local KiViDrag ViT](models/vit/README.md) contains the model card,
  Apache-2.0 terms and preprocessing notes.

The original dataset, model binaries and upstream source and license files are
unchanged. Each asset retains its own license. This project is for research and
teaching, not clinical diagnosis. Use branches such as `HeuVik2` and the project's
usual review process for contributions.
