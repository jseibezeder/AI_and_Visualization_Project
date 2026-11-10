# BloodMNIST: black-box model behavior

Comparison of frozen ResNet18, supplied ResNet50 and ViT pipelines using only
**3,421 TEST images**. The analysis studies predictions, confidence and responses
to input changes—not internal activations or a causal explanation of cell morphology.

## Read and run

Run these notebooks from the repository root, in order:

1. **[01_baseline.ipynb](01_baseline.ipynb)** — verify cached TEST predictions,
   labels, logits and frozen checkpoint hashes; optional checkpoint replay is disabled.
2. **[02_model_behavior.ipynb](02_model_behavior.ipynb)** — disagreement, confidence,
   input sensitivity, matched error properties and the focused IG/neutrophil investigation.
   Complete error galleries and the optional ROI study follow the conclusions as supplements.
3. **[03_robustness.ipynb](03_robustness.ipynb)** — the two full-test transformation
   grids: label changes and accuracy.

See **[RESULTS.md](RESULTS.md)** for conclusions, reproducibility and limitations.
The notebooks run from committed binary caches without training or new inference.
Open **[visualizations/transformation_explorer.html](visualizations/transformation_explorer.html)**
for the self-contained, offline interactive visualization; no model server is needed.

With Git LFS installed, obtain the active model files if necessary:

```bash
git lfs pull --include="models/vit/*,weights_bloodmnist/*"
python -m venv .venv
```

Activate `.venv` (`source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell), then:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Select that environment's Python kernel. Use Python 3.10+ with a supported PyTorch
build; CUDA is optional and is not used by the default cache-only notebook workflow.
Historical archived checkpoints do not need to be downloaded to read/run the results.

## Repository layout

```text
01_baseline.ipynb
02_model_behavior.ipynb
03_robustness.ipynb
RESULTS.md
visualizations/                 # Active offline transformation explorer only
predictions/                    # Eight NPZ caches used by the current notebooks
weights_bloodmnist/              # Only the two current CNN checkpoints
models/                         # Shared architecture code, local ViT and attribution
data/                           # Original dataset, class order and licenses
agent/                          # Agent-only notes; local scripts/logs/backups ignored
archived/
  analysis_history.ipynb         # All superseded analysis/code/figures; not Run-All
  legacy_visualization/          # Former singular visualization/ entry point
  predictions/                   # Historical runs, train/val exports, CSV copies, PCA
  checkpoints/                   # Unused experiment runs and the original 28px reference
  workbench/                     # Paused editor/workbench, UI and development notes
  requirements.txt               # Extra dependencies for archived work only
  outputs/                       # Local intermediate caches (ignored by Git)
```

`visualization/` was an older exploration with a different ResNet50 result; it was
not a second supported visualization pipeline. Only the plural `visualizations/`
folder remains active. Nothing was deleted: see [archived/README.md](archived/README.md)
for relocated artifacts and the paused workbench's status.

## Frozen pipelines and shared assets

- ResNet18: `weights_bloodmnist/resnet18_224_1.pth`, nearest-neighbor resize to224.
- ResNet50: `weights_bloodmnist/resnet50_28_1.pth`, native28, using the upstream
  `ResNet50` class retained in `models/resnet18/models.py`.
- ViT: local `models/vit/` weights/configuration and its bilinear224 processor.

These checkpoints and preprocessing are not interchangeable with historical runs.
Data, active checkpoints and shared model code stay outside `archived/` because the
current notebooks and optional replay need them; the archived workbench reuses these
assets rather than duplicating large files. Its additional dependencies and all
application-specific source/UI are archived together.

## Sources and licenses

- [BloodMNIST / MedMNIST](data/README.md): dataset attribution, official splits,
  eight labels and CC BY4.0 terms.
- [MedMNIST experiment architecture](models/resnet18/README.md): upstream Apache-2.0
  code, historical checkpoint provenance and [weights release](https://zenodo.org/records/7782114).
- [Local KiViDrag ViT](models/vit/README.md): source model card, Apache-2.0 terms
  and preprocessing notes.

Original dataset/model binaries and upstream source/license files are unchanged;
no single replacement license is imposed on the assets. This is a course research
and teaching project, **not clinical diagnosis**. Continue collaboration on branches
such as `HeuVik2` and merge through the project's usual review process.
