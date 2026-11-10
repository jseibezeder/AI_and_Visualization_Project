# Archived editor and hypothesis workbench

**Status: paused prototype, not part of the supported analysis or a finished
application release.** The latest raw pixel-buffer/browser transport changes were
unfinished when development stopped; earlier browser-validation reports do not
validate that later state. Archiving changes only paths/import setup, not features
or scientific behavior. The current notebooks do not depend on this application.

## Application-specific files

- `image_sandbox.py` — localhost HTTP server, shared frozen-checkpoint inference
  engine and original raw-28 transformation functions.
- `hypothesis_workbench.py` — experiment/cohort/ROI/comparison backend.
- `index.html` — complete frontend, including drawing/right-button erasing and
  the paused canonical pixel-buffer work.
- `development_notes.md` — unmodified pre-archiving project README draft, including
  the proposed workbench features. Its old filenames/instructions are historical,
  not a claim that every feature is validated.
- `../requirements.txt` — all extra archived-code dependencies, referencing shared
  root requirements rather than duplicating them.

## Shared inputs (deliberately not duplicated)

From the repository root, the prototype uses:

- `data/bloodmnist.npz` (TEST images only).
- Current caches in `predictions/`: baseline model archives, confidence analysis,
  targeted follow-up, stability and occlusion.
- `archived/predictions/test_pca.npz` for optional pixel-PC probes.
- The two active CNN weights in `weights_bloodmnist/`, local `models/vit/`, and
  `models/resnet18/models.py` for the supplied ResNet50 architecture.

The server resolves these from its own location and loads its UI from this
folder's `index.html`. Application source hashes are likewise resolved locally.
No root-level `image_sandbox.py`, `hypothesis_workbench.py` or second UI folder is needed.

For deliberate future development only, from the repository root:

```bash
python -m pip install -r archived/requirements.txt
python archived/workbench/image_sandbox.py
```

This binds to `http://127.0.0.1:8765` (or pass `--port 8766`) and requires actual
model inference. It is not launched by the analysis notebooks. Original labels
may be invalid after editing, masks are not expert segmentations, selected-cohort
statistics are exploratory, and exact unedited/edited pixel transport must be
verified before drawing conclusions from this prototype.

For supported interactive results, use the root
`visualizations/transformation_explorer.html`, which needs no Python server.
