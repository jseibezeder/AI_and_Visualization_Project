# Archived material

This folder contains historical analysis and paused development—not additional
current results. Start with the three numbered notebooks in the repository root.
Nothing was deleted or numerically changed by the reorganization.

| Location | Contents |
|---|---|
| `analysis_history.ipynb` | Complete superseded notebooks, code, figures, outputs and interpretations, plus snapshots taken before each consolidation/rename. Historical paths are intentionally preserved. **Not a Run-All entry point.** |
| `legacy_visualization/` | The former singular `visualization/` folder. Its original computational notebook is preserved in the history; the later redirect is retained here. |
| `predictions/` | Older checkpoint runs, train/validation exports, CSV copies, derived HC-error exports and PCA archives not read by the current notebooks. |
| `checkpoints/` | Unused CNN runs, plus the original bundled 28px MedMNIST reference under `original_medmnist/`. The two active CNN checkpoints remain in the root `weights_bloodmnist/` folder. |
| `workbench/` | All application-specific source, UI and development notes for the paused editor/hypothesis workbench. This is not a finished/validated application release. |
| `requirements.txt` | Archived-only dependencies, including SciPy, scikit-learn and threadpoolctl; shared notebook/replay dependencies are referenced from the root requirements. |
| `outputs/` | Local resumable arrays, mask reviews and intermediate caches moved from the old root `outputs/`. Ignored by Git, so not present in a fresh clone unless recomputed. |

## Selective historical replay

The history preserves old code and claims as a record, not as one executable
pipeline. Earlier sections can contain superseded findings, train/validation jobs,
expensive model queries and filenames from before folder cleanup. Do not Run All.
Start a selected replay from the repository root, install `archived/requirements.txt`
and read the section's cohort/provenance first.

Relocations of historical input paths:

- Old `predictions/<file>` files not among the eight active NPZ caches are now
  `archived/predictions/<file>`.
- Unused `weights_bloodmnist/<checkpoint>` files are now
  `archived/checkpoints/<checkpoint>`.
- The original `models/resnet18/resnet18_28_1.pth` is now
  `archived/checkpoints/original_medmnist/resnet18_28_1.pth`; its old notes are kept
  as `README_before_move.md`, and the updated shared-code notes remain in `models/resnet18/`.
- The old root application scripts/UI are under `archived/workbench/`.
- Old local `outputs/` caches are under `archived/outputs/`.

The explicitly marked targeted-driver replay cell has its **caller** adapted to
these import/cache locations and writes a replay artifact under `archived/outputs/`
instead of replacing `predictions/targeted_followup.npz`. Its original recorded
experiment source is unchanged. Other historical snapshots retain original paths;
consult this mapping before deliberately replaying them.

Shared data, active model files and upstream architecture code remain in the root
`data/`, `models/` and `weights_bloodmnist/` directories because both the current
notebooks and archived replay need them. They are not duplicated or moved out of
the supported workflow. Model/source license and attribution files remain with
the corresponding assets.
