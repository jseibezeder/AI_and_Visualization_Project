# Presentation and repository handoff

## Latest presentation refinement

- `02_model_behavior.ipynb`: counts-only confusion annotations; reliability-bin
  counts and hollow low-support markers (n<30), without connecting sparse/empty bins.
- Main narrative: 54 cells/12 figures. Full shared/exclusive galleries and optional
  ROI investigation remain as Supplements A/B (four more figures).
- Historical matching comparisons and paused-workbench commentary removed from
  the main narrative; scientific cohort distinctions/limitations retained.
- Full preceding 64-cell notebook preserved in `archived/analysis_history.ipynb`,
  now 461 cells. No scientific code, values, caches or inference changed.
- Fresh execution, exact unmodified-output checks, ECE/count equivalence and links
  validated; see `runs/presentation-polish/validation.json`. The two changed
  figures were visually reviewed. Helpers/previews stay under this agent folder.

## Previous repository cleanup

Branch: `HeuVik2`. Scientific baseline before cleanup: `076dd50`.

## Current layout

- Supported notebooks: `01_baseline.ipynb`, `02_model_behavior.ipynb`, `03_robustness.ipynb`.
- Only active visualization folder: `visualizations/` (offline explorer).
- Historical notebooks, unused predictions/checkpoints and paused application:
  `archived/`; read its README before any selective historical replay.
- Agent helpers/logs/handoffs collected under `agent/runs/`, ignored by Git.
  `runs/relocations.json` records 103 moved files and their move-time hashes.
  Scientific image/HTML previews and intermediate arrays in web uploads were left
  in place; global Pi configuration and unrelated projects were not touched.

## Verified

- All three current notebooks pass fresh-kernel execution without inference:
  baseline 2.16s, behavior 8.94s (16 figures), robustness 2.68s (2 figures).
- Active data/checkpoints/caches and the canonical explorer are unchanged.
- 68 historical prediction/checkpoint payload moves checked against original Git
  blobs/LFS hashes; CSV checkout line endings accounted for.
- All original notebook cells/outputs were preserved in the 396-cell cleanup history;
  original code ASTs match all current numerical code.
- Current navigation, archived-script compilation, JS syntax and dependency
  coverage pass. The optional targeted replay caller writes under ignored
  `archived/outputs/`, never over the published cache.
- Archived application imports and cache-only Workbench initialization pass:
  relocated PCA input hash, notebook feature equivalence and local source hashes.
  No server or model inference was launched.
- Original unfinished UI and README draft are preserved byte-for-byte;
  application source changes are restricted to relocation/import/hash paths.

Detailed cleanup validation: `runs/repo-cleanup/validation.json`.
Historical one-shot helpers are records, not safe general-purpose replay scripts.

## Paused development

The application remains unfinished. Its latest canonical pixel-buffer/browser
transport patch is not browser-validated, and old browser validation does not
validate that patch. Do not resume feature work or treat the prototype as
scientific evidence without a new request. No experiment jobs are pending.
