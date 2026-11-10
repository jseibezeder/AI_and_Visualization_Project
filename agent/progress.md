# Repository cleanup handoff

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
- All original notebook cells/outputs remain preserved in the 396-cell history;
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
