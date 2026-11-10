# Presentation and repository handoff

## Latest interactive presentation variant

- Added `presentation_view_2.ipynb`, keeping every cell/figure from the code-free
  view and adding the original transformation explorer after the robustness curves.
- Explorer data and canonical `visualizations/transformation_explorer.html` are
  unchanged. The new view embeds the full HTML in a scripts-only sandbox. No
  workbench, model runtime, external assets or new inference is included.
- Presentation-specific sizing and a Full screen button are added to the embedded
  copy only. Dropdown, all 80 cases, level selection, linked zoom buttons, wheel,
  drag, fit, reset and full screen were tested in JupyterLab at 1366x650.
- Interactive block is 528.36px in the 529px notebook area, without internal
  scrolling. One source-hidden cell reconstructs the HTML without display warnings.
- Notebook is signed locally. Other users may need Jupyter's Trust Notebook
  command once for active HTML; this cannot be embedded as portable trust.
- Helpers/checks/screenshots: `runs/presentation-view-2/`. All earlier notebooks,
  figures, caches and datasets remain unchanged. Preview server is stopped.

## Previous presentation delivery

- Added `presentation_view.ipynb`, a Markdown-only screen-sharing copy. Its 15
  cells embed the same seven PNGs as the executable presentation, with no code
  cells or collapsed-input bars. `presentation.ipynb` stays byte-for-byte unchanged.
- Added `presentation_aid.md` with brief unslopped talking points for all eight
  sections. It is separate from both notebooks. The aid also explains confidence
  thresholds, overlapping pair counts, the resizing tradeoff and causal limits.
- Actual JupyterLab inspection at 1366x650 confirmed zero code cells/placeholders,
  seven rendered attachment images and a largest section of about 456px in the 529px
  notebook area. All source images, original analyses and scientific assets match.
- Helpers, screenshots and checks are in `runs/presentation-view/`.

## Previous executable presentation

- Created `presentation.ipynb` for the BloodMNIST Phase I midpoint presentation.
  It has 15 cells, six main scientific figures, a compact opening image strip
  and three final findings. No timing annotations, notes or scripts are visible.
- All images are embedded PNG outputs. Code uses native source-hidden metadata
  and blank first lines so collapsed inputs do not expose code previews.
- Clean-kernel execution passed without inference. Cached values and both
  directions' appearance bootstrap means/intervals exactly match the original
  analysis calculations. No original notebook, dataset, checkpoint or cache changed.
- Inspected all eight blocks in JupyterLab at a 1366x650 viewport. The notebook
  area is 529px high; the largest section is 499.3px. No visible code or clipped
  images. Native collapsed-input ellipses remain as normal notebook controls.
- Estimated speaking duration is 2:45 without the opening strip, 2:55 with it.
  Modern JupyterLab is verified; older Classic Notebook may ignore source hiding.
- Detailed checks, screenshots, helpers and source hashes are in
  `runs/presentation/`. Preview server is shut down after inspection.

## Previous all-error class summary

- Added per-class recall and HC/all-error counts to `02_model_behavior.ipynb`
  before the HC-only analysis. All 3,421 TEST predictions contribute; the total
  row reports overall accuracy and errors, not macro-averaged recall.
- IG recall is 90.16/90.67/94.65%; neutrophil recall is 98.20/97.15/97.45%.
  IG HC/all-error counts are 36/57, 28/54 and 18/31. The counts do not establish
  per-class calibration or a biological cause.
- Main narrative now has 57 cells and 12 figures. No added figure or inference.
- Fresh execution passed in 14.82s. Independent per-model cache checks reproduce
  every confusion count, recall and threshold count. Existing scientific code
  and outputs, all original cells, data/weights and historical text are unchanged.
- New prose and existing interpretations were reviewed against unslop. Helpers,
  original snapshots and validation are under `runs/all-class-performance/`.

## Previous prose revision

- Applied the manually invoked Cursor `unslop` skill to all Markdown cells in
  the three current notebooks, plus `README.md` and `RESULTS.md`.
- Replaced repetitive interpretation templates, dense shorthand and ornamental
  punctuation with direct prose. Retained cohort definitions, numerical claims,
  uncertainty qualifications, citations and supplementary links.
- Code cells, outputs, cell order, notebook metadata, historical text, scientific
  data and model files remain unchanged. No inference or experiments were run.
- Originals and validation are under `runs/unslop/`, ignored by Git.
- User requested the exact commit subject `rework analysis`.

## Previous presentation refinement

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
