# Experiment 4 — color and approximate cell/background dependence

Implementation: `skript/color_background.py`; tests: `skript/test_color_background.py`.
No training, downloads, weight changes, or modifications to previous experiments.

## Dataset, selection and masks

4A reuses **the exact ordered 48 test images from Experiment 3**, verified against `results/occlusion/subset/manifest.json`, labels and RGB SHA-256. Six images per class, intentionally enriched for disagreements: this is an explanation set, not an unbiased test population.

4B uses **16 of those images, two per class**. Individual ellipses were selected from original images before Experiment 4 inference, not chosen by new response magnitudes. The coding assistant visually checked the contact sheet and every original/contour/intervention-preview row. Details, exclusions, per-image limitations, reviewer identity and ellipse coordinates are saved in `masks/review.json`. **This is assistant visual review, not human/clinical expert approval.** Masks are approximate central-cell regions, not validated segmentations. Sample 2055 has an upper protrusion partly outside its ellipse; other contours can mix edge/background pixels. Peripheral regions often contain other cells. Check `pilot/masks.png` before presenting biological interpretations; a human can revise the mask review and run into a new output directory.

## 4A: color dependence

Fourteen conditions including the exact unmodified original:

- Grayscale (Pillow luminance conversion replicated to RGB).
- Saturation factors 0.5, 0.8, 1.2 and 1.5 (Pillow `ImageEnhance.Color`).
- Red and blue channel multipliers separately, 0.95 and 1.05, clipped and rounded to uint8.
- Brightness factors 0.9, 0.95, 1.05 and 1.1 (Pillow `ImageEnhance.Brightness`).

**Grayscale and saturation 0.5/1.5 are feature-ablation/stress tests.** "Mild" denotes the configured magnitude of the remaining changes; it does not establish plausible stain variation, diagnostic label preservation, or realistic microscope conditions. This is RGB manipulation, not stain-vector estimation/normalization. Brightness/channel changes may clip; actual pixel budgets and RGB intensity changes are exported.

`figures/color_probability_matrix.png` has one image per row and one condition per column, separate model panels on the same [-1,1] color scale. Values are transformed minus original probability of each model's own original decision. Numbers appear only at label flips and give the new class ID; row labels include the true and baseline class IDs. Different models' own target classes can differ. The complete vectors and change in **true-class** probability are exported too.

`figures/color_flip_difference.png` distinguishes ResNet-only and ViT-only flips. Its neutral value means both flip **or** neither flips, not necessarily equal decisions. `tables/mild_model_specific_flips.csv` contains every one-model-only flip under mild conditions. The gallery, if nonempty, shows at most eight cases ranked by absolute inter-model difference in probability change; ties use image ID then condition name. These are deliberately informative, not representative examples.

## 4B: central versus peripheral regions

### Whole-region ablations (unequal budgets)

Apply full-image Gaussian blur (sigma 2 px) or grayscale conversion, but replace pixels only within the reviewed central mask or its complement. Outside the selected region original bytes remain unchanged. This preserves an original, background-modified and central-region-modified version for every reviewed image.

These interventions differ in area and effective RGB change. They are descriptive context only; **do not interpret a larger effect after changing an entire cell as proof of cell preference relative to a smaller background change**. Pixel counts, actually changed counts, L1 RGB energy, selected-area mean absolute change and whole-image RGB RMS change are saved for every intervention.

### Exact matched controls

Within each region, select up to 64 eligible pixels, using the smaller available region count for both. Every selected channel must be in [16,239], so additive RGB shifts of -16, -8, +8 or +16 cannot clip. A paired central/peripheral intervention therefore has exactly equal:

- Nonzero changed pixel count (at least 24).
- Signed shift per RGB channel.
- Total L1 RGB intensity change and whole-image L2/RMS change.

Three reproducible spatial selections per image and region; each is reused for every signed magnitude. Save all selected masks. Average repetitions **within image/sign**, not as independent biological samples. These are dispersed-pixel controls, not contiguous whole-cell replacements. Equal area/energy does not match spatial texture, local contrast, or semantics. Eligible-pixel selection also constrains which bright/dark pixels can enter the comparison.

`matched_region_comparison.png` shows paired absolute probability responses. Above the diagonal means a greater central response **for that image/intervention**, not biological attribution. `whole_region_ablations.png` explicitly labels unequal budgets. Four `region_examples_*.png` pages show all 16 originals, both regional grayscale ablations, and a matched +16 pair, including predictions, p(original), altered area and L1 budget.

## Layout

```text
results/color_background/
  README.md, PROGRESS.md
  config.json                  # Fixed transformations and control budgets
  protocol.json                # Hashes of code, assets, exact source set, review, baselines
  summary.json                 # Aggregate results and limitations
  subset/manifest.json         # Exact reused 48 IDs and 16 region IDs
  masks/review.json            # Ellipses, review identity, acceptance and limitations
  pilot/{region_candidates,color_examples,masks}.png
  interventions/color/*.npz    # Shared original-resolution uint8 images
  interventions/regions/*     # Shared images, selected masks, intervention metadata
  predictions/{resnet,vit}/    # Logits, probabilities, IDs, input/protocol/environment hashes
  tables/
    color_per_image.csv, color_summary.csv
    mild_model_specific_flips.csv
    region_per_image.csv
    matched_region_pairs.csv, matched_region_summary.csv
  figures/                     # Matrices, paired comparisons, examples
  logs/                        # Execution evidence, when captured externally
```

## Run/resume

From the repository root, using the repository environment:

```bash
python -m unittest discover -s skript -p 'test_color_background.py' -v
python skript/color_background.py prepare
# Inspect pilot/masks.png and color_examples.png.
# Explicitly record reviewer/status and accepted flags in masks/review.json.
python skript/color_background.py run --device cuda:0 --batch-size 32
python skript/color_background.py report
```

`prepare` does not load models. `run` uses existing local `LocalPredictor`, one model at a time, batched deterministic evaluation/inference mode and shared uint8 inputs before model-specific preprocessing. Report generation does not load models. Prediction writes are atomic; all existing caches are validated before reuse. Frozen signatures include transformation code, inference code, checkpoint/config hashes, source manifest, baseline probabilities and complete mask-review records. A resumed model cannot silently mix software/device/batch environments. To change a frozen protocol, preserve this run and use a **new `--output` directory**; use the same masks/review policy explicitly for a new run.

All 48 unmodified images are inferred on the current device and gated against trusted `preds/` exports: decisions must match exactly and probabilities must pass `allclose(atol=5e-4, rtol=1e-4)`. Paired effects use these within-device originals, avoiding subtraction across devices. Two additional repeats on eight originals (one per class) audit numerical variability. This is 96 original evaluations plus 32 extra audit evaluations, beyond **2,144 intervention evaluations**. Original full-dataset predictions are not recomputed. The audit is not a cross-device bitwise determinism guarantee.

Scientific use: descriptive research/teaching only. Sensitivity is not correctness, calibration, label preservation, biological feature recognition, clinical validity or architecture superiority. Do not turn this small selected experiment into a general ranking of CNNs versus transformers.
