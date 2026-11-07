# Experiment 4 — findings and interpretation

Completed on `HeuVik` from base commit `f07eae3`, using the existing local checkpoints. No previous experiment, weight or dataset file was changed. See [README.md](README.md) for the complete protocol and [summary.json](summary.json) for machine-readable results.

## 4A — color dependence on the same 48 explanation images

All 48 Experiment 3 IDs and their order were reused. Fourteen conditions include the original; the other thirteen change grayscale/saturation, red/blue channels or brightness.

### Label changes from each model's own original decision

| Condition | ResNet flips / 48 | ViT flips / 48 |
|---|---:|---:|
| Grayscale — feature ablation | 43 | 22 |
| Saturation ×0.5 — strong ablation | 37 | 7 |
| Saturation ×0.8 | 16 | 4 |
| Saturation ×1.2 | 10 | 2 |
| Saturation ×1.5 — strong stress test | 25 | 6 |
| Red ×0.95 | 8 | 2 |
| Red ×1.05 | 15 | 4 |
| Blue ×0.95 | 8 | 3 |
| Blue ×1.05 | 2 | 3 |
| Brightness ×0.9 | 13 | 1 |
| Brightness ×0.95 | 11 | 0 |
| Brightness ×1.05 | 7 | 2 |
| Brightness ×1.1 | 11 | 2 |

Across the ten operationally mild conditions there are **106 image-condition cases where only one model flips**, spanning **31 distinct images**: 92 ResNet-only and 14 ViT-only cases. These are repeated interventions on the same 48 images, **not 106 independent observations**.

Counting all mild flips (including cases where both models flip), ResNet flips 101 times and ViT 23 times across 480 image-condition pairs. However, **stability is not accuracy**: ResNet's mild flips include 30 wrong-to-correct and 48 correct-to-wrong transitions; ViT's include 9 wrong-to-correct and 11 correct-to-wrong transitions. Remaining flips are wrong-to-different-wrong. On the deliberately enriched original set ResNet is correct on 33/48 and ViT on 42/48; these rates must not replace full-test accuracy.

### Clear small-channel example: image #194

True class: **neutrophil (6)**. Both models are originally correct.

After multiplying only the red channel by 0.95:

- **ResNet:** changes to **monocyte (5)**. Probability of the original/true class drops from **0.999329 to 0.000233**.
- **ViT:** stays at **neutrophil (6)**. Probability of that same class changes from **0.999734 to 0.999679**.

This example is also visible in `figures/mild_flip_examples.png`. It shows strong sensitivity to this particular RGB intervention, not a verified realistic stain change. No clinical expert established that the modified image preserves diagnostic information.

### Visualizations

- `figures/color_probability_matrix.png`: all 48 rows, 14 columns, one panel per model; color shows change in p(original decision), numbers mark new class IDs on flips.
- `figures/color_flip_difference.png`: ResNet-only versus ViT-only flips; neutral means both or neither flip.
- `figures/mild_flip_examples.png`: eight ranked image-condition examples, not a representative sample. Repeated appearances of an image are separate conditions, not independent examples.
- `tables/color_per_image.csv`: predictions, full probability vectors, true-class probability changes and actual intervention budgets.
- `tables/mild_model_specific_flips.csv`: all 106 cases, not just the gallery.

## 4B — approximate central versus peripheral regions

Sixteen original images (two per class) were selected and their individual ellipse masks visually reviewed before Experiment 4 inference. Review is by the coding assistant, **not a human annotator or clinical expert**. The regions are approximate; surrounding regions can include other cells. See per-image limitations in `masks/review.json` and inspect `pilot/masks.png` before using these results in a presentation.

### Whole-region comparisons

Original, central-grayscale and peripheral-grayscale versions are shown for every reviewed image, together with the model predictions. Region-specific blur was also evaluated. These full-region interventions alter different areas and intensities, so their raw effects cannot establish cell-versus-background preference on their own.

### Exact matched controls

Every reviewed image has sufficient eligible pixels to use **exactly 64 changed pixels in each region**. Paired additive RGB modifications use -16, -8, +8 and +16 intensity units per channel, without clipping. Every central/peripheral pair has identical changed pixel count, signed channel shift, total L1 and whole-image RMS intensity change. Three deterministic spatial selections are averaged within image/sign.

For a shared target across both models, the following descriptive means use the **true-class probability**. Values are mean absolute probability changes in **percentage points**, with equal weighting across images, positive/negative signs and spatial repeats:

| RGB change magnitude | ResNet central | ResNet peripheral | ViT central | ViT peripheral |
|---|---:|---:|---:|---:|
| ±8 / 255 per channel | 4.42 pp | 2.12 pp | 2.47 pp | 1.34 pp |
| ±16 / 255 per channel | 17.14 pp | 4.39 pp | 7.94 pp | 3.51 pp |

These shared-target figures can be reproduced from `region_per_image.csv`: filter `design == matched`, take the absolute value of `true_class_probability_change`, then average by model, region and absolute delta. Each image/sign has exactly three rows, so this is equivalent to averaging repeats first.

Both models have larger average central-region responses under these controls, but there are individual exceptions and signed/intensity-specific effects. This is **not evidence that one model uses only the cell and the other uses only the background**, and is not a direct measure of attention or biological feature recognition. The primary paired plots/tables use each model's own baseline class; that target can differ on disagreement images, which is why shared true-class values are also exported and reported here.

### Visualizations

- `figures/matched_region_comparison.png`: per-image/sign central/peripheral absolute response, averaging three repeats; axes may have different numerical ranges across model panels, so read the scales.
- `figures/whole_region_ablations.png`: explicitly unequal-budget whole-region effects.
- `figures/region_examples_01.png` through `_04.png`: all 16 original images, both regional grayscale ablations and a matched +16 pair, with predictions and pixel/intensity budgets.

## Validation and computational evidence

- Five focused tests passed: identity/grayscale and input immutability; exact equivalence to existing brightness/channel transformations; deterministic matched interventions and exact area/intensity budgets; insufficient-eligible-pixel rejection; complete synthetic reporting plus cache/environment/protocol rejection.
- All 48 current-device original decisions agree exactly with the trusted `preds/` exports for **both models**.
- Maximum absolute baseline probability discrepancy: **3.76e-6 ResNet**, **1.67e-6 ViT**.
- Maximum probability range in the eight-image repeat audit: **8.34e-7 ResNet**, **1.19e-7 ViT**; no determinism warnings recorded.
- **2,144 intervention evaluations + 96 originals + 32 extra repeat-audit evaluations**. Batched local GPU inference, followed by reporting, completed in approximately **41 seconds** on the Quadro RTX 4000.
- Resumed execution validated every cached input/prediction and rebuilt reports **without running inference again**.
- Every regional intervention left pixels outside its selected mask unchanged. All matched controls used exactly 64 nonzero changed pixels per region.

## What is justified

On this fixed, selected explanation set, the supplied ResNet is generally more sensitive than the supplied ViT to these color manipulations; several small-channel changes produce striking one-model-only class changes. Under exact matched additive controls, both models respond more strongly on average to the approximate central regions than to peripheral regions.

What is **not** justified: general CNN-versus-transformer superiority, clinically realistic stain robustness, label preservation under ablation, validated cell segmentation, proof of biological structure use, statistical significance from repeated modifications, or generalization beyond this small disagreement-enriched set. Research/teaching only.
