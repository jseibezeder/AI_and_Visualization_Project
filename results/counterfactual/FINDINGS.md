# Experiment 5 — counterfactual decision boundaries: findings

Completed on `HeuVik` from base `f07eae3`. Prior experiments/checkpoints/data unchanged. See `README.md` for protocol, `summary.json` for search results, and `validation/transition_endpoint_audit.json` for independent repeated checks of the resolved transitions.

## Search and selection

Eight images were fixed before new inference using evidence from Experiments 1–4, all within the existing 48-image explanation set. Exact criteria, source scores and provenance are in `subset/selection.csv` and the frozen protocol.

Search horizontal translation ±3 pixels and rotation ±30 degrees. Complete global scans at 0.125 pixels / 0.25 degrees, followed by refining every observed adjacent top-1 change for either model to 0.0125 pixels / 0.025 degrees. Search positive and negative directions independently, including changes between non-original classes and switches back to the original decision.

**Everything below is a nearest observed switch on this finite, adaptive grid**, not a continuous or globally minimum perturbation. Equal-endpoint global-scan intervals can still hide narrower class islands. A no-observed-switch result is range/grid censoring, not a proof of robustness.

## Nearest observed switches

Each entry is the **signed parameter with smallest observed absolute value over either direction**, chosen independently for each model. Negative translation moves content left; positive rotation is counterclockwise. Different signs may produce each model's minimum. A dash means no switch was observed on the tested grid in the specified range.

| Test image | ResNet rotation (degrees) | ViT rotation (degrees) | ResNet horizontal shift (pixels) | ViT horizontal shift (pixels) |
|---|---:|---:|---:|---:|
| #65 | -0.425 | +26.450 | -0.1000 | — |
| #105 | -5.250 | +9.100 | +0.5125 | — |
| #194 | +12.900 | — | -1.5000 | — |
| #432 | -1.075 | -8.125 | -0.1625 | -0.5250 |
| #809 | — | -1.600 | -0.8500 | -0.1000 |
| #906 | +1.125 | +3.150 | +0.1875 | +0.7750 |
| #2494 | +11.725 | — | -2.0000 | — |
| #2878 | +5.025 | — | +0.6250 | — |

The behavior is not uniformly in one model's favor. On **#809**, ViT switches under a smaller leftward translation (-0.1 px) than ResNet (-0.85 px). ViT also switches under rotation while no ResNet rotation switch is observed within ±30 degrees. However, ResNet's original prediction on #809 is already wrong; its switch at -0.85 px **corrects** the decision. This cannot be interpreted as a simple robustness ranking.

For **#2878**, both models are originally correct (monocyte, class 5). ResNet changes under rotation and subpixel translation, while no ViT switch is observed on either tested axis/range. This is informative about these specific models/images/transformations, not CNN-versus-transformer architectures generally.

## Strong non-monotonic example: #2878 rotation

True class: **monocyte (5)**. ResNet and ViT are originally correct.

| Positive rotation | ResNet class | ResNet p(original class 5) | ViT class | ViT p(original class 5) |
|---|---:|---:|---:|---:|
| 0 degrees | 5 | 0.99853 | 5 | 0.99864 |
| +5.025 degrees | 3 | 0.47674 | 5 | 0.99823 |
| +5.050 degrees | 5 | 0.50055 | 5 | 0.99824 |
| +5.075 degrees | 3 | 0.49643 | 5 | 0.99824 |

ResNet therefore shows **switch → return → switch** across consecutive refined test values, while ViT stays at class 5 in this illustration. Further out, ResNet returns again at +5.6 degrees and departs again at +6.75 degrees. A single monotonic-boundary assumption would conceal this structure.

The local zoom was selected **after** the completed search as an illustration; the eight-image search set itself was selected beforehand. Bilinear interpolation, gray fill, low source resolution and uint8 rounding are part of the intervention and can produce narrow reentrant decisions. Do not read these fine-scale transitions as biological changes or smooth geometric invariance estimates.

## Complete paths and returns

Across eight images × two models × two axes × two outward directions:

- **64 directional searches**; 24 have no observed switch within the tested range/grid.
- **137 observed class transitions**, of which **57 return to that model's original decision**.
- Rotation: 48 ResNet transitions, 39 ViT transitions; 20 and 17 baseline returns respectively.
- Translation: 30 ResNet transitions, 20 ViT transitions; 11 and 9 baseline returns respectively.

These are transitions in intentionally selected qualitative paths, not independent statistical observations or full-test error rates. Closely spaced flips are separately recorded, not collapsed into one boundary.

## Validation

- **Six focused tests pass**, including exact byte equivalence to existing integer-shift/rotation rendering, subpixel/sign/border semantics, a switch-and-return island hidden between same-label coarse endpoints, signed non-monotonic rays, source-selection quotas, full synthetic reporting, and cache/refinement/protocol rejection.
- All eight original decisions match trusted exports for both models. Maximum probability error: **9.02e-7 ResNet**, **1.10e-6 ViT**.
- **72 common-level comparisons to Experiment 2** match, with maximum probability discrepancy **3.55e-6**.
- Search baseline repeat audits record zero probability variation and no determinism warnings.
- **255 unique image/axis/tick transition endpoints** (union from both models) were independently re-evaluated **three times for each model**: **1,530 additional evaluations**, **zero argmax mismatches**.
- Endpoint maximum probability discrepancy from cached search outputs: **5.07e-7 ResNet**, **1.17e-9 ViT**; repeat-to-repeat range zero in both audits.
- Smallest top-two probability margin among audited endpoints: **1.43e-4 ResNet**, **2.04e-4 ViT**, larger than the measured numerical discrepancies. The fine-scale observed switches survived repeated inference; they are not explained by the measured repeat noise.

The endpoint audit covers observed transition endpoints, not all untested points or arbitrary devices.

## Computation and reproducibility

The main search and reports completed in approximately **59.5 seconds** on the Quadro RTX 4000, using batches of 32 and one model at a time. Per model:

- Coarse stage: 160 image/axis/tick evaluations.
- Complete global-scan additions: 2,160.
- Refinement additions: 927.
- Total saved path entries: **3,247**.
- Raw-byte memoization results in **3,231 new intervention inference inputs**, plus eight original inputs and sixteen extra repeat-audit evaluations. Original pixels are exactly reused across axes.

The additional endpoint repeat pass is separate from those search counts. All cached arrays/signatures are checked before reuse. A completed resume regenerates reports without loading models or performing search inference. Primary protocol hash: `0558573cfb3a1ec92cc86819f645d57971e6ed19e90e1e3ad5a978be946ddae6`.

## Deliverables

- `figures/nearest_switch_comparison.png`: observed-minimum comparison across all eight images.
- `figures/boundary_zoom_sample_02878.png`: local switch/return sequence with original/changed images and both models' probabilities.
- `figures/counterfactuals/*.png`: 16 original/nearest-counterfactual figures (one per image/axis), showing **both models on every candidate image** and full 8-class distributions.
- `figures/probability_paths/*.png`: 16 full-range path plots, all class probabilities and top-1 color strips.
- `tables/probability_paths.csv`: every tested parameter, prediction, probability vector and actual pixel/RGB change diagnostics.
- `tables/directional_boundaries.csv`: last tested same-class neighbor, first observed switched point, signed direction, bracket width, transition/return counts and censoring.
- `tables/all_observed_transitions.csv`: every observed departure/class change/return.
- `tables/nearest_observed_switches.csv`: signed and absolute nearest observed parameters for both models.
- `validation/transition_endpoint_audit.{json,csv}`: explicit endpoint repeatability and probability margins.

Research/teaching only. Family-specific, range/grid-specific and rasterization-specific results; no global mathematical minimum, clinically preserved-label guarantee, clinical diagnosis claim or general model-superiority conclusion.
