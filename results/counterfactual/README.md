# Experiment 5 — counterfactual decision boundaries

Code: `skript/counterfactual.py`. Tests: `skript/test_counterfactual.py`. Additional endpoint validation/local zoom: `skript/audit_counterfactual.py`.
No training/downloads or changes to previous experiments, models or data.

## Question and scope

For selected informative images, find the **nearest observed top-1 switch** along a specified signed transformation family. Compare both models on each candidate image and show the full probability paths, including observed returns to an earlier/original decision.

This is a **finite-grid qualitative search**, not a globally minimal adversarial perturbation, continuous robustness certificate, or architecture benchmark. Minimum rotation degrees and minimum horizontal translation pixels are different notions; neither is a minimum over all possible image modifications.

## Selection fixed before new inference

Eight unique images remain within the original 48-image explanation set. Source scores and exact reasons are persisted in `subset/selection.csv` and `subset/manifest.json`, together with source-file hashes in `protocol.json`:

- **Experiment 1:** largest JS divergence among differing-class predictions; largest among same-class predictions.
- **Experiment 2:** largest inter-model gap in retained-original-decision probability under rotation; same under translation, restricted to images present in both the stability set and explanation set.
- **Experiment 3:** two largest inter-model differences in peak absolute **true-class** occlusion response.
- **Experiment 4:** two largest mild red/blue one-model-only response differences on initially both-correct images.

Quotas are processed in that order; duplicate IDs are skipped, scores sorted descending, ties by ID. No Experiment 5 outcomes are used for selection. The resulting set is **#65, #105, #194, #432, #809, #906, #2494, #2878**. It is deliberately informative, not representative. Several original decisions are wrong; a switch can improve rather than worsen accuracy.

## Transformation definitions

Both operate on the original **28×28 uint8 RGB image**, before model-specific processing. Shared modified arrays are fed to both models unchanged. Gray RGB `(128,128,128)` border fill, bilinear interpolation, fixed 28×28 canvas, no wraparound or expansion.

- **Rotation:** -30 to +30 degrees; positive values rotate counterclockwise (Pillow `Image.rotate`).
- **Horizontal translation:** -3 to +3 pixels; positive values move content right. Fractional shifts use Pillow inverse affine mapping `(1,0,-dx,0,1,0)`.
- Zero is an explicit byte-identical copy. Integer shifts and configured rotations are tested for exact RGB equivalence to Experiment 2.

Rotation/translation can introduce interpolation and border-fill effects and remove edge content. A very small numerical angle/shift is not necessarily a proportional or biologically meaningful image difference. The numeric exports include actually changed pixel counts, RGB L1 and RMS changes. These are diagnostics, not the optimization objective.

## Coarse-to-fine search without monotonicity assumptions

1. **Coarse grid:** rotations every 5 degrees; horizontal translations every 1 pixel. Include both endpoints, both signs and zero.
2. **Complete global guard scan:** evaluate all rotation values every **0.25 degrees** and translations every **0.125 pixels**, not just cells with differing coarse endpoint classes. This catches reentrant/temporary switches inside many same-endpoint coarse intervals.
3. **Transition refinement:** find every adjacent guard-grid pair whose **top-1 class differs for either model**. Evaluate every intermediate tick in the union of these brackets for both models, to **0.025 degrees** or **0.0125 pixels**. Refine transitions between any two classes, not only the first departure from baseline.
4. Inspect positive and negative rays separately, starting at zero and proceeding outward. Save first observed switches, their last tested same-class neighbors, all observed class transitions, and every observed return to the original class. Also report each model's smallest observed absolute switch over either sign; ties choose the negative sign deterministically.

**Resolution caveat:** equal-endpoint *guard* intervals are not exhaustively evaluated at the final refinement spacing. Narrower class islands can still be missed there. A refined 0.025-degree bracket is not a proof that no earlier unsampled rotation could flip. Output names intentionally say **nearest observed**. No observed switch within the tested grid is censoring, not a robustness certificate. Do not call the observed first bracket a proven unique boundary or assume predictions remain switched beyond it.

## Computation, equivalence and caching

- Local `LocalPredictor` loads each model in evaluation/inference mode with fixed seeds and deterministic algorithms (warnings recorded). Batches default to 32 on the available GPU.
- Initial predictions on the eight originals must match trusted `preds/` argmax values exactly and pass probability equivalence (`atol=5e-4`, `rtol=1e-4`). Two extra repeats on all eight originals audit variability. The exact same baseline outputs are reused across axes.
- At overlapping Experiment 2 image/angle/shift/model combinations, coarse predictions must also match the saved geometric predictions before continuing to finer scans. Those reference NPZs are included in protocol hashes.
- A raw-RGB-byte memo avoids model calls for identical rasterized inputs, across both axes and all stages. It reuses logits/probabilities without changing the tick grid. The full per-tick paths remain exported.
- Shared input caches, prediction caches, ticks, sample IDs, logits/softmax consistency and model-specific environment signatures are validated before reuse. Existing invalid caches raise rather than being silently accepted. Writes are atomic.
- The protocol freezes selection sources, reference predictions, transform/inference/helper code, checkpoints, preprocessors, masks-independent source data, and configuration. Changing frozen inputs requires a new output directory. Resuming a model cannot mix device/software/batch environments.
- `report` loads cached probabilities, not models. All curves and examples can be regenerated without inference.

## Outputs

```text
results/counterfactual/
  README.md, FINDINGS.md, PROGRESS.md
  config.json, protocol.json, summary.json
  subset/{manifest.json,selection.csv}
  pilot/selected_images.png
  interventions/{rotation,translation}/sample_*/{coarse,scan,refine}.npz
  predictions/{resnet,vit}/
    reproducibility.json
    {rotation,translation}/sample_*/{coarse,scan,refine}.npz
  validation/
    prior_geometry_equivalence.csv
    stage_counts.json
    transition_endpoint_audit.{json,csv} # Three-repeat validation at every observed transition endpoint
  tables/
    probability_paths.csv              # All tested ticks, both models, 8 probabilities, pixel/RGB diagnostics
    directional_boundaries.csv         # Each image × model × axis × direction, including no-observed-switch cases
    nearest_observed_switches.csv      # Min observed absolute value over either sign
    all_observed_transitions.csv       # All observed top-1 changes and returns, outward from zero
  figures/
    nearest_switch_comparison.png
    boundary_zoom_sample_02878.png       # Post-search illustrative local switch/return sequence
    probability_paths/*.png            # 16 figures, both models and every class
    counterfactuals/*.png               # 16 figures, original and each model's nearest observed flip
  logs/                                # Captured execution evidence
```

Counterfactual figures show the original and each model's nearest observed flip candidate, with **both models' full 8-class probability distributions at every displayed image**. Thus the ResNet-minimum image is also evaluated by ViT, and vice versa. If no switch is observed, that model's panel explicitly says so rather than inventing a candidate.

Probability-path figures show all eight classes, emphasizing the original decision, plus a colored top-1 class strip. Lines connect tested points only; they do not certify intermediate probabilities. The two models' nearest candidates may use different signs. Read signed directional tables for like-direction comparisons.

## Run/resume

From the repository root with the repository environment:

```bash
python -m unittest discover -s skript -p 'test_counterfactual.py' -v
python skript/counterfactual.py prepare
python skript/counterfactual.py run --device cuda:0 --batch-size 32
python skript/counterfactual.py report
# Optional explicit endpoint repeat audit and illustrative local zoom:
python skript/audit_counterfactual.py --device cuda:0 --batch-size 32 --zoom-index 2878
```

`prepare` persists prior-evidence selection and a contact sheet without loading models. To change the ranges/grid, use a new output directory and configure its `config.json` before inference. Keep nested integer ticks including zero/endpoints. The smallest unit must evenly divide every desired coarse/reference level if common-level checks are expected.

## Endpoint repeat audit and local zoom

The completed validation pass reruns the **union of all observed transition endpoints from both models**: 255 image/axis/tick inputs, three repeats for each model (1,530 additional evaluations). Every argmax must match its saved search output, and probabilities must pass tight equivalence. The audit records top-two probability margins and its own source-code hash under the unchanged primary protocol. It does not validate unsampled values or guarantee cross-device identity.

The local zoom is a **post-search illustration**, not a change to image selection or a separately selected benchmark: image #2878, ResNet's first positive rotation switch, its first return, and next departure. The plot retains all eight class probabilities and explicitly labels tested points and rasterization limitations.

Research/teaching only. No clinical claims, certified label-preservation, formal significance test, model calibration claim or general CNN-versus-transformer superiority follows from this small selected experiment.
