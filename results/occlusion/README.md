# Experiment 3 — spatial sensitivity through blurred-patch occlusion

Code: `skript/occlusion.py`. Local inference helper: `skript/inference.py`.

This is a new experiment. Original predictions are reused from `preds/`; Experiment 2 (`results/stability/`) and the prediction comparison (`results/analysis/`) are not recomputed or modified.

## Structure

```text
results/occlusion/
├── README.md
├── config.json                       # Patch size, stride, blur sigma, subset seed
├── protocol.json                     # Configuration, checkpoint/data/baseline signatures
├── summary.json                      # Written when all queries and reports finish
├── subset/
│   ├── manifest.json                 # Exact original test IDs, labels and selection rule
│   └── images.csv                    # True class + both baseline predictions
├── pilot/
│   └── occlusion_examples.png        # One image/class; original and three patch positions
├── interventions/
│   └── sample_*.npz                  # 169 shared raw RGB occlusions per image
├── predictions/
│   ├── resnet/
│   │   ├── sample_*.npz              # Logits, probabilities, coordinates, shared input hash
│   │   └── reproducibility.json
│   └── vit/                          # Same schema
├── maps/
│   └── sample_*.npz                  # 13×13 importance maps, target IDs and coordinates
├── tables/
│   ├── occlusion_sensitivity.csv     # Every position, target and model
│   └── map_summary.csv               # Signed and absolute summary statistics per map
├── figures/
│   └── sensitivity_maps/
│       └── sample_*.png              # Original | ResNet map | ViT map
└── logs/
    └── run.log
```

## Protocol

- **48 test images, six per true class**, sampled with seed 43. Up to two disagreement images per class are intentionally included; the remainder are sampled without replacement from the same class. This enrichment enables preferred-class comparisons but makes the set illustrative rather than representative. Exact IDs are persisted.
- **4×4-pixel patches, stride 2**, with top-left coordinates `(0,0)` through `(24,24)` in row-major order: **13×13 = 169 positions per image per model**.
- Blur the original RGB image with Pillow GaussianBlur radius/σ **2.0 pixels**, then replace only the current 4×4 patch with the corresponding blurred pixels. The full-image blur provides surrounding context; it is not fed as an entirely blurred image. Outside the patch, original pixels remain unchanged.
- Both models use the **same cached modified uint8 RGB arrays**, applied before any model-specific resizing or normalization. ResNet stays at 28×28; ViT's local processor resizes to 224×224. Eight output class IDs match the dataset mapping.
- Model calls expose only input images and output logits/probabilities. No gradients, attention weights, activations, or internal attribution methods are used.
- There are **16,224 intervention evaluations** (`48 × 169 × 2`), batched for efficiency. A further 48 small audit evaluations (`8 × 3 × 2`) check repeatability and consistency with the reused baseline. Additional targets use already-saved probability vectors and cost no extra queries.

## What each map means

For a fixed target class `c`, importance is:

`I_m(r) = p_m(c | original) - p_m(c | occlusion at r)`.

Positive/red means that replacing the patch reduced the target's probability; negative/blue means that replacing it increased the probability.

Each figure always compares **the same target class across both model columns**:

1. A true-class row for every image.
2. On disagreement images, separate ResNet-preferred and ViT-preferred rows. Even if a preferred target equals the true class, its explicitly named row remains for clarity.
3. If models agree on an incorrect class, an additional shared-preferred row.

Do not compare own-prediction maps for different target classes as though they explain the same decision. Target IDs are fixed from original predictions, not recomputed after occlusion.

All figures use one **global symmetric color scale**, shared across both models, all targets, and all 48 images. A large response elsewhere can make weaker maps look pale; consult numeric maps/tables for exact values.

The heatmap is a **13×13 grid of patch positions**, not a 28×28 per-pixel attribution map. Axes show patch centers measured from image pixel boundaries: 2, 4, ..., 26. Neighboring windows overlap; scores are not independent pixel contributions and should not be summed as such.

## Run/resume

From the repository root, using an environment with working repository dependencies:

```bash
python skript/occlusion.py prepare
python skript/occlusion.py run
python skript/occlusion.py report
python -m unittest discover -s skript -p 'test_occlusion.py' -v
```

On this machine the working interpreter is `/Users/master/opt/anaconda3/envs/work/bin/python`.

`prepare` selects/reuses the manifest and creates the pilot. `run` skips existing per-model/per-image caches and writes completed inference outputs atomically. `report` reads cached probabilities only. Model/data/config signatures prevent silently mixing experiments; use a new `--output results/occlusion_<name>` to change blur settings. Avoid changing dependencies or hardware during a resumed run.

Evaluation mode, inference mode, fixed seeds, deterministic algorithms with warnings recorded, and a three-repeat eight-image audit are used. The audit is limited to those baseline images, not a guarantee of cross-device bitwise equality. Each model's reproducibility file records measured numerical variability, cached-baseline differences, device/version and warnings.

## Scientific limitations

Occlusion maps reveal sensitivity to **artificial interventions**, not direct recognition of a nucleus, cytoplasm, or other biological feature. Results depend on blur strength, patch size, stride, source resolution, and preprocessing. A constant/smooth region may barely change under blur, so low sensitivity need not imply that a region is unused. Baseline predictions can be wrong, and probabilities are not established as calibrated.

RISE was optional and is not implemented: it would be a distinct masking experiment with additional queries, not a necessary part of this protocol.

Research/teaching only; not clinical use.
