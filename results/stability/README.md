# Controlled prediction-stability experiment

Implementation: `skript/stability.py`. Existing `preds/` exports are the baseline; the previous comparison in `results/analysis/` is neither repeated nor modified. The 320-image subset did not previously exist locally, so its exact IDs are now persisted: **40 images per true class from the test split**, sampled without replacement with seed 42.

## Layout

```text
results/stability/
├── README.md
├── config.json                  # Transform levels, seed, split, RGB border fill
├── protocol.json                # Frozen configuration + baseline/data signature
├── summary.json                 # Created after inference/report completes
├── subset/
│   └── manifest.json            # Exact original split-local image IDs and hashes
├── pilot/
│   ├── *.png                    # One example per true class, every level
│   ├── review.json              # Signature and explicit approval status
│   └── visual_review.md         # Exploratory pilot limitations
├── transformed_images/
│   └── {transformation}/*.npz    # Shared raw uint8 RGB inputs for both models
├── predictions/
│   ├── resnet/
│   │   ├── *.npz                # Logits, probabilities, sample IDs, input hash
│   │   └── reproducibility.json # Evaluation mode + repeated-output audit
│   └── vit/                     # Same schema
├── tables/
│   ├── per_image_metrics.csv
│   ├── stability_metrics.csv
│   ├── behavioral_fingerprints.csv
│   └── selected_transition_examples.csv
├── figures/
│   ├── curves/*.png             # 2A: baseline probability, flip rate, retention
│   ├── behavioral_fingerprints.png  # 2B: both models + ViT minus ResNet
│   └── transition_grids/*.png    # 2C: original and transformed image sequences
└── logs/
    └── run.log
```

## Run / resume

From the repository root, in an environment with the repository requirements installed:

```bash
python skript/stability.py pilot
# Inspect pilot PNGs and adjust config.json if necessary, before inference.
python skript/stability.py run --approve-pilot
python skript/stability.py report
python -m unittest discover -s skript -p 'test_stability.py' -v
```

On this machine, `.../opt/anaconda3/envs/work/bin/python` has a working plotting/inference stack. The default base Python has incompatible NumPy/Matplotlib binaries.

`run` skips existing condition/model prediction files, caches transformed RGB images once, and writes completed condition outputs atomically. `report` does not load models. A signature guards against changing configuration, images, or baseline probabilities after inference starts. Use a new `--output results/stability_<name>` for a new protocol; do not silently combine different interventions. A fresh output directory will need its own pilot. Do not modify model weights between resumed runs.

## Protocol

- All transformations act on **original 28×28 RGB images before model preprocessing**. The transformed uint8 arrays are cached and reused by both models; their hashes must match in reports.
- ResNet uses the official small-image ResNet-18 checkpoint, 28×28 RGB, scaled to `[0,1]` and normalized with mean/std 0.5. ViT uses its local supplied processor, resizing to 224×224 and applying the same scale/normalization.
- Rotation: −20, −10, −5, +5, +10, +20 degrees, bilinear interpolation, fixed canvas. Horizontal shift: ±1, ±2, ±3 pixels, no wraparound. Both use RGB `(128,128,128)` fill.
- Brightness: ×0.8, ×0.9, ×1.1, ×1.2. Gaussian blur: Pillow GaussianBlur radius/σ = 0.3, 0.6, 1.0, 1.5 pixels.
- Gaussian noise: σ = 0.01, 0.025, 0.05, 0.1 on `[0,1]` RGB, clipped, quantized back to uint8. Each image × severity has three independent, reproducible realizations keyed by seed, fixed subset position, severity, and repeat. All three are shared across models.
- Channel scaling: each of R, G, B separately ×0.9 and ×1.1; other channels unchanged. Three channel-specific columns are used in the fingerprint heatmap.
- There are 38 transformed conditions, including the repeated noise conditions: 12,160 transformed images evaluated per model. Baseline probabilities are reused, not regenerated across the full dataset.

## Metrics and aggregation

For each model and image, the reference class is its own original argmax. Retained baseline probability is the transformed softmax probability of that fixed class. Flip fraction compares transformed argmax to that original argmax. **Stability is not accuracy.**

Correct-label retention is computed on the **same subset initially classified correctly by both models**; its denominator is recorded globally and per true class. Accuracy on all images is also saved in the metrics table. An empty correct-retention subset has an undefined value, not zero.

Noise repetitions are averaged **within each image first**, not counted as three independent images. The metrics table additionally records minimum/maximum flip fractions across the three realizations, and the per-image table preserves every realization.

2A shows retained probability, flip fraction, and correct-label retention, including the cached unmodified reference. Signed rotation/translation levels remain signed (sensitivity may be asymmetric).

2B groups by **true class**, computes change in baseline-class probability relative to the original, and averages configured levels equally (after averaging noise repeats). The ViT-minus-ResNet panel is the difference of these changes, not the difference of raw baseline confidences. Results depend on the chosen severity grid; this is descriptive, not a formal architecture-ranking test.

2C picks two images per transformation with the largest baseline-probability drop across either model and the levels. Noise grids show realization 0. Every image shows each model's predicted class, its probability, and probability of the original decision. The selection rule and exact IDs are recorded. These are deliberately informative examples, **not representative samples**.

## Reproducibility and limitations

Models use evaluation mode, inference mode, fixed seeds, deterministic algorithms with warnings recorded, and disabled cuDNN benchmarking. Three inference repeats on one baseline image per class measure output variability and compare with cached baseline predictions. This audit covers eight images, not every transformed input, and does not guarantee cross-device bitwise reproducibility. See each model's `reproducibility.json` for numerical differences and warnings.

The pilot was reviewed by the coding assistant, not a clinical expert; see `pilot/visual_review.md`. Strong noise/blur are exploratory stress tests. Interventions are **not guaranteed to preserve labels or diagnostic morphology**. Do not infer model superiority from stability alone, especially when both models are initially wrong. Research/teaching only; not clinical use.
