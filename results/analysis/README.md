# ResNet–ViT prediction analysis

Generated from aligned local predictions by `skript/analysis.py`. No training or inference is performed here. All sample indices are zero-based and local to their split in `data/bloodmnist.npz`.

## Layout

```text
results/analysis/
├── README.md
├── summary.json                  # All requested splits
└── {train,val,test}/
    ├── summary.json              # Counts, accuracy, thresholds and provenance
    ├── tables/
    │   └── all_samples.csv       # Both predictions and all 16 probabilities
    ├── groups/
    │   ├── both_correct.csv
    │   ├── only_resnet_correct.csv
    │   ├── only_vit_correct.csv
    │   └── both_incorrect.csv
    ├── distribution_differences/
    │   ├── same_class_ranked.csv
    │   └── same_class_substantial.csv
    ├── matrices/
    │   ├── disagreement_counts.csv
    │   └── disagreement_proportions.csv
    └── figures/
        ├── disagreement_matrix.png
        ├── disagreement_matrix.pdf
        ├── correctness_groups.png
        └── examples_*.png         # Up to 12 original RGB images per nonempty group
```

The four correctness groups are mutually exclusive and exhaustive. `both_incorrect` includes both agreeing and disagreeing incorrect predictions.

## Distribution differences

For two probability vectors p and q, Jensen–Shannon divergence is
`JS(p,q) = 0.5 KL₂(p || m) + 0.5 KL₂(q || m)`, where `m = (p+q)/2`.
It ranges from 0 (identical distributions) to 1 bit. The default exploratory threshold is **JS >= 0.1 bits**, restricted to samples where both models predict the same class. This is a descriptive cutoff, not a significance test. All same-class samples are also saved in descending JS order; gallery examples show the largest divergences first.

## Matrix interpretation

Rows are ResNet predictions; columns are ViT predictions. Counts include every analysed image. Proportions divide each cell by the total number of analysed images, not by row totals. Diagonal entries mean prediction agreement, not correctness.

## Regenerate

From the repository root, using an environment with working NumPy and Matplotlib:

```bash
python skript/analysis.py
python skript/analysis.py --splits test --js-threshold 0.15 --output results/analysis_js015
python skript/analysis.py --no-plots
```

Rerunning overwrites generated files in the selected output directory. Use a new `--output` directory to preserve separate experiments. Empty groups still receive a CSV header but no gallery. Existing unrelated files are not removed.

Train-set results are descriptive, not estimates of generalisation. Neither this analysis nor its galleries are intended for clinical use.
