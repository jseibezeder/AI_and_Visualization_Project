#!/usr/bin/env python3
"""Compare saved ResNet/ViT predictions; no model inference is required.

From the repository root:
    python skript/analysis.py
    python skript/analysis.py --splits test --js-threshold 0.1

Jensen-Shannon divergence is measured in bits (range 0..1). The default
0.1-bit cutoff is an exploratory definition of 'substantially different',
not a statistical significance test. All same-class examples are also ranked
so the threshold can be reconsidered without losing information.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "both_correct", "only_resnet_correct", "only_vit_correct", "both_incorrect"
)
SHORT_NAMES = (
    "Basophil", "Eosinophil", "Erythroblast", "Immature granulocytes",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet",
)


def js_divergence(p, q):
    """Row-wise Jensen-Shannon divergence in bits, including zero probabilities."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    midpoint = (p + q) / 2

    def kl(a):
        terms = np.zeros_like(a)
        positive = a > 0
        terms[positive] = a[positive] * np.log2(a[positive] / midpoint[positive])
        return terms.sum(axis=1)

    return np.clip((kl(p) + kl(q)) / 2, 0, 1)


def load_predictions(path, class_names):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in (
            "sample_index", "labels", "predicted_labels", "probabilities", "class_names"
        )}
    n = len(result["labels"])
    if n == 0:
        raise ValueError(f"Empty prediction file: {path}")
    for key in ("sample_index", "labels", "predicted_labels"):
        array = result[key]
        if array.shape != (n,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{path}: {key} must be a one-dimensional integer array")
    if len(np.unique(result["sample_index"])) != n or (result["sample_index"] < 0).any():
        raise ValueError(f"Invalid or duplicate sample indices in {path}")
    if result["class_names"].tolist() != class_names:
        raise ValueError(f"Class mapping mismatch in {path}")
    p = result["probabilities"]
    if p.shape != (n, 8) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"Invalid probability matrix in {path}")
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
    for key in ("labels", "predicted_labels"):
        if ((result[key] < 0) | (result[key] >= 8)).any():
            raise ValueError(f"Invalid class IDs in {path}")
    np.testing.assert_array_equal(result["predicted_labels"], p.argmax(axis=1))
    return result


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def plot_results(directory, split, counts, masks, images, resnet, vit,
                 divergence, same_different, max_examples):
    # Import lazily: numeric exports still work with --no-plots on headless/minimal setups.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140}):
        fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
        n = counts.sum()
        for ax, values, title, percent in (
            (axes[0], counts, "Prediction pairs · counts", False),
            (axes[1], counts / n, "Prediction pairs · fraction of all images", True),
        ):
            im = ax.imshow(values, cmap="Blues", vmin=0)
            for i in range(8):
                for j in range(8):
                    text = f"{values[i, j]:.1%}" if percent else str(values[i, j])
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                            color="white" if values[i, j] > values.max() * 0.55 else "#172638")
            ax.set(xticks=np.arange(8), yticks=np.arange(8),
                   xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                   xlabel="ViT predicted class", ylabel="ResNet predicted class", title=title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.colorbar(im, ax=ax, shrink=.65)
        fig.suptitle(f"BloodMNIST · {split} · {n:,} images\nDiagonal = same predicted class, not necessarily correct", fontsize=15)
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"disagreement_matrix.{suffix}", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        sizes = [int(masks[g].sum()) for g in GROUPS]
        bars = ax.bar([g.replace("_", " ").title() for g in GROUPS], sizes,
                      color=["#238b75", "#397cb7", "#a16bb0", "#de785c"])
        for bar, size in zip(bars, sizes):
            ax.annotate(f"{size:,}\n({size/n:.1%})", (bar.get_x()+bar.get_width()/2, size),
                        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
        ax.set(ylabel="Number of images", title=f"Correctness groups · {split}")
        ax.set_ylim(0, max(sizes) * 1.22 + 1)
        fig.savefig(figures / "correctness_groups.png", bbox_inches="tight")
        plt.close(fig)

        # Galleries display original archive RGB images, not normalized model inputs.
        selections = {g: np.flatnonzero(masks[g])[:max_examples] for g in GROUPS}
        selections["same_class_different_distributions"] = np.flatnonzero(same_different)[
            np.argsort(-divergence[same_different], kind="stable")
        ][:max_examples]
        for name, positions in selections.items():
            if len(positions) == 0:
                continue
            columns = min(4, len(positions))
            rows = (len(positions) + columns - 1) // columns
            fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.4, rows * 3.5),
                                     squeeze=False, constrained_layout=True)
            for ax in axes.flat:
                ax.axis("off")
            for ax, position in zip(axes.flat, positions):
                index = int(resnet["sample_index"][position])
                r = int(resnet["predicted_labels"][position])
                v = int(vit["predicted_labels"][position])
                y = int(resnet["labels"][position])
                ax.imshow(images[index])
                ax.set_title(
                    f"#{index} · True: {SHORT_NAMES[y]}\n"
                    f"R: {SHORT_NAMES[r]} ({resnet['probabilities'][position, r]:.1%})\n"
                    f"V: {SHORT_NAMES[v]} ({vit['probabilities'][position, v]:.1%})\n"
                    f"JS: {divergence[position]:.3f} bits", fontsize=9)
            fig.suptitle(f"{split} · {name.replace('_', ' ')}", fontsize=14)
            fig.savefig(figures / f"examples_{name}.png", bbox_inches="tight")
            plt.close(fig)


def analyse_split(split, args, names):
    r = load_predictions(args.predictions / "resnet" / f"{split}.npz", names)
    v = load_predictions(args.predictions / "vit" / f"{split}.npz", names)
    # Never silently compare different subsets or sample orders.
    np.testing.assert_array_equal(r["sample_index"], v["sample_index"])
    np.testing.assert_array_equal(r["labels"], v["labels"])
    with np.load(args.dataset, allow_pickle=False) as archive:
        labels = archive[f"{split}_labels"].reshape(-1)
        images = archive[f"{split}_images"] if not args.no_plots else None
        indices = r["sample_index"]
        if indices.max() >= len(labels):
            raise ValueError(f"Sample index outside {split} dataset")
        np.testing.assert_array_equal(r["labels"], labels[indices])

    rp, vp, y = r["predicted_labels"], v["predicted_labels"], r["labels"]
    rc, vc = rp == y, vp == y
    masks = dict(zip(GROUPS, (rc & vc, rc & ~vc, ~rc & vc, ~rc & ~vc)))
    assert np.all(np.sum(list(masks.values()), axis=0) == 1)
    divergence = js_divergence(r["probabilities"], v["probabilities"])
    same = rp == vp
    substantial = same & (divergence >= args.js_threshold)
    directory = args.output / split
    fields = ["sample_index", "split", "true_label", "true_class", "resnet_prediction",
              "resnet_class", "vit_prediction", "vit_class", "resnet_confidence",
              "vit_confidence", "group", "same_prediction", "js_divergence_bits",
              *[f"resnet_prob_{i}" for i in range(8)], *[f"vit_prob_{i}" for i in range(8)]]
    group_labels = np.empty(len(y), dtype=object)
    for group, mask in masks.items():
        group_labels[mask] = group

    def rows(positions):
        for i in positions:
            yield [int(indices[i]), split, int(y[i]), names[y[i]], int(rp[i]), names[rp[i]],
                   int(vp[i]), names[vp[i]], float(r["probabilities"][i].max()),
                   float(v["probabilities"][i].max()), group_labels[i], bool(same[i]),
                   float(divergence[i]), *r["probabilities"][i].tolist(),
                   *v["probabilities"][i].tolist()]

    write_csv(directory / "tables/all_samples.csv", fields, rows(range(len(y))))
    for group, mask in masks.items():
        write_csv(directory / f"groups/{group}.csv", fields, rows(np.flatnonzero(mask)))
    ranked = np.flatnonzero(same)[np.argsort(-divergence[same], kind="stable")]
    write_csv(directory / "distribution_differences/same_class_ranked.csv", fields, rows(ranked))
    write_csv(directory / "distribution_differences/same_class_substantial.csv", fields,
              rows(ranked[divergence[ranked] >= args.js_threshold]))

    counts = np.zeros((8, 8), dtype=np.int64)
    np.add.at(counts, (rp, vp), 1)
    assert counts.sum() == len(y)
    matrix_fields = ["resnet_prediction / vit_prediction", *names]
    for name, matrix in (("counts", counts), ("proportions", counts / len(y))):
        write_csv(directory / f"matrices/disagreement_{name}.csv", matrix_fields,
                  ([names[i], *matrix[i].tolist()] for i in range(8)))
    summary = {
        "split": split, "n_images": len(y), "class_names": names,
        "resnet_accuracy": float(rc.mean()), "vit_accuracy": float(vc.mean()),
        "groups": {g: {"count": int(m.sum()), "proportion": float(m.mean())} for g, m in masks.items()},
        "same_prediction_count": int(same.sum()), "different_prediction_count": int((~same).sum()),
        "same_class_substantial_count": int(substantial.sum()),
        "js_threshold_bits": args.js_threshold,
        "js_definition": "0.5*KL_base2(p || (p+q)/2) + 0.5*KL_base2(q || (p+q)/2)",
        "matrix_orientation": "rows=ResNet prediction; columns=ViT prediction",
        "matrix_proportions_denominator": len(y),
        "prediction_directory": str(args.predictions.resolve()),
        "dataset": str(args.dataset.resolve()),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plots:
        plot_results(directory, split, counts, masks, images, r, v, divergence,
                     substantial, args.max_examples)
    print(f"{split}: {len(y):,} images; " + ", ".join(f"{g}={m.sum()}" for g, m in masks.items()))
    print(f"  Same-class substantial distribution differences: {substantial.sum()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--predictions", type=Path, default=ROOT / "preds")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/bloodmnist.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--js-threshold", type=float, default=0.1, help="Jensen-Shannon divergence cutoff in bits (default: 0.1)")
    parser.add_argument("--max-examples", type=int, default=12, help="Gallery size per group")
    parser.add_argument("--no-plots", action="store_true", help="Export tables only; do not import matplotlib")
    args = parser.parse_args()
    if not 0 < args.js_threshold <= 1 or args.max_examples < 1:
        parser.error("Require 0 < --js-threshold <= 1 and --max-examples >= 1")
    names_dict = json.loads((ROOT / "data/metadata.json").read_text())["label"]
    names = [names_dict[str(i)] for i in range(8)]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {split: analyse_split(split, args, names) for split in args.splits}
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
