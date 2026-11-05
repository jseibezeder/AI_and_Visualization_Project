#!/usr/bin/env python3
"""Controlled image interventions, cached inference, and visualizations 2A–2C.

python skript/stability.py pilot
# Review results/stability/pilot/*.png before explicitly approving the levels:
python skript/stability.py run --approve-pilot
python skript/stability.py report   # no model inference; regenerate plots/tables

Original predictions are reused from preds/. No training or prior analysis is repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from analysis import ROOT, SHORT_NAMES, load_predictions, write_csv

FAMILIES = ("resnet", "vit")
SCHEMA = 1
DEFAULT_LEVELS = {
    "rotation": [-20, -10, -5, 5, 10, 20],
    "translation": [-3, -2, -1, 1, 2, 3],
    "brightness": [0.8, 0.9, 1.1, 1.2],
    "blur": [0.3, 0.6, 1.0, 1.5],
    "noise": [0.01, 0.025, 0.05, 0.1],
    "red_scale": [0.9, 1.1],
    "green_scale": [0.9, 1.1],
    "blue_scale": [0.9, 1.1],
}
TRANSFORM_NAMES = {
    "rotation": "Rotation (degrees)", "translation": "Horizontal shift (pixels)",
    "brightness": "Brightness factor", "blur": "Gaussian blur σ (pixels)",
    "noise": "Gaussian noise σ ([0,1])", "red_scale": "Red channel factor",
    "green_scale": "Green channel factor", "blue_scale": "Blue channel factor",
}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def setup(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if not config_path.exists():
        save_json(config_path, {
            "schema_version": SCHEMA, "split": "test", "per_class": 40, "seed": 42,
            "noise_realizations": 3, "levels": DEFAULT_LEVELS,
            "border_fill_rgb": [128, 128, 128],
            "note": "Edit levels after visual pilot, before running inference. New config requires a new output directory.",
        })
    config = json.loads(config_path.read_text())
    if config["schema_version"] != SCHEMA or config["per_class"] != 40 or config["noise_realizations"] != 3:
        raise ValueError("This protocol requires 40 images/class and 3 noise realizations.")
    levels = config["levels"]
    if set(levels) != set(DEFAULT_LEVELS) or any(not values for values in levels.values()):
        raise ValueError("Configure nonempty levels for each supported transformation.")
    for kind, values in levels.items():
        if len(set(values)) != len(values) or not np.isfinite(values).all():
            raise ValueError(f"Duplicate/non-finite levels: {kind}")
        if kind in ("brightness", "blur", "noise", "red_scale", "green_scale", "blue_scale") and min(values) <= 0:
            raise ValueError(f"Levels must be positive: {kind}")
        if kind == "translation" and any(int(v) != v for v in values):
            raise ValueError("Translation levels must be integral pixels")
    names_dict = json.loads((ROOT / "data/metadata.json").read_text())["label"]
    names = [names_dict[str(i)] for i in range(8)]
    split = config["split"]
    with np.load(ROOT / "data/bloodmnist.npz", allow_pickle=False) as archive:
        all_images = archive[f"{split}_images"]
        all_labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
    if all_images.dtype != np.uint8 or all_images.shape[1:] != (28, 28, 3):
        raise ValueError("Expected original 28×28 uint8 RGB images")
    # Persist exact IDs, not just a seed. Reuse the manifest on subsequent runs.
    subset_path = out / "subset/manifest.json"
    if subset_path.exists():
        subset = json.loads(subset_path.read_text())
        if (subset["split"], subset["seed"]) != (split, config["seed"]):
            raise ValueError("Subset configuration changed; use a new output directory")
        indices = np.asarray(subset["sample_indices"], dtype=np.int64)
    else:
        rng = np.random.default_rng(config["seed"])
        indices = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(all_labels == c), size=40, replace=False) for c in range(8)
        ]))
        subset = {"split": split, "seed": config["seed"], "sample_indices": indices.tolist(),
                  "class_names": names, "selection": "40 uniformly sampled without replacement per true class"}
    images, labels = all_images[indices], all_labels[indices]
    if len(indices) != 320 or len(np.unique(indices)) != 320 or not np.all(np.bincount(labels, minlength=8) == 40):
        raise ValueError("Subset is not 320 distinct images balanced across 8 classes")
    if "images_sha256" in subset and subset["images_sha256"] != digest(images):
        raise ValueError("Subset image contents changed")
    subset.update(images_sha256=digest(images), labels_sha256=digest(labels))
    save_json(subset_path, subset)
    # Reuse saved baseline probabilities, verifying IDs, labels, and class order.
    baseline = {}
    for family in FAMILIES:
        original = load_predictions(ROOT / f"preds/{family}/{split}.npz", names)
        lookup = {int(index): row for row, index in enumerate(original["sample_index"])}
        rows = [lookup[int(index)] for index in indices]
        np.testing.assert_array_equal(original["labels"][rows], labels)
        baseline[family] = original["probabilities"][rows]
    signature = hashlib.sha256(json.dumps({
        "config": config, "images": digest(images), "labels": digest(labels),
        "baseline": {f: digest(p) for f, p in baseline.items()},
    }, sort_keys=True).encode()).hexdigest()
    lock = out / "protocol.json"
    if lock.exists() and json.loads(lock.read_text())["signature"] != signature:
        raise ValueError("Protocol/data/baseline changed after inference. Use a new --output directory.")
    return config, signature, images, labels, indices, baseline


def conditions(config):
    for kind, levels in config["levels"].items():
        for level in levels:
            for realization in range(config["noise_realizations"] if kind == "noise" else 1):
                key = f"{kind}_{level:g}_r{realization}"
                yield {"kind": kind, "level": level, "realization": realization, "key": key}


def transform(images, condition, seed, fill):
    """Transform original RGB, once for BOTH models, before resizing/normalization."""
    kind, level = condition["kind"], condition["level"]
    result = []
    for index, image in enumerate(images):
        pil = Image.fromarray(image, mode="RGB")
        if kind == "rotation":
            modified = np.asarray(pil.rotate(level, resample=Image.Resampling.BILINEAR,
                                             expand=False, fillcolor=tuple(fill)))
        elif kind == "translation":
            # Positive shift moves content to the right. No wrapping.
            modified = np.full_like(image, fill)
            shift = int(level)
            if shift == 0:
                modified[:] = image
            elif 0 < shift < image.shape[1]:
                modified[:, shift:] = image[:, :-shift]
            elif -image.shape[1] < shift < 0:
                modified[:, :shift] = image[:, -shift:]
        elif kind == "brightness":
            modified = np.asarray(ImageEnhance.Brightness(pil).enhance(level))
        elif kind == "blur":
            modified = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=level)))
        elif kind == "noise":
            # Independent reproducible realization per image × severity × repeat.
            key = f"{seed}:{index}:{level}:{condition['realization']}"
            noise_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            rng = np.random.default_rng(noise_seed)
            modified = np.rint(np.clip(image.astype(np.float64) / 255 +
                                      rng.normal(0, level, image.shape), 0, 1) * 255).astype(np.uint8)
        elif kind in ("red_scale", "green_scale", "blue_scale"):
            channel = {"red_scale": 0, "green_scale": 1, "blue_scale": 2}[kind]
            modified = image.astype(np.float64).copy()
            modified[..., channel] *= level
            modified = np.rint(np.clip(modified, 0, 255)).astype(np.uint8)
        else:
            raise ValueError(kind)
        result.append(modified)
    return np.stack(result)


def cached_images(out, config, signature, images, condition):
    path = out / "transformed_images" / condition["kind"] / f"{condition['key']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["protocol_signature"]) != signature:
                raise ValueError(f"Stale transformed images: {path}")
            return z["images"]
    transformed = transform(images, condition, config["seed"], config["border_fill_rgb"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=transformed, protocol_signature=signature)
    return transformed


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pilot(out, config, signature, images, labels):
    plt = plotting()
    for kind, levels in config["levels"].items():
        # One representative of every true class; all levels; repeat 0 for noise.
        positions = [int(np.flatnonzero(labels == c)[0]) for c in range(8)]
        fig, axes = plt.subplots(8, len(levels) + 1,
                                 figsize=(2 * (len(levels) + 1), 15), squeeze=False,
                                 constrained_layout=True)
        for row, pos in enumerate(positions):
            axes[row, 0].imshow(images[pos]); axes[row, 0].set_title("Original", fontsize=9)
            axes[row, 0].set_ylabel(SHORT_NAMES[row], fontsize=8)
        for col, level in enumerate(levels, start=1):
            condition = {"kind": kind, "level": level, "realization": 0}
            modified = transform(images, condition, config["seed"], config["border_fill_rgb"])
            for row, pos in enumerate(positions):
                axes[row, col].imshow(modified[pos]); axes[row, col].set_title(f"{level:g}", fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Morphology review · {TRANSFORM_NAMES[kind]}\nGray fill for rotation/translation; noise uses realization 0", fontsize=13)
        path = out / "pilot" / f"{kind}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120); plt.close(fig)
    save_json(out / "pilot/review.json", {
        "protocol_signature": signature, "status": "awaiting_review",
        "instructions": "Review PNGs for recognizable cell morphology. To change levels, edit config.json and rerun pilot before inference. Explicit --approve-pilot acknowledges this review; the code cannot establish label preservation.",
    })
    print(f"Pilot saved to {out / 'pilot'}. Review before running --approve-pilot.", flush=True)


def run(args, config, signature, images, labels, indices, baseline):
    review_path = args.output / "pilot/review.json"
    if not args.approve_pilot or not review_path.exists():
        raise ValueError("Run pilot, review images, then pass --approve-pilot.")
    review = json.loads(review_path.read_text())
    if review["protocol_signature"] != signature:
        raise ValueError("Pilot is stale; rerun pilot for the current configuration")
    review["status"] = "approved_via_cli"
    save_json(review_path, review)
    save_json(args.output / "protocol.json", {"signature": signature, "config": config,
                                              "pilot_approved_by_cli": True})
    # Import ML libraries only for inference, so pilot/report remain lightweight.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models.resnet18.models import ResNet18
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    warning_messages = []

    @torch.inference_mode()
    def predict(model, processor, batch_images):
        logits_list = []
        for start in range(0, len(batch_images), args.batch_size):
            batch = batch_images[start:start + args.batch_size]
            if processor is None:
                x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float()
                logits = model(((x / 255 - .5) / .5).to(device))
            else:
                inputs = processor(images=[Image.fromarray(x) for x in batch], return_tensors="pt")
                if tuple(inputs["pixel_values"].shape[1:]) != (3, 224, 224):
                    raise ValueError("Unexpected ViT input dimensions")
                logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            logits_list.append(logits.float().cpu())
        logits = torch.cat(logits_list)
        return logits.numpy(), logits.softmax(-1).numpy()

    names = json.loads((args.output / "subset/manifest.json").read_text())["class_names"]
    all_conditions = list(conditions(config))
    for family in FAMILIES:
        directory = args.output / "predictions" / family
        directory.mkdir(parents=True, exist_ok=True)
        missing = [c for c in all_conditions if not (directory / f"{c['key']}.npz").exists()]
        if not missing:
            print(f"{family}: all {len(all_conditions)} cached conditions found; skipping inference")
            continue
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if family == "resnet":
                model = ResNet18(in_channels=3, num_classes=8)
                checkpoint = torch.load(ROOT / "models/resnet18/resnet18_28_1.pth", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["net"], strict=True)
                processor = None
            else:
                processor = ViTImageProcessor.from_pretrained(ROOT / "models/vit", local_files_only=True)
                model = ViTForImageClassification.from_pretrained(ROOT / "models/vit", local_files_only=True)
                if [model.config.id2label[i] for i in range(8)] != names:
                    raise ValueError("ViT class mapping mismatch")
            model = model.to(device).eval()
            # Three baseline repeats on one example/class audit numerical variability.
            # Existing full baseline predictions are NOT recomputed.
            audit_positions = np.array([np.flatnonzero(labels == c)[0] for c in range(8)])
            repeats = [predict(model, processor, images[audit_positions]) for _ in range(3)]
            probabilities = np.stack([r[1] for r in repeats])
            if not np.all(probabilities.argmax(-1) == baseline[family][audit_positions].argmax(-1)):
                raise ValueError("Baseline audit disagrees with saved predictions; investigate model/preprocessing")
            audit = {
                "device": str(device), "torch_version": str(torch.__version__),
                "evaluation_mode": not model.training, "deterministic_algorithms": True,
                "deterministic_warn_only": True, "seed": config["seed"],
                "audit_sample_indices": indices[audit_positions].tolist(), "repeats": 3,
                "max_probability_range_across_repeats": float(np.ptp(probabilities, axis=0).max()),
                "prediction_changes_across_repeats": int(np.any(probabilities.argmax(-1) != probabilities[0].argmax(-1), axis=0).sum()),
                "max_abs_difference_from_cached_baseline": float(np.abs(probabilities[0] - baseline[family][audit_positions]).max()),
                "scope": "Audit covers eight baseline images, not every transformed image; no cross-device bitwise guarantee.",
            }
            for number, condition in enumerate(missing, 1):
                transformed = cached_images(args.output, config, signature, images, condition)
                logits, probabilities = predict(model, processor, transformed)
                np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-5)
                path = directory / f"{condition['key']}.npz"
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, probabilities=probabilities, logits=logits,
                                    sample_index=indices, protocol_signature=signature,
                                    transformed_sha256=digest(transformed))
                temporary.replace(path)
                print(f"{family} {number}/{len(missing)}: {condition['key']}", flush=True)
            warning_messages = sorted({str(w.message) for w in caught})
        audit["warnings"] = warning_messages
        audit["elapsed_seconds"] = time.perf_counter() - started
        save_json(directory / "reproducibility.json", audit)
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    report(args.output, config, signature, images, labels, indices, baseline)


def report(out, config, signature, images, labels, indices, baseline):
    plt = plotting()
    all_conditions = list(conditions(config))
    base_classes = {f: baseline[f].argmax(1) for f in FAMILIES}
    both_correct = np.logical_and.reduce([base_classes[f] == labels for f in FAMILIES])
    original_positions = np.arange(len(labels))
    records = []
    sample_records = []
    loaded = {}
    for condition in all_conditions:
        per_family = {}
        hashes = []
        for family in FAMILIES:
            path = out / "predictions" / family / f"{condition['key']}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing cached predictions: {path}; resume run first")
            with np.load(path, allow_pickle=False) as z:
                if str(z["protocol_signature"]) != signature:
                    raise ValueError(f"Stale predictions: {path}")
                np.testing.assert_array_equal(z["sample_index"], indices)
                probabilities = z["probabilities"]
                hashes.append(str(z["transformed_sha256"]))
            if probabilities.shape != (320, 8) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid probabilities: {path}")
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-5)
            per_family[family] = probabilities
        if hashes[0] != hashes[1]:
            raise ValueError("Models were evaluated on different transformed images")
        loaded[condition["key"]] = per_family

    # Average stochastic repeats per image FIRST; they are not independent images.
    for kind, levels in config["levels"].items():
        for level in levels:
            chosen = [c for c in all_conditions if c["kind"] == kind and c["level"] == level]
            for family in FAMILIES:
                ps = np.stack([loaded[c["key"]][family] for c in chosen])
                predictions = ps.argmax(2)
                stability = ps[:, original_positions, base_classes[family]]
                flips = predictions != base_classes[family][None, :]
                correct = predictions == labels[None, :]
                delta = stability - baseline[family][original_positions, base_classes[family]][None, :]
                for c, prob, st, flip, corr in zip(chosen, ps, stability, flips, correct):
                    for i in original_positions:
                        sample_records.append([
                            int(indices[i]), int(labels[i]), bool(both_correct[i]), family,
                            kind, level, c["realization"], int(base_classes[family][i]),
                            int(prob[i].argmax()), float(st[i]), bool(flip[i]), bool(corr[i]),
                        ])
                for class_id in [-1, *range(8)]:
                    mask = np.ones(320, dtype=bool) if class_id == -1 else labels == class_id
                    retained_mask = mask & both_correct
                    records.append({
                        "model": family, "transformation": kind, "level": level,
                        "class_id": class_id, "n_images": int(mask.sum()), "n_realizations": len(chosen),
                        "baseline_probability": float(baseline[family][original_positions, base_classes[family]][mask].mean()),
                        "retained_baseline_probability": float(stability.mean(0)[mask].mean()),
                        "probability_change": float(delta.mean(0)[mask].mean()),
                        "flip_fraction": float(flips.mean(0)[mask].mean()),
                        "accuracy": float(correct.mean(0)[mask].mean()),
                        "initially_both_correct_n": int(retained_mask.sum()),
                        "correct_label_retention": float(correct.mean(0)[retained_mask].mean()) if retained_mask.any() else None,
                        "noise_repeat_flip_fraction_min": float(flips[:, mask].mean(1).min()),
                        "noise_repeat_flip_fraction_max": float(flips[:, mask].mean(1).max()),
                    })
    table = out / "tables"
    keys = list(records[0])
    write_csv(table / "stability_metrics.csv", keys, ([r[k] for k in keys] for r in records))
    write_csv(table / "per_image_metrics.csv", [
        "sample_index", "true_label", "initially_both_correct", "model", "transformation",
        "level", "realization", "baseline_prediction", "transformed_prediction",
        "retained_baseline_probability", "label_flipped", "correct_after_transform",
    ], sample_records)
    figures = out / "figures"
    (figures / "curves").mkdir(parents=True, exist_ok=True)
    colors = {"resnet": "#2678ad", "vit": "#c25a7c"}
    for kind, levels in config["levels"].items():
        reference = 1 if kind in ("brightness", "red_scale", "green_scale", "blue_scale") else 0
        x = sorted([*levels, reference])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for family in FAMILIES:
            relevant = {r["level"]: r for r in records if r["model"] == family and r["transformation"] == kind and r["class_id"] == -1}
            base_prob = baseline[family][original_positions, base_classes[family]].mean()
            for ax, metric, ref_value, title in zip(
                axes, ("retained_baseline_probability", "flip_fraction", "correct_label_retention"),
                (base_prob, 0, 1 if both_correct.any() else np.nan),
                ("Probability of original decision", "Fraction with changed label", "Correct-label retention · initially both correct"),
            ):
                values = [ref_value if v == reference else relevant[v][metric] for v in x]
                ax.plot(x, values, "o-", color=colors[family], label=family.upper())
                ax.set(xlabel=TRANSFORM_NAMES[kind], ylabel=title, ylim=(-.03, 1.03))
                ax.grid(alpha=.2); ax.legend()
        fig.suptitle(f"2A · {kind} · N=320; correct-retention subset N={both_correct.sum()}\nNoise: mean of 3 fixed realizations per image; stability ≠ correctness", fontsize=12)
        fig.savefig(figures / "curves" / f"{kind}.png", dpi=150)
        plt.close(fig)

    # Equal weight per configured severity level; baseline reference is excluded.
    kinds = list(config["levels"])
    fingerprint = {}
    fp_rows = []
    for family in FAMILIES:
        matrix = np.zeros((8, len(kinds)))
        for c in range(8):
            for j, kind in enumerate(kinds):
                matrix[c, j] = np.mean([r["probability_change"] for r in records
                                        if r["model"] == family and r["class_id"] == c and r["transformation"] == kind])
                fp_rows.append([family, c, kind, matrix[c, j]])
        fingerprint[family] = matrix
    write_csv(table / "behavioral_fingerprints.csv", ["model", "true_class", "transformation", "mean_probability_change"], fp_rows)
    difference = fingerprint["vit"] - fingerprint["resnet"]
    matrices = [fingerprint["resnet"], fingerprint["vit"], difference]
    scale = max(max(float(np.abs(m).max()) for m in matrices), .01)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, ["ResNet Δ stability", "ViT Δ stability", "ViT − ResNet Δ stability"]):
        im = ax.imshow(matrix, vmin=-scale, vmax=scale, cmap="RdBu")
        ax.set(xticks=np.arange(len(kinds)), xticklabels=kinds, yticks=np.arange(8), yticklabels=SHORT_NAMES, title=title)
        plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
        for c in range(8):
            for j in range(len(kinds)):
                ax.text(j, c, f"{matrix[c,j]:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle("2B · Behavioral fingerprints · mean change in probability of original decision\nEqual weighting over tested levels; positive ViT − ResNet means a smaller loss (or greater gain) for ViT", fontsize=12)
    fig.savefig(figures / "behavioral_fingerprints.png", dpi=160); plt.close(fig)

    # Selected, not representative: per transformation show the largest drop in baseline probability.
    grid_dir = figures / "transition_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for kind, levels in config["levels"].items():
        cs = [c for c in all_conditions if c["kind"] == kind and c["realization"] == 0]
        drops = np.stack([
            baseline[f][original_positions, base_classes[f]] - loaded[c["key"]][f][original_positions, base_classes[f]]
            for c in cs for f in FAMILIES
        ]).max(0)
        positions = np.argsort(-drops, kind="stable")[:2]
        transformed = {c["key"]: cached_images(out, config, signature, images, c) for c in cs}
        for pos in positions:
            selected.append([kind, int(indices[pos]), float(drops[pos]), "largest original-decision probability drop; noise realization 0"])
            fig, axes = plt.subplots(1, len(cs)+1, figsize=(3*(len(cs)+1), 4.7), squeeze=False, constrained_layout=True)
            for col, c in enumerate([None, *cs]):
                ax = axes[0, col]
                ax.imshow(images[pos] if c is None else transformed[c["key"]][pos])
                title = "Original" if c is None else f"{c['level']:g}"
                lines = []
                for f in FAMILIES:
                    p = baseline[f][pos] if c is None else loaded[c["key"]][f][pos]
                    pred = int(p.argmax())
                    lines.append(f"{f.upper()}: {SHORT_NAMES[pred]}\np(pred)={p[pred]:.1%}; p(original)={p[base_classes[f][pos]]:.1%}")
                ax.set_title(title, fontsize=10); ax.set_xlabel("\n".join(lines), fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"2C · {kind} · sample #{indices[pos]} · true: {SHORT_NAMES[labels[pos]]}\nSelected for a large change; not a representative sample", fontsize=12)
            fig.savefig(grid_dir / f"{kind}_sample_{indices[pos]}.png", dpi=130); plt.close(fig)
    write_csv(table / "selected_transition_examples.csv", ["transformation", "sample_index", "max_probability_drop", "selection_rule"], selected)
    save_json(out / "summary.json", {
        "protocol_signature": signature, "n_images": 320,
        "initially_both_correct_n": int(both_correct.sum()),
        "baseline_accuracy": {f: float((base_classes[f] == labels).mean()) for f in FAMILIES},
        "n_conditions": len(all_conditions), "n_transformations": len(config["levels"]),
        "interpretation": "Stability is not correctness; changes are not guaranteed label preserving. No architecture superiority is assumed.",
        "fingerprint_aggregation": "Mean probability change from cached baseline; average noise repeats per image, then levels equally; group by true class.",
    })
    print(f"2A/2B/2C reports saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/stability")
    parser.add_argument("--approve-pilot", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    config, signature, images, labels, indices, baseline = setup(args)
    if args.stage == "pilot":
        pilot(args.output, config, signature, images, labels)
    elif args.stage == "run":
        run(args, config, signature, images, labels, indices, baseline)
    else:
        report(args.output, config, signature, images, labels, indices, baseline)


if __name__ == "__main__":
    main()
