#!/usr/bin/env python3
"""Experiment 3: black-box, blurred-patch occlusion at original RGB resolution.

python skript/occlusion.py prepare
python skript/occlusion.py run       # resumable; reports generated on completion
python skript/occlusion.py report    # figures/tables only; no inference
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter

from analysis import ROOT, SHORT_NAMES, load_predictions, write_csv

FAMILIES = ("resnet", "vit")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def positions():
    """Row-major (top, left) coordinates for 4x4 windows, stride 2."""
    return np.asarray([(top, left) for top in range(0, 25, 2)
                       for left in range(0, 25, 2)], dtype=np.int64)


def occluded_versions(image, blur_sigma):
    """Replace only each 4x4 region with the corresponding blurred RGB pixels.

    Blur the full original image first so local patches use surrounding context,
    not artificial blur boundaries within each 4x4 crop. No resizing/normalizing
    occurs here; both models receive these exact modified uint8 arrays.
    """
    if image.dtype != np.uint8 or image.shape != (28, 28, 3):
        raise ValueError("Expected one 28x28 uint8 RGB image")
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=blur_sigma)))
    modified = np.repeat(image[None], 169, axis=0)
    for i, (top, left) in enumerate(positions()):
        modified[i, top:top+4, left:left+4] = blurred[top:top+4, left:left+4]
    return modified


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if not config_path.exists():
        save_json(config_path, {"schema_version": 1, "split": "test", "per_class": 6,
                                "seed": 43, "patch_size": 4, "stride": 2, "blur_sigma_pixels": 2.0,
                                "selection": "Up to two model-disagreement images per class, remaining slots sampled uniformly without replacement from that class"})
    config = json.loads(config_path.read_text())
    if (config["per_class"], config["patch_size"], config["stride"]) != (6, 4, 2) or config["blur_sigma_pixels"] <= 0:
        raise ValueError("Require six images/class, 4x4 patches, stride 2, positive blur sigma")
    split = config["split"]
    names_dict = json.loads((ROOT / "data/metadata.json").read_text())["label"]
    names = [names_dict[str(i)] for i in range(8)]
    original = {f: load_predictions(ROOT / f"preds/{f}/{split}.npz", names) for f in FAMILIES}
    # Alignment is explicit rather than assuming file row equals sample_index.
    np.testing.assert_array_equal(original["resnet"]["sample_index"], original["vit"]["sample_index"])
    np.testing.assert_array_equal(original["resnet"]["labels"], original["vit"]["labels"])
    with np.load(ROOT / "data/bloodmnist.npz", allow_pickle=False) as z:
        all_images = z[f"{split}_images"]
        all_labels = z[f"{split}_labels"].reshape(-1).astype(np.int64)
    candidate_ids = original["resnet"]["sample_index"]
    np.testing.assert_array_equal(all_labels[candidate_ids], original["resnet"]["labels"])
    disagree = original["resnet"]["predicted_labels"] != original["vit"]["predicted_labels"]
    manifest_path = out / "subset/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["split"] != split or manifest["seed"] != config["seed"]:
            raise ValueError("Subset settings changed; use a new output directory")
        indices = np.asarray(manifest["sample_indices"], dtype=np.int64)
    else:
        rng = np.random.default_rng(config["seed"])
        selected = []
        for cls in range(8):
            candidates = candidate_ids[original["resnet"]["labels"] == cls]
            disagreement_ids = candidate_ids[(original["resnet"]["labels"] == cls) & disagree]
            priority = rng.choice(disagreement_ids, size=min(2, len(disagreement_ids)), replace=False)
            rest = rng.choice(np.setdiff1d(candidates, priority), size=6-len(priority), replace=False)
            selected.extend([*priority, *rest])
        indices = np.sort(selected).astype(np.int64)
        manifest = {"split": split, "seed": config["seed"], "sample_indices": indices.tolist(),
                    "selection": config["selection"], "class_names": names}
    images, labels = all_images[indices], all_labels[indices]
    if len(indices) != 48 or len(np.unique(indices)) != 48 or not np.all(np.bincount(labels, minlength=8) == 6):
        raise ValueError("Expected 48 distinct images, six per true class")
    baseline = {}
    for family in FAMILIES:
        lookup = {int(x): i for i, x in enumerate(original[family]["sample_index"])}
        rows = [lookup[int(x)] for x in indices]
        baseline[family] = original[family]["probabilities"][rows]
    # Weights/config hashes protect resumed runs against silently mixing checkpoints.
    assets = {rel: file_digest(ROOT / rel) for rel in (
        "models/resnet18/resnet18_28_1.pth", "models/resnet18/models.py",
        "models/vit/model.safetensors", "models/vit/config.json", "models/vit/preprocessor_config.json")}
    signature = hashlib.sha256(json.dumps({
        "config": config, "indices": indices.tolist(), "images": digest(images),
        "labels": digest(labels), "baseline": {f: digest(p) for f, p in baseline.items()},
        "assets": assets,
    }, sort_keys=True).encode()).hexdigest()
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["signature"] != signature:
        raise ValueError("Protocol/assets/baseline changed; use a new --output directory")
    save_json(protocol_path, {"signature": signature, "config": config, "asset_sha256": assets})
    manifest.update(images_sha256=digest(images), true_labels=labels.tolist(),
                    disagreement_count=int((baseline["resnet"].argmax(1) != baseline["vit"].argmax(1)).sum()))
    save_json(manifest_path, manifest)
    write_csv(out / "subset/images.csv", ["sample_index", "true_label", "true_class", "resnet_prediction", "vit_prediction"],
              ([int(index), int(labels[i]), names[labels[i]], int(baseline['resnet'][i].argmax()),
                int(baseline['vit'][i].argmax())] for i, index in enumerate(indices)))
    return config, signature, indices, images, labels, baseline, names


def cached_interventions(out, signature, image, index, sigma):
    path = out / "interventions" / f"sample_{index:05d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["protocol_signature"]) != signature:
                raise ValueError(f"Stale interventions: {path}")
            modified = z["images"]
            if modified.shape != (169, 28, 28, 3) or modified.dtype != np.uint8:
                raise ValueError(f"Invalid intervention array: {path}")
            return modified
    modified = occluded_versions(image, sigma)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=modified, positions=positions(), protocol_signature=signature)
    return modified


def run(args, config, signature, indices, images, labels, baseline, names):
    from inference import LocalPredictor
    for family in FAMILIES:
        folder = args.output / "predictions" / family
        folder.mkdir(parents=True, exist_ok=True)
        pending = [i for i, index in enumerate(indices) if not (folder / f"sample_{index:05d}.npz").exists()]
        if not pending:
            print(f"{family}: all 48 cached images found; skipping inference", flush=True)
            continue
        started = time.perf_counter()
        predictor = LocalPredictor(family, names, args.device, args.batch_size, config["seed"])
        # Small reproducibility/preprocessing audit; full cached baseline is reused.
        audit_pos = np.asarray([np.flatnonzero(labels == c)[0] for c in range(8)])
        repeats = np.stack([predictor.predict(images[audit_pos])[1] for _ in range(3)])
        if not np.all(repeats.argmax(-1) == baseline[family][audit_pos].argmax(-1)):
            raise ValueError("Cached baseline decisions differ from current model/preprocessing")
        for count, i in enumerate(pending, 1):
            index = int(indices[i])
            modified = cached_interventions(args.output, signature, images[i], index, config["blur_sigma_pixels"])
            logits, probabilities = predictor.predict(modified)
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-5)
            path = folder / f"sample_{index:05d}.npz"
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, logits=logits, probabilities=probabilities,
                                positions=positions(), sample_index=index, protocol_signature=signature,
                                intervention_sha256=digest(modified))
            tmp.replace(path)
            print(f"{family} {count}/{len(pending)}: sample #{index}, 169 occlusions", flush=True)
        info = dict(predictor.info,
                    audit_sample_indices=indices[audit_pos].tolist(), audit_repeats=3,
                    max_probability_range_across_repeats=float(np.ptp(repeats, axis=0).max()),
                    max_abs_difference_from_cached_baseline=float(np.abs(repeats[0]-baseline[family][audit_pos]).max()),
                    audit_prediction_changes=int(np.any(repeats.argmax(-1) != repeats[0].argmax(-1), axis=0).sum()),
                    audit_scope="Eight baseline images only; not a cross-device determinism guarantee",
                    warnings=sorted(predictor.warning_messages), elapsed_seconds=time.perf_counter()-started)
        save_json(folder / "reproducibility.json", info)
        predictor.close()
    report(args.output, config, signature, indices, images, labels, baseline, names)


def targets(true_class, resnet_class, vit_class):
    result = [("true_class", int(true_class))]
    if resnet_class != vit_class:
        # Each row compares the SAME target for BOTH models, even if targets repeat.
        result.extend([("resnet_preferred", int(resnet_class)), ("vit_preferred", int(vit_class))])
    elif resnet_class != true_class:
        result.append(("shared_preferred", int(resnet_class)))
    return result


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pilot(out, config, indices, images, labels):
    plt = plotting()
    fig, axes = plt.subplots(8, 4, figsize=(9, 17), constrained_layout=True)
    for c in range(8):
        i = int(np.flatnonzero(labels == c)[0])
        modified = occluded_versions(images[i], config["blur_sigma_pixels"])
        for j, (title, img) in enumerate([
            ("Original", images[i]), ("Top-left patch", modified[0]),
            ("Central patch", modified[84]), ("Bottom-right patch", modified[-1])]):
            axes[c, j].imshow(img); axes[c, j].set_title(title, fontsize=9)
            axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
        axes[c, 0].set_ylabel(f"{SHORT_NAMES[c]}\n#{indices[i]}", fontsize=8)
    fig.suptitle(f"4×4 blurred-patch intervention pilot · σ={config['blur_sigma_pixels']:g}px\nOnly the patch is replaced; modifications precede model preprocessing")
    path = out / "pilot/occlusion_examples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def report(out, config, signature, indices, images, labels, baseline, names):
    plt = plotting()
    loaded = {}
    # Validate every cache before writing final reports.
    for index in indices:
        hashes = []
        for family in FAMILIES:
            path = out / "predictions" / family / f"sample_{index:05d}.npz"
            with np.load(path, allow_pickle=False) as z:
                if str(z["protocol_signature"]) != signature or int(z["sample_index"]) != index:
                    raise ValueError(f"Cache mismatch: {path}")
                np.testing.assert_array_equal(z["positions"], positions())
                p = z["probabilities"]
                if p.shape != (169, 8) or not np.isfinite(p).all() or (p < 0).any():
                    raise ValueError(f"Invalid probabilities: {path}")
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
                loaded[int(index), family] = p
                hashes.append(str(z["intervention_sha256"]))
        if hashes[0] != hashes[1]:
            raise ValueError("Models saw different occluded inputs")
    table_rows, sample_rows = [], []
    all_maps = []
    # Shared symmetric scale ACROSS all displayed images, targets, and models.
    for i, index in enumerate(indices):
        for target_name, target_class in targets(labels[i], baseline['resnet'][i].argmax(), baseline['vit'][i].argmax()):
            for family in FAMILIES:
                all_maps.append(baseline[family][i, target_class] - loaded[int(index), family][:, target_class])
    vmax = max(max(float(np.abs(a).max()) for a in all_maps), 1e-6)
    for i, index in enumerate(indices):
        rp, vp = int(baseline['resnet'][i].argmax()), int(baseline['vit'][i].argmax())
        chosen_targets = targets(labels[i], rp, vp)
        fig, axes = plt.subplots(len(chosen_targets), 3, figsize=(12, 4.2*len(chosen_targets)),
                                 squeeze=False, constrained_layout=True)
        image_maps = {}
        for row, (target_name, target_class) in enumerate(chosen_targets):
            axes[row, 0].imshow(images[i])
            axes[row, 0].set_title(f"Original #{index} · true: {SHORT_NAMES[labels[i]]}\n"
                                   f"ResNet: {SHORT_NAMES[rp]} · ViT: {SHORT_NAMES[vp]}", fontsize=10)
            axes[row, 0].axis("off")
            for col, family in enumerate(FAMILIES, 1):
                original_probability = float(baseline[family][i, target_class])
                sensitivity = original_probability - loaded[int(index), family][:, target_class]
                matrix = sensitivity.reshape(13, 13)
                image_maps[f"{target_name}_{family}"] = matrix
                im = axes[row, col].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                           extent=(1, 27, 27, 1), interpolation="nearest")
                axes[row, col].set_title(f"{family.upper()} · target: {SHORT_NAMES[target_class]}\n"
                                         f"{target_name} · baseline p={original_probability:.3f}", fontsize=10)
                axes[row, col].set(xlabel="Occlusion center x (pixel boundary coordinates)",
                                    ylabel="Occlusion center y (pixel boundary coordinates)",
                                    xticks=[2, 8, 14, 20, 26], yticks=[2, 8, 14, 20, 26])
                for j, (top, left) in enumerate(positions()):
                    table_rows.append([int(index), int(labels[i]), rp, vp, target_name, target_class,
                                       family, int(top), int(left), int(top+2), int(left+2),
                                       original_probability, float(loaded[int(index), family][j, target_class]),
                                       float(sensitivity[j])])
                sample_rows.append([int(index), target_name, target_class, family,
                                    original_probability, float(sensitivity.min()), float(sensitivity.max()),
                                    float(sensitivity.mean()), float(np.abs(sensitivity).mean())])
        fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=.65,
                     label="Baseline p(target) − occluded p(target)\nPositive: occlusion reduces target score")
        fig.suptitle("Experiment 3 · blurred-patch occlusion sensitivity\nSame target across models in each row; globally matched symmetric color scale", fontsize=13)
        path = out / "figures/sensitivity_maps" / f"sample_{index:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140); plt.close(fig)
        maps_path = out / "maps" / f"sample_{index:05d}.npz"
        maps_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(maps_path, **image_maps, sample_index=index,
                            true_class=labels[i], resnet_prediction=rp, vit_prediction=vp,
                            positions=positions(), target_names=np.asarray([t[0] for t in chosen_targets]),
                            target_classes=np.asarray([t[1] for t in chosen_targets]))
    write_csv(out / "tables/occlusion_sensitivity.csv", [
        "sample_index", "true_label", "resnet_prediction", "vit_prediction", "target_type",
        "target_class", "model", "window_top", "window_left", "center_y", "center_x",
        "baseline_target_probability", "occluded_target_probability", "importance"], table_rows)
    write_csv(out / "tables/map_summary.csv", [
        "sample_index", "target_type", "target_class", "model", "baseline_target_probability",
        "min_importance", "max_importance", "mean_importance", "mean_absolute_importance"], sample_rows)
    save_json(out / "summary.json", {
        "protocol_signature": signature, "n_images": 48, "per_true_class": 6,
        "disagreement_images": int(sum(baseline['resnet'][i].argmax() != baseline['vit'][i].argmax() for i in range(48))),
        "occlusions_per_image_per_model": 169, "total_intervention_evaluations": 48*169*2,
        "extra_audit_evaluations": 8*3*2,
        "baseline": "Reused from preds/; only a small reproducibility audit is rerun",
        "global_color_scale": [-vmax, vmax], "blur_sigma_pixels": config["blur_sigma_pixels"],
        "importance": "baseline p(target) - occluded p(target)",
        "map_coordinates": "13x13 row-major patch positions, not a per-pixel attribution map",
        "selection_bias": "Balanced six/class, enriched for disagreements; descriptive examples, not a population estimate",
        "limitations": "Artificial intervention sensitivity only; does not establish recognition of biological structures. Blur method, sigma, and interpolation affect maps.",
    })
    print(f"Experiment 3 reports saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/occlusion")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    bundle = prepare(args)
    if args.stage == "prepare":
        config, signature, indices, images, labels, baseline, names = bundle
        pilot(args.output, config, indices, images, labels)
        print(f"Prepared 48 images and pilot in {args.output}")
    elif args.stage == "run":
        run(args, *bundle)
    else:
        report(args.output, *bundle)


if __name__ == "__main__":
    main()
