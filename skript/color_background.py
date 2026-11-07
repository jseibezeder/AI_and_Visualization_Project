#!/usr/bin/env python3
"""Experiment 4: color ablations and reviewed approximate cell/periphery controls.

python skript/color_background.py prepare
# Inspect pilot/masks.png and pilot/color_examples.png; document mask review.
python skript/color_background.py run
python skript/color_background.py report

No training/downloads. Experiment 3's exact ordered 48-image manifest is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from analysis import ROOT, SHORT_NAMES, load_predictions, write_csv
from occlusion import FAMILIES, digest, file_digest, plotting, save_json

SCHEMA = 1
# Pixel-boundary coordinates (x, y, radius_x, radius_y), individually selected
# from original RGB images BEFORE Experiment 4 inference. Not segmentations.
MASK_PROPOSALS = {
    65: (14, 14, 10, 10), 164: (13.5, 14, 10, 10),
    873: (14.5, 18.5, 10, 9), 906: (14, 14.5, 10, 10.5),
    689: (14, 15.5, 8.5, 8.5), 1013: (14, 15, 9, 9),
    48: (14, 16, 10.5, 10.5), 105: (14, 14, 10.5, 10.5),
    1213: (14, 15, 9, 10), 2055: (14, 16, 9, 10.5),
    50: (14, 14, 11, 10.5), 432: (15, 15, 11, 12),
    194: (15, 17, 11, 10.5), 809: (14, 15, 10, 11),
    447: (14, 13.5, 4.5, 5), 1151: (13.5, 13.5, 5, 5),
}


def color_conditions():
    result = [{"key": "original", "kind": "identity", "level": 1., "mild": False}]
    result.append({"key": "grayscale", "kind": "saturation", "level": 0., "mild": False})
    for level in (.5, .8, 1.2, 1.5):
        result.append({"key": f"saturation_{level:g}", "kind": "saturation",
                       "level": level, "mild": level in (.8, 1.2)})
    for kind in ("red_scale", "blue_scale"):
        for level in (.95, 1.05):
            result.append({"key": f"{kind}_{level:g}", "kind": kind, "level": level, "mild": True})
    for level in (.9, .95, 1.05, 1.1):
        result.append({"key": f"brightness_{level:g}", "kind": "brightness", "level": level, "mild": True})
    return result


def color_transform(images, condition):
    """Original-resolution uint8 transformations; identity preserves exact bytes."""
    kind, level = condition["kind"], condition["level"]
    if kind == "identity":
        return images.copy()
    if kind in ("red_scale", "blue_scale"):
        result = images.astype(np.float64)
        result[..., 0 if kind == "red_scale" else 2] *= level
        return np.rint(np.clip(result, 0, 255)).astype(np.uint8)
    enhance = {"saturation": ImageEnhance.Color, "brightness": ImageEnhance.Brightness}[kind]
    return np.stack([np.asarray(enhance(Image.fromarray(x)).enhance(level)) for x in images])


def ellipse_mask(ellipse):
    cx, cy, rx, ry = ellipse
    if rx <= 0 or ry <= 0:
        raise ValueError("Ellipse radii must be positive")
    y, x = np.mgrid[:28, :28] + .5
    mask = ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1
    if not mask.any() or mask.all():
        raise ValueError("Need nonempty interior and exterior")
    return mask


def intervention_stats(original, changed, mask):
    d = changed.astype(np.float64) - original.astype(np.float64)
    changed_pixels = np.any(d != 0, axis=-1)
    area = int(mask.sum())
    return {"selected_pixels": area, "actually_changed_pixels": int(changed_pixels.sum()),
            "selected_fraction": area / 784,
            "mean_abs_rgb_change_selected": float(np.abs(d[mask]).mean()),
            "l1_rgb_total": float(np.abs(d).sum()),
            "rms_rgb_change_whole_image": float(np.sqrt(np.mean(d*d))),
            "outside_selected_changed_pixels": int((changed_pixels & ~mask).sum())}


def region_versions(image, mask, index, config):
    """Full-region ablations plus paired exact-area/exact-intensity controls.

    Matched masks draw k pixels from each region with identical RNG policy,
    three repeats. Eligible RGB channels are all in [16,239], so adding any
    configured signed integer delta cannot clip. Therefore both members have
    identical nonzero area, L1 and L2 energy; no per-image response normalization.
    """
    versions, metadata, selections = [], [], []

    def append(key, changed, selection, **extra):
        if np.any(changed[~selection] != image[~selection]):
            raise ValueError("Intervention modified pixels outside its mask")
        versions.append(changed)
        selections.append(selection)
        metadata.append(dict(key=key, **intervention_stats(image, changed, selection), **extra))

    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(config["region_blur_sigma"])))
    gray = color_transform(image[None], {"kind": "saturation", "level": 0})[0]
    for operation, replacement in (("blur", blurred), ("grayscale", gray)):
        for region, selection in (("central", mask), ("peripheral", ~mask)):
            changed = image.copy()
            changed[selection] = replacement[selection]
            append(f"whole_{region}_{operation}", changed, selection, design="whole_region",
                   region=region, operation=operation, repeat=-1, delta=0)
    budget = max(abs(x) for x in config["matched_deltas"])
    eligible = ((image >= budget) & (image <= 255-budget)).all(-1)
    candidates = {"central": np.flatnonzero(mask & eligible),
                  "peripheral": np.flatnonzero(~mask & eligible)}
    k = min(config["matched_max_pixels"], *(len(x) for x in candidates.values()))
    if k < config["matched_min_pixels"]:
        raise ValueError(f"sample {index}: only {k} eligible pixels per region; review mask")
    for repeat in range(config["matched_repeats"]):
        for region, pool in candidates.items():
            seed = int.from_bytes(hashlib.sha256(f'{config["seed"]}:{index}:{repeat}:{region}'.encode()).digest()[:8], "little")
            selected = np.random.default_rng(seed).choice(pool, size=k, replace=False)
            selection = np.zeros((28, 28), dtype=bool)
            selection.ravel()[selected] = True
            for delta in config["matched_deltas"]:
                changed = image.copy()
                changed[selection] = (image[selection].astype(np.int16) + delta).astype(np.uint8)
                append(f"matched_{region}_{delta:+d}_r{repeat}", changed, selection,
                       design="matched", region=region, operation="additive_rgb",
                       repeat=repeat, delta=delta)
    return np.stack(versions), metadata, np.stack(selections)


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **values)
    temp.replace(path)


def setup(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if not config_path.exists():
        save_json(config_path, {"schema_version": SCHEMA, "seed": 44,
            "source_manifest": "results/occlusion/subset/manifest.json",
            "color_conditions": color_conditions(), "region_blur_sigma": 2.,
            "matched_deltas": [-16, -8, 8, 16], "matched_max_pixels": 64,
            "matched_min_pixels": 24, "matched_repeats": 3,
            "baseline_atol": 5e-4, "baseline_rtol": 1e-4,
            "note": "Grayscale/strong saturation are feature ablations. Mild is an operational magnitude label, not evidence of stain realism or label preservation."})
    config = json.loads(config_path.read_text())
    if config["schema_version"] != SCHEMA or config["color_conditions"] != color_conditions():
        raise ValueError("Unsupported schema/condition grid; use a new output for protocol changes")
    if (config["matched_repeats"] < 1 or config["matched_min_pixels"] < 1
            or config["matched_max_pixels"] < config["matched_min_pixels"]
            or not config["matched_deltas"] or any(type(d) is not int or d == 0 or abs(d) > 127 for d in config["matched_deltas"])
            or len(set(config["matched_deltas"])) != len(config["matched_deltas"])
            or config["region_blur_sigma"] <= 0):
        raise ValueError("Invalid intervention configuration")
    source = json.loads((ROOT / config["source_manifest"]).read_text())
    if source["split"] != "test":
        raise ValueError("Require Experiment 3 test subset")
    ids = np.array(source["sample_indices"], dtype=np.int64)
    names = source["class_names"]
    with np.load(ROOT / "data/bloodmnist.npz", allow_pickle=False) as z:
        images = z["test_images"][ids]
        labels = z["test_labels"][ids].reshape(-1).astype(np.int64)
    if len(ids) != 48 or len(np.unique(ids)) != 48 or not np.all(np.bincount(labels, minlength=8) == 6):
        raise ValueError("Require original 48 images, six per class")
    if digest(images) != source["images_sha256"] or labels.tolist() != source["true_labels"]:
        raise ValueError("Experiment 3 subset does not match dataset")
    cached_baseline = {}
    for family in FAMILIES:
        p = load_predictions(ROOT / f"preds/{family}/test.npz", names)
        lookup = {int(x): i for i, x in enumerate(p["sample_index"])}
        rows = [lookup[int(x)] for x in ids]
        np.testing.assert_array_equal(p["labels"][rows], labels)
        cached_baseline[family] = p["probabilities"][rows]
    mask_path = out / "masks/review.json"
    if not mask_path.exists():
        save_json(mask_path, {"status": "awaiting_visual_review", "reviewer": None,
            "scope": "Approximate geometric central-cell regions, not expert biological segmentation. Peripheral regions can contain other cells.",
            "selection": "Two visually inspectable original images per class, chosen before Experiment 4 inference; no selection by new model effects.",
            "excluded_examples": {"118": "Target is clipped at the left image boundary.",
                                  "881": "Cell is cropped/irregular near the lower-right edge."},
            "images_sha256": digest(images),
            "regions": [{"sample_index": i, "ellipse": list(e), "accepted": False,
                         "note": "Review original, contour and intervention previews before accepting."}
                        for i, e in sorted(MASK_PROPOSALS.items())]})
    review = json.loads(mask_path.read_text())
    if review["images_sha256"] != digest(images):
        raise ValueError("Mask review belongs to different source images")
    records = review["regions"]
    mask_ids = [r["sample_index"] for r in records]
    if len(mask_ids) != 16 or len(set(mask_ids)) != 16 or not set(mask_ids) <= set(ids.tolist()):
        raise ValueError("Require 16 unique reviewed images from the explanation set")
    id_to_pos = {int(index): pos for pos, index in enumerate(ids)}
    if not np.all(np.bincount([labels[id_to_pos[i]] for i in mask_ids], minlength=8) == 2):
        raise ValueError("Mask subset must have two images per class")
    masks = {r["sample_index"]: ellipse_mask(r["ellipse"]) for r in records}
    assets = {rel: file_digest(ROOT / rel) for rel in (
        "models/resnet18/resnet18_28_1.pth", "models/resnet18/models.py",
        "models/vit/model.safetensors", "models/vit/config.json", "models/vit/preprocessor_config.json")}
    # Freeze implementation too: modified code cannot silently reuse old caches.
    code = {rel: file_digest(ROOT / rel) for rel in ("skript/color_background.py", "skript/inference.py")}
    frozen = {"config": config, "source_manifest": source, "mask_review": review,
              "asset_sha256": assets, "implementation_sha256": code,
              "baseline_sha256": {f: digest(p) for f, p in cached_baseline.items()}}
    signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["signature"] != signature:
        raise ValueError("Frozen protocol changed. Preserve this run; choose a new --output directory.")
    save_json(out / "subset/manifest.json", dict(source, reused_from=config["source_manifest"],
              region_sample_indices=mask_ids, note="Exact Experiment 3 order retained, including disagreement enrichment."))
    return config, signature, frozen, ids, images, labels, cached_baseline, names, review, masks


def cache_inputs(out, signature, key, images, ids, selections=None):
    path = out / "interventions" / f"{key}.npz"
    expected_hash = digest(images)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["protocol_signature"]) != signature or str(z["input_sha256"]) != expected_hash:
                raise ValueError(f"Stale/corrupt intervention cache: {path}")
            np.testing.assert_array_equal(z["sample_index"], ids)
            np.testing.assert_array_equal(z["images"], images)
            if selections is not None:
                np.testing.assert_array_equal(z["selected_masks"], selections)
    else:
        extra = {} if selections is None else {"selected_masks": selections}
        atomic_npz(path, images=images, sample_index=ids, input_sha256=expected_hash,
                   protocol_signature=signature, **extra)
    return expected_hash


def jobs(config, ids, images, masks):
    for c in config["color_conditions"]:
        yield f'color/{c["key"]}', color_transform(images, c), ids, None, None
    for index, mask in masks.items():
        image = images[int(np.flatnonzero(ids == index)[0])]
        changed, meta, selections = region_versions(image, mask, index, config)
        yield f"regions/sample_{index:05d}", changed, np.full(len(changed), index, dtype=np.int64), meta, selections


def validate_prediction(path, signature, ids, input_hash, environment_signature=None):
    with np.load(path, allow_pickle=False) as z:
        if str(z["protocol_signature"]) != signature or str(z["input_sha256"]) != input_hash:
            raise ValueError(f"Stale prediction cache: {path}")
        if environment_signature is not None and str(z["environment_signature"]) != environment_signature:
            raise ValueError("Cannot mix prediction environments/devices in a resumed run")
        np.testing.assert_array_equal(z["sample_index"], ids)
        p, logits = z["probabilities"], z["logits"]
        if p.shape != (len(ids), 8) or logits.shape != p.shape or not np.isfinite(logits).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"Invalid predictions: {path}")
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
        x = logits.astype(np.float64) - logits.max(1, keepdims=True)
        expected = np.exp(x); expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(p, expected, atol=2e-6, rtol=2e-5)
        return p.copy()


def prepare(args, bundle):
    config, signature, frozen, ids, images, labels, baseline, names, review, masks = bundle
    plt = plotting()
    preview_positions = [int(np.flatnonzero(labels == c)[0]) for c in range(8)]
    conditions = config["color_conditions"]
    fig, axes = plt.subplots(8, len(conditions), figsize=(23, 14), constrained_layout=True)
    for j, c in enumerate(conditions):
        changed = color_transform(images, c)
        for row, pos in enumerate(preview_positions):
            ax = axes[row, j]; ax.imshow(changed[pos]); ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(c["key"].replace("_", "\n"), fontsize=7)
            if j == 0: ax.set_ylabel(f"#{ids[pos]}\n{SHORT_NAMES[row]}", fontsize=8)
    fig.suptitle("4A pilot: original-resolution color changes. Grayscale and strong saturation are feature ablations, not validated stain changes.")
    path = args.output / "pilot/color_examples.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    ordered = sorted(masks, key=lambda index: (int(labels[np.flatnonzero(ids == index)[0]]), index))
    fig, axes = plt.subplots(len(ordered), 4, figsize=(10, 2.1*len(ordered)), constrained_layout=True)
    for row, index in enumerate(ordered):
        pos = int(np.flatnonzero(ids == index)[0]); image = images[pos]; mask = masks[index]
        changed, meta, _ = region_versions(image, mask, index, config)
        for col, img in enumerate((image, image, changed[3], changed[2])):
            ax = axes[row, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if col == 1: ax.contour(mask, levels=[.5], colors=["lime"], linewidths=1)
            if row == 0: ax.set_title(("Original", "Approximate region", "Peripheral grayscale", "Central grayscale")[col])
        axes[row, 0].set_ylabel(f"#{index} {SHORT_NAMES[labels[pos]]}\nmask={mask.sum()}px; matched k={meta[4]['selected_pixels']}", fontsize=8)
    fig.suptitle("4B mask review: geometric approximations; green contour is not a validated biological boundary.")
    fig.savefig(args.output / "pilot/masks.png", dpi=120); plt.close(fig)
    print(f"Pilots written. Review {args.output / 'masks/review.json'} before run.", flush=True)


def run(args, bundle):
    from inference import LocalPredictor
    import PIL
    import torch
    import transformers
    config, signature, frozen, ids, images, labels, cached, names, review, masks = bundle
    if review["status"] != "accepted_approximate_visual_review" or not review.get("reviewer") or not all(r["accepted"] for r in review["regions"]):
        raise ValueError("Inspect pilot/masks.png and record explicit per-image visual review first")
    save_json(args.output / "protocol.json", dict(signature=signature, **frozen))
    all_jobs = list(jobs(config, ids, images, masks))
    hashes = {}
    for key, changed, sample_ids, meta, selections in all_jobs:
        hashes[key] = cache_inputs(args.output, signature, key, changed, sample_ids, selections)
        if meta is not None:
            save_json(args.output / "interventions" / f"{key}.json", meta)
    audit_pos = np.array([np.flatnonzero(labels == c)[0] for c in range(8)])
    for family in FAMILIES:
        started = time.perf_counter()
        folder = args.output / "predictions" / family
        missing = [job for job in all_jobs if not (folder / f"{job[0]}.npz").exists()]
        audit_path = folder / "reproducibility.json"
        if not missing and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["protocol_signature"] != signature:
                raise ValueError("Stale reproducibility audit")
            for key, _, sample_ids, _, _ in all_jobs:
                validate_prediction(folder / f"{key}.npz", signature, sample_ids, hashes[key], audit["environment_signature"])
            print(f"{family}: all caches validated; no inference needed", flush=True)
            continue
        predictor = LocalPredictor(family, names, args.device, args.batch_size, config["seed"])
        env = dict(predictor.info, numpy_version=np.__version__, pillow_version=PIL.__version__,
                   transformers_version=transformers.__version__, batch_size=args.batch_size,
                   device_name=torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else str(predictor.device))
        environment_signature = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        for key, _, sample_ids, _, _ in all_jobs:
            path = folder / f"{key}.npz"
            if path.exists():
                validate_prediction(path, signature, sample_ids, hashes[key], environment_signature)
        # All 48 originals establish within-device paired baselines and an explicit
        # equivalence gate to trusted exports. Only the small 8-image repeat audit
        # is repeated; no full-dataset inference is performed.
        original_path = folder / "color/original.npz"
        original_hash = hashes["color/original"]
        if original_path.exists():
            base = validate_prediction(original_path, signature, ids, original_hash, environment_signature)
        else:
            logits, base = predictor.predict(images)
            np.testing.assert_allclose(base, cached[family], atol=config["baseline_atol"], rtol=config["baseline_rtol"])
            np.testing.assert_array_equal(base.argmax(1), cached[family].argmax(1))
            atomic_npz(original_path, logits=logits, probabilities=base, sample_index=ids,
                       protocol_signature=signature, input_sha256=original_hash,
                       environment_signature=environment_signature)
        np.testing.assert_allclose(base, cached[family], atol=config["baseline_atol"], rtol=config["baseline_rtol"])
        np.testing.assert_array_equal(base.argmax(1), cached[family].argmax(1))
        repeats = np.stack([base[audit_pos], *[predictor.predict(images[audit_pos])[1] for _ in range(2)]])
        np.testing.assert_allclose(repeats, np.broadcast_to(repeats[0], repeats.shape), atol=2e-5, rtol=1e-4)
        for number, (key, changed, sample_ids, meta, selections) in enumerate(all_jobs, 1):
            path = folder / f"{key}.npz"
            if path.exists():
                continue
            logits, probabilities = predictor.predict(changed)
            atomic_npz(path, logits=logits, probabilities=probabilities, sample_index=sample_ids,
                       protocol_signature=signature, input_sha256=hashes[key],
                       environment_signature=environment_signature)
            validate_prediction(path, signature, sample_ids, hashes[key], environment_signature)
            print(f"{family} {number}/{len(all_jobs)}: {key} ({len(changed)} inputs)", flush=True)
        save_json(audit_path, dict(env, protocol_signature=signature, environment_signature=environment_signature,
            baseline_images=48, extra_repeat_evaluations=16, repeat_sample_indices=ids[audit_pos].tolist(),
            baseline_max_abs_error=float(np.abs(base-cached[family]).max()),
            repeat_max_probability_range=float(np.ptp(repeats, axis=0).max()),
            baseline_equivalence="All 48 decisions identical and probabilities allclose to trusted exports.",
            warnings=sorted(predictor.warning_messages), elapsed_seconds=time.perf_counter()-started))
        predictor.close()
    report(args.output, bundle)


def metric_row(index, true_label, family, base, p):
    decision = int(base.argmax()); pred = int(p.argmax())
    return {"sample_index": int(index), "true_label": int(true_label), "model": family,
            "baseline_prediction": decision, "prediction": pred,
            "label_flipped": int(pred != decision), "correct": int(pred == true_label),
            "baseline_class_probability": float(base[decision]),
            "retained_baseline_probability": float(p[decision]),
            "probability_change": float(p[decision]-base[decision]),
            "true_class_probability_change": float(p[true_label]-base[true_label]),
            "probabilities": json.dumps([float(v) for v in p])}


def csv_dicts(path, rows):
    if not rows:
        raise ValueError(f"Need explicit CSV header for empty table: {path}")
    keys = list(rows[0]); write_csv(path, keys, ([r[k] for k in keys] for r in rows))


def report(out, bundle):
    config, signature, frozen, ids, images, labels, cached, names, review, masks = bundle
    if not (out / "protocol.json").exists():
        raise ValueError("No completed inference protocol; run first")
    plt = plotting()
    loaded, metadata = {}, {}
    for key, changed, sample_ids, meta, selections in jobs(config, ids, images, masks):
        input_hash = cache_inputs(out, signature, key, changed, sample_ids, selections)
        for family in FAMILIES:
            audit = json.loads((out / f"predictions/{family}/reproducibility.json").read_text())
            if audit["protocol_signature"] != signature:
                raise ValueError("Audit/protocol mismatch")
            loaded[key, family] = validate_prediction(out / f"predictions/{family}/{key}.npz", signature, sample_ids, input_hash, audit["environment_signature"])
        metadata[key] = meta
    baseline = {f: loaded["color/original", f] for f in FAMILIES}
    for f in FAMILIES:
        np.testing.assert_array_equal(baseline[f].argmax(1), cached[f].argmax(1))
        np.testing.assert_allclose(baseline[f], cached[f], atol=config["baseline_atol"], rtol=config["baseline_rtol"])
    conditions = config["color_conditions"]
    color_rows, effects, flips = [], {}, {}
    order = np.lexsort((ids, labels))
    for family in FAMILIES:
        rows = []
        for j, c in enumerate(conditions):
            changed = color_transform(images, c)
            for i, index in enumerate(ids):
                row = metric_row(index, labels[i], family, baseline[family][i], loaded[f'color/{c["key"]}', family][i])
                row.update(transformation=c["key"], mild=c["mild"], **intervention_stats(images[i], changed[i], np.ones((28, 28), bool)))
                color_rows.append(row); rows.append(row)
        effects[family] = np.array([r["probability_change"] for r in rows]).reshape(len(conditions), 48).T
        flips[family] = np.array([r["label_flipped"] for r in rows], bool).reshape(len(conditions), 48).T
    csv_dicts(out / "tables/color_per_image.csv", color_rows)
    aggregate = []
    for family in FAMILIES:
        for j, c in enumerate(conditions):
            relevant = [r for r in color_rows if r["model"] == family and r["transformation"] == c["key"]]
            aggregate.append({"model": family, "transformation": c["key"], "mild": c["mild"], "n_images": 48,
                              "flip_count": int(flips[family][:, j].sum()), "flip_fraction": float(flips[family][:, j].mean()),
                              "mean_probability_change": float(effects[family][:, j].mean()),
                              "accuracy": float(np.mean([r["correct"] for r in relevant]))})
    csv_dicts(out / "tables/color_summary.csv", aggregate)
    figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 17), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES):
        im = ax.imshow(effects[family][order], cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for y, pos in enumerate(order):
            for j, c in enumerate(conditions):
                if flips[family][pos, j]:
                    ax.text(j, y, str(loaded[f'color/{c["key"]}', family][pos].argmax()), ha="center", va="center", fontsize=7,
                            color="white" if abs(effects[family][pos, j]) > .55 else "black")
        ax.set(xticks=np.arange(len(conditions)), xticklabels=[c["key"] for c in conditions],
               yticks=np.arange(48), yticklabels=[f"#{ids[i]} y={labels[i]} base={baseline[family][i].argmax()}" for i in order],
               title=f"{family.upper()}: probability change; number = new class on flip")
        ax.tick_params(axis="y", labelsize=7); plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.5, label="Transformed minus original p(original decision)")
    fig.suptitle("4A: same 48 explanation images, shared color scale\nClass IDs follow dataset; grayscale and saturation 0.5/1.5 are feature ablations", fontsize=13)
    fig.savefig(figures / "color_probability_matrix.png", dpi=140); plt.close(fig)
    difference = flips["vit"].astype(int) - flips["resnet"].astype(int)
    fig, ax = plt.subplots(figsize=(12, 15), constrained_layout=True)
    im = ax.imshow(difference[order], cmap="PiYG", vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=np.arange(len(conditions)), xticklabels=[c["key"] for c in conditions], yticks=np.arange(48),
           yticklabels=[f"#{ids[i]} y={labels[i]}" for i in order], title="4A: model-specific flips relative to each model's own baseline")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8); ax.tick_params(axis="y", labelsize=7)
    cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=.5)
    cb.ax.set_yticklabels(["ResNet only flips", "Both flip or neither flips", "ViT only flips"])
    fig.savefig(figures / "color_flip_difference.png", dpi=130); plt.close(fig)
    # Selection rule fixed: mild one-model-only flips, ranked by inter-model
    # difference in retained baseline probability change, then sample ID/key.
    candidates = []
    for j, c in enumerate(conditions):
        if not c["mild"]:
            continue
        for i, index in enumerate(ids):
            if flips["resnet"][i, j] != flips["vit"][i, j]:
                candidates.append({"sample_index": int(index), "transformation": c["key"],
                    "true_label": int(labels[i]), "flipped_model": "resnet" if flips["resnet"][i, j] else "vit",
                    "delta_difference_abs": float(abs(effects["vit"][i, j]-effects["resnet"][i, j])),
                    "selection": "All mild one-model-only flips; ranked for illustration, not prevalence."})
    candidates.sort(key=lambda r: (-r["delta_difference_abs"], r["sample_index"], r["transformation"]))
    fields = ["sample_index", "transformation", "true_label", "flipped_model", "delta_difference_abs", "selection"]
    write_csv(out / "tables/mild_model_specific_flips.csv", fields, ([r[k] for k in fields] for r in candidates))
    if candidates:
        chosen = candidates[:8]
        fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 3*len(chosen)), squeeze=False, constrained_layout=True)
        for row, record in enumerate(chosen):
            i = int(np.flatnonzero(ids == record["sample_index"])[0]); c = next(c for c in conditions if c["key"] == record["transformation"])
            changed = color_transform(images[i:i+1], c)[0]
            for col, image in enumerate((images[i], changed)):
                ax = axes[row, col]; ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
                lines = []
                for family in FAMILIES:
                    p = baseline[family][i] if col == 0 else loaded[f'color/{c["key"]}', family][i]
                    decision = baseline[family][i].argmax()
                    lines.append(f"{family}: class {p.argmax()}, p(original)={p[decision]:.3f}")
                ax.set_title((f"Original #{ids[i]} true={labels[i]}" if col == 0 else c["key"]) + "\n" + "\n".join(lines), fontsize=9)
        fig.suptitle("Selected mild one-model-only flips: exploratory color sensitivity, not validated microscope/stain variation")
        fig.savefig(figures / "mild_flip_examples.png", dpi=120); plt.close(fig)
    region_rows = []
    for index in masks:
        i = int(np.flatnonzero(ids == index)[0]); key = f"regions/sample_{index:05d}"
        for family in FAMILIES:
            for j, meta in enumerate(metadata[key]):
                row = metric_row(index, labels[i], family, baseline[family][i], loaded[key, family][j])
                row.update(meta); region_rows.append(row)
    csv_dicts(out / "tables/region_per_image.csv", region_rows)
    paired = []
    for index in masks:
        for family in FAMILIES:
            for delta in config["matched_deltas"]:
                a = [r for r in region_rows if r["sample_index"] == index and r["model"] == family and r["design"] == "matched" and r["delta"] == delta]
                central = sorted([r for r in a if r["region"] == "central"], key=lambda r: r["repeat"])
                peripheral = sorted([r for r in a if r["region"] == "peripheral"], key=lambda r: r["repeat"])
                for c, p in zip(central, peripheral):
                    if c["selected_pixels"] != p["selected_pixels"] or c["actually_changed_pixels"] != p["actually_changed_pixels"] or c["l1_rgb_total"] != p["l1_rgb_total"] or c["rms_rgb_change_whole_image"] != p["rms_rgb_change_whole_image"]:
                        raise ValueError("Matched pair area/intensity mismatch")
                paired.append({"sample_index": int(index), "true_label": central[0]["true_label"], "model": family,
                    "delta": delta, "pixels_per_region": central[0]["selected_pixels"], "repeats": len(central),
                    "central_mean_probability_change": float(np.mean([r["probability_change"] for r in central])),
                    "peripheral_mean_probability_change": float(np.mean([r["probability_change"] for r in peripheral])),
                    "central_mean_abs_probability_change": float(np.mean([abs(r["probability_change"]) for r in central])),
                    "peripheral_mean_abs_probability_change": float(np.mean([abs(r["probability_change"]) for r in peripheral])),
                    "central_flip_fraction": float(np.mean([r["label_flipped"] for r in central])),
                    "peripheral_flip_fraction": float(np.mean([r["label_flipped"] for r in peripheral]))})
    csv_dicts(out / "tables/matched_region_pairs.csv", paired)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for col, family in enumerate(FAMILIES):
        for row, delta in enumerate((8, 16)):
            ax = axes[row, col]
            chosen = [r for r in paired if r["model"] == family and abs(r["delta"]) == delta]
            for c in range(8):
                same = [r for r in chosen if r["true_label"] == c]
                ax.scatter([r["peripheral_mean_abs_probability_change"] for r in same],
                           [r["central_mean_abs_probability_change"] for r in same], label=SHORT_NAMES[c], s=26, alpha=.75)
            maximum = max([r["central_mean_abs_probability_change"] for r in chosen] + [r["peripheral_mean_abs_probability_change"] for r in chosen] + [.01]) * 1.08
            ax.plot([0, maximum], [0, maximum], "k--", lw=1)
            ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Peripheral mean |probability change|", ylabel="Central mean |probability change|",
                   title=f"{family.upper()} · RGB delta ±{delta}; each point = image × sign")
            if row == 0 and col == 1: ax.legend(fontsize=7)
    fig.suptitle("4B matched controls: equal changed pixel count and RGB intensity per pair\n3 spatial repeats averaged within image/sign; above diagonal = greater central response, not anatomical attribution")
    fig.savefig(figures / "matched_region_comparison.png", dpi=140); plt.close(fig)
    whole_keys = ["whole_central_blur", "whole_peripheral_blur", "whole_central_grayscale", "whole_peripheral_grayscale"]
    region_ids = sorted(masks, key=lambda index: (int(labels[np.flatnonzero(ids == index)[0]]), index))
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES):
        matrix = np.array([[next(r["probability_change"] for r in region_rows if r["sample_index"] == index and r["model"] == family and r["key"] == key) for key in whole_keys] for index in region_ids])
        im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set(xticks=np.arange(4), xticklabels=whole_keys, yticks=np.arange(16),
               yticklabels=[f"#{index} central area={masks[index].sum()}/784" for index in region_ids], title=family.upper())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, label="Change in p(original decision)", shrink=.6)
    fig.suptitle("4B whole-region ablations: UNEQUAL area/intensity; descriptive only\nUse matched controls, not this panel alone, to compare central and peripheral responses")
    fig.savefig(figures / "whole_region_ablations.png", dpi=140); plt.close(fig)
    # Every reviewed image is shown, not just examples chosen for large effects.
    example_keys = ["whole_peripheral_grayscale", "whole_central_grayscale",
                    "matched_peripheral_+16_r0", "matched_central_+16_r0"]
    for page, start in enumerate(range(0, len(region_ids), 4), 1):
        fig, axes = plt.subplots(4, 5, figsize=(17, 14), constrained_layout=True)
        for row, index in enumerate(region_ids[start:start+4]):
            pos = int(np.flatnonzero(ids == index)[0]); key = f"regions/sample_{index:05d}"
            changed, meta, selections = region_versions(images[pos], masks[index], index, config)
            for col, condition_key in enumerate([None, *example_keys]):
                ax = axes[row, col]
                j = None if condition_key is None else next(j for j, r in enumerate(meta) if r["key"] == condition_key)
                ax.imshow(images[pos] if j is None else changed[j]); ax.set_xticks([]); ax.set_yticks([])
                if j is not None:
                    ax.contour(selections[j], levels=[.5], colors=["lime"], linewidths=.35)
                lines = []
                for family in FAMILIES:
                    p = baseline[family][pos] if j is None else loaded[key, family][j]
                    original_decision = baseline[family][pos].argmax()
                    lines.append(f"{family}: class {p.argmax()}, p(base)={p[original_decision]:.3f}")
                budget = "Original" if j is None else f"k={meta[j]['actually_changed_pixels']}; L1={meta[j]['l1_rgb_total']:.0f}"
                ax.set_title((f"#{index} true={labels[pos]}" if j is None else condition_key.replace("whole_", "").replace("matched_", "")) + "\n" + budget, fontsize=9)
                ax.set_xlabel("\n".join(lines), fontsize=8)
        fig.suptitle("4B originals and regional changes: green marks selected pixels\nWhole-region changes have unequal budgets; matched +16 controls have equal nonzero area and RGB intensity")
        fig.savefig(figures / f"region_examples_{page:02d}.png", dpi=120); plt.close(fig)
    region_summary = []
    for family in FAMILIES:
        for delta in config["matched_deltas"]:
            rows = [r for r in paired if r["model"] == family and r["delta"] == delta]
            region_summary.append({"model": family, "delta": delta, "n_images": len(rows),
                **{key: float(np.mean([r[key] for r in rows])) for key in (
                    "central_mean_abs_probability_change", "peripheral_mean_abs_probability_change", "central_flip_fraction", "peripheral_flip_fraction")}})
    csv_dicts(out / "tables/matched_region_summary.csv", region_summary)
    summary = {"protocol_signature": signature, "color_images": 48, "region_images": 16,
        "color_conditions_including_original": len(conditions), "mild_one_model_only_flip_cases": len(candidates),
        "mild_one_model_only_flip_images": len(set(r["sample_index"] for r in candidates)),
        "color_results": aggregate, "matched_region_results": region_summary,
        "intervention_evaluations": 2*(48*(len(conditions)-1)+sum(len(metadata[f"regions/sample_{index:05d}"]) for index in masks)),
        "fresh_original_evaluations": 96, "extra_repeat_audit_evaluations": 32,
        "reference": "Within-device original probabilities; all 48 gated against trusted preds exports for each model.",
        "mask_review": {"status": review["status"], "reviewer": review["reviewer"], "scope": review["scope"]},
        "limitations": ["48-image set is class-balanced and disagreement-enriched, not representative.",
            "Color perturbations are RGB manipulations, not validated stain normalization or label-preserving microscope simulations.",
            "Grayscale and strong saturation are feature ablations; whole-region interventions have unequal area/intensity.",
            "Geometric masks are approximate assistant-reviewed regions, not expert cell segmentation. Peripheral regions can contain other cells.",
            "Matched controls use dispersed eligible pixels, not contiguous whole-cell replacements. Areas/intensities are exactly matched but spatial texture/context is not.",
            "Three spatial repeats are averaged within image; they are not independent biological observations.",
            "Changes in each model's own baseline probability need not refer to the same class; true-class changes are also exported.",
            "Sensitivity/stability is not correctness, calibration, biological feature recognition, or architecture superiority."]}
    save_json(out / "summary.json", summary)
    print(f"Experiment 4 complete: {len(candidates)} mild model-specific flip cases; results in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/color_background")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "cuda:0", "cuda:1"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    bundle = setup(args)
    if args.stage == "prepare": prepare(args, bundle)
    elif args.stage == "run": run(args, bundle)
    else: report(args.output, bundle)


if __name__ == "__main__":
    main()
