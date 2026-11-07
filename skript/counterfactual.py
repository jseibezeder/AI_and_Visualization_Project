#!/usr/bin/env python3
"""Experiment 5: finite-grid, non-monotonic geometric decision-boundary searches.

python skript/counterfactual.py prepare
python skript/counterfactual.py run --device cuda:0
python skript/counterfactual.py report

Search both signs; scan the ENTIRE range before refining every detected class
transition. A minimum is an observed grid minimum, never a continuous/global
adversarial-distance claim. No training, downloads, or changes to prior results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from analysis import ROOT, SHORT_NAMES, load_predictions, write_csv
from color_background import atomic_npz, csv_dicts, intervention_stats, validate_prediction
from occlusion import FAMILIES, digest, file_digest, plotting, save_json

STAGES = ("coarse", "scan", "refine")
SOURCE_FILES = (
    "results/occlusion/subset/manifest.json",
    "results/analysis/test/tables/all_samples.csv",
    "results/stability/subset/manifest.json",
    "results/stability/tables/per_image_metrics.csv",
    "results/occlusion/tables/map_summary.csv",
    "results/color_background/tables/color_per_image.csv",
)
DEFAULT_AXES = {
    "rotation": {"unit": .025, "max_tick": 1200, "coarse_stride": 200, "scan_stride": 10, "unit_name": "degrees"},
    "translation": {"unit": .0125, "max_tick": 240, "coarse_stride": 80, "scan_stride": 10, "unit_name": "horizontal pixels"},
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_examples():
    """Eight unique images selected exclusively using pre-existing evidence.

    All IDs remain inside the exact 48-image explanation set. A selection quota
    uses one E1 disagreement and one same-decision distribution difference,
    one E2 example per geometry family, two E3 true-class occlusion differences,
    and two E4 mild channel-specific flips from initially both-correct images.
    """
    source = json.loads((ROOT / SOURCE_FILES[0]).read_text())
    candidates = set(source["sample_indices"])
    selected = []

    def take(ranked, count, experiment, criterion):
        for score, index, evidence in sorted(ranked, key=lambda x: (-x[0], x[1])):
            if index in {r["sample_index"] for r in selected}:
                continue
            selected.append({"sample_index": index, "experiment": experiment,
                             "selection_criterion": criterion, "score": float(score), "evidence": evidence})
            count -= 1
            if not count:
                break
        if count:
            raise ValueError(f"Not enough distinct prior examples for {criterion}")

    analysis = [r for r in read_csv(ROOT / SOURCE_FILES[1]) if int(r["sample_index"]) in candidates]
    for same in (False, True):
        take([(float(r["js_divergence_bits"]), int(r["sample_index"]),
               f"JS={r['js_divergence_bits']} bits; group={r['group']}; ResNet={r['resnet_prediction']}; ViT={r['vit_prediction']}")
              for r in analysis if (r["same_prediction"] == "True") == same], 1, "1",
             "Largest probability-vector JS among " + ("same-class predictions" if same else "different-class predictions"))
    stability = [r for r in read_csv(ROOT / SOURCE_FILES[3]) if int(r["sample_index"]) in candidates
                 and r["transformation"] in DEFAULT_AXES]
    for axis in DEFAULT_AXES:
        paired = {}
        for r in stability:
            if r["transformation"] == axis:
                paired.setdefault((int(r["sample_index"]), float(r["level"])), {})[r["model"]] = r
        ranked = {}
        for (index, level), by_model in paired.items():
            if set(by_model) != set(FAMILIES):
                raise ValueError("Incomplete existing stability comparison")
            values = [float(by_model[f]["retained_baseline_probability"]) for f in FAMILIES]
            score = abs(values[0]-values[1])
            evidence = f"{axis}={level:g}; retained p(original): ResNet={values[0]:.8g}, ViT={values[1]:.8g}"
            if index not in ranked or score > ranked[index][0]:
                ranked[index] = score, index, evidence
        take(list(ranked.values()), 1, "2", f"Largest inter-model retained-probability gap under prior {axis}, within E2/E3 overlap")
    maps = [r for r in read_csv(ROOT / SOURCE_FILES[4]) if r["target_type"] == "true_class"]
    paired = {}
    for r in maps:
        paired.setdefault(int(r["sample_index"]), {})[r["model"]] = r
    ranked = []
    for index, by_model in paired.items():
        peaks = [max(abs(float(by_model[f]["min_importance"])), abs(float(by_model[f]["max_importance"]))) for f in FAMILIES]
        ranked.append((abs(peaks[0]-peaks[1]), index,
                       f"True-class occlusion peak |response|: ResNet={peaks[0]:.8g}, ViT={peaks[1]:.8g}"))
    take(ranked, 2, "3", "Largest inter-model difference in peak absolute true-class occlusion response")
    color = [r for r in read_csv(ROOT / SOURCE_FILES[5]) if r["transformation"].startswith(("red_scale", "blue_scale"))]
    paired = {}
    for r in color:
        paired.setdefault((int(r["sample_index"]), r["transformation"]), {})[r["model"]] = r
    ranked = {}
    for (index, condition), by_model in paired.items():
        a, b = [by_model[f] for f in FAMILIES]
        if (a["baseline_prediction"] != a["true_label"] or b["baseline_prediction"] != b["true_label"]
                or a["label_flipped"] == b["label_flipped"]):
            continue
        score = abs(float(a["probability_change"])-float(b["probability_change"]))
        evidence = f"{condition}; both initially correct; flips ResNet={a['label_flipped']}, ViT={b['label_flipped']}"
        if index not in ranked or score > ranked[index][0]:
            ranked[index] = score, index, evidence
    take(list(ranked.values()), 2, "4", "Largest mild red/blue one-model-only response difference, initially both correct")
    if len(selected) != 8 or not {r["sample_index"] for r in selected} <= candidates:
        raise ValueError("Selection must contain eight distinct prior explanation images")
    return sorted(selected, key=lambda r: r["sample_index"])


def render(image, axis, ticks, spec, fill):
    """Shared uint8 interventions; x translation positive = content moves right.

    Pixel-centered inverse affine mapping with bilinear interpolation; integer
    shifts and rotations reproduce Experiment 2 exactly. No wrap or expansion.
    """
    if image.dtype != np.uint8 or image.shape != (28, 28, 3):
        raise ValueError("Expected original uint8 28x28 RGB image")
    original = Image.fromarray(image)
    result = []
    for tick in ticks:
        value = round(int(tick)*spec["unit"], 8)
        if tick == 0:
            changed = image.copy()
        elif axis == "rotation":
            changed = np.asarray(original.rotate(value, resample=Image.Resampling.BILINEAR,
                                                  expand=False, fillcolor=tuple(fill)))
        elif axis == "translation":
            changed = np.asarray(original.transform((28, 28), Image.Transform.AFFINE,
                (1, 0, -value, 0, 1, 0), resample=Image.Resampling.BILINEAR, fillcolor=tuple(fill)))
        else:
            raise ValueError(axis)
        result.append(changed)
    return np.stack(result) if result else np.empty((0, 28, 28, 3), dtype=np.uint8)


def grid_ticks(spec, stage):
    coarse = np.arange(-spec["max_tick"], spec["max_tick"]+1, spec["coarse_stride"], dtype=np.int64)
    if stage == "coarse":
        return coarse
    scan = np.arange(-spec["max_tick"], spec["max_tick"]+1, spec["scan_stride"], dtype=np.int64)
    return np.setdiff1d(scan, coarse)


def refinement_ticks(ticks, predictions):
    """Union of every differing-neighbor bracket for either model.

    The global scan has already tested equal-endpoint coarse cells internally.
    Equal-endpoint *scan* cells can still hide narrower islands; no continuous
    minimum/no-flip certificate is claimed.
    """
    order = np.argsort(ticks); ticks = np.asarray(ticks)[order]
    if len(np.unique(ticks)) != len(ticks):
        raise ValueError("Duplicate tested ticks")
    changed = np.zeros(len(ticks)-1, bool)
    for values in predictions:
        classes = np.asarray(values)[order]
        changed |= classes[:-1] != classes[1:]
    chunks = [np.arange(a+1, b, dtype=np.int64) for a, b, change in zip(ticks[:-1], ticks[1:], changed) if change and b-a > 1]
    return np.unique(np.concatenate(chunks)) if chunks else np.array([], dtype=np.int64)


def ray_summary(ticks, probabilities, baseline_class, spec, sign):
    ticks = np.asarray(ticks)
    chosen = np.flatnonzero(ticks*sign >= 0)
    chosen = chosen[np.argsort(np.abs(ticks[chosen]))]
    ordered = ticks[chosen]; classes = probabilities[chosen].argmax(1)
    if len(ordered) == 0 or ordered[0] != 0 or classes[0] != baseline_class:
        raise ValueError("Ray must start at the original decision")
    first = np.flatnonzero(classes != baseline_class)
    transitions = []
    for j in range(1, len(ordered)):
        if classes[j] != classes[j-1]:
            transitions.append({"previous_tick": int(ordered[j-1]), "switch_tick": int(ordered[j]),
                "previous_value": round(float(ordered[j-1])*spec["unit"], 8),
                "switch_value": round(float(ordered[j])*spec["unit"], 8),
                "bracket_width": round(float(abs(ordered[j]-ordered[j-1]))*spec["unit"], 8),
                "from_class": int(classes[j-1]), "to_class": int(classes[j]),
                "returns_to_baseline": bool(classes[j] == baseline_class)})
    return {"first_switch_tick": int(ordered[first[0]]) if len(first) else None,
            "previous_same_tick": int(ordered[first[0]-1]) if len(first) else None,
            "transitions": transitions, "return_count": sum(r["returns_to_baseline"] for r in transitions)}


def setup(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if not path.exists():
        save_json(path, {"schema_version": 1, "seed": 45, "axes": DEFAULT_AXES,
            "fill_rgb": [128]*3, "baseline_atol": 5e-4, "baseline_rtol": 1e-4,
            "scope": "Coarse grid, complete global scan, then all observed top-1 brackets refined to one tick. Narrower unobserved islands remain possible."})
    config = json.loads(path.read_text())
    if config["schema_version"] != 1 or set(config["axes"]) != set(DEFAULT_AXES):
        raise ValueError("Unsupported schema or axes")
    for spec in config["axes"].values():
        if (spec["unit"] <= 0 or not np.isfinite(spec["unit"]) or
            any(type(spec[k]) is not int or spec[k] <= 0 for k in ("max_tick", "coarse_stride", "scan_stride")) or
            spec["max_tick"] % spec["coarse_stride"] or spec["coarse_stride"] % spec["scan_stride"]):
            raise ValueError("Axes need positive units and nested integer grids containing zero/endpoints")
    if len(config["fill_rgb"]) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in config["fill_rgb"]):
        raise ValueError("Invalid RGB fill")
    selection = select_examples()
    ids = np.array([r["sample_index"] for r in selection], dtype=np.int64)
    with np.load(ROOT / "data/bloodmnist.npz", allow_pickle=False) as z:
        images = z["test_images"][ids]; labels = z["test_labels"][ids].reshape(-1).astype(np.int64)
    names_dict = json.loads((ROOT / "data/metadata.json").read_text())["label"]
    names = [names_dict[str(i)] for i in range(8)]
    baseline = {}
    for family in FAMILIES:
        p = load_predictions(ROOT / f"preds/{family}/test.npz", names)
        lookup = {int(x): j for j, x in enumerate(p["sample_index"])}
        rows = [lookup[int(i)] for i in ids]
        np.testing.assert_array_equal(p["labels"][rows], labels)
        baseline[family] = p["probabilities"][rows]
    manifest = {"split": "test", "sample_indices": ids.tolist(), "true_labels": labels.tolist(),
                "class_names": names, "images_sha256": digest(images), "selection": selection,
                "selection_bias": "Eight deliberately informative examples from the class-balanced, disagreement-enriched 48-image set; no E5 outcomes used for selection."}
    manifest_path = out / "subset/manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Persisted selection changed; choose a new output directory")
    save_json(manifest_path, manifest)
    csv_dicts(out / "subset/selection.csv", selection)
    dependencies = [*SOURCE_FILES, "skript/counterfactual.py", "skript/inference.py", "skript/analysis.py",
                    "skript/occlusion.py", "skript/color_background.py", "models/resnet18/resnet18_28_1.pth",
                    "models/resnet18/models.py", "models/vit/model.safetensors", "models/vit/config.json",
                    "models/vit/preprocessor_config.json"]
    for axis, levels in (("rotation", (-20, -10, -5, 5, 10, 20)), ("translation", (-3, -2, -1, 1, 2, 3))):
        dependencies.extend(f"results/stability/predictions/{family}/{axis}_{level:g}_r0.npz"
                            for family in FAMILIES for level in levels)
    frozen = {"config": config, "manifest": manifest,
              "input_sha256": {rel: file_digest(ROOT / rel) for rel in dependencies},
              "baseline_sha256": {f: digest(p) for f, p in baseline.items()}}
    signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["signature"] != signature:
        raise ValueError("Frozen inputs/code/protocol changed; preserve results and choose a new --output")
    return config, signature, frozen, ids, images, labels, baseline, names


def input_job(out, signature, axis, index, stage, ticks, image, spec, fill):
    key = f"{axis}/sample_{index:05d}/{stage}"
    changed = render(image, axis, ticks, spec, fill)
    image_hash = digest(changed)
    path = out / "interventions" / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["protocol_signature"]) != signature or str(z["input_sha256"]) != image_hash:
                raise ValueError(f"Stale/corrupt intervention {path}")
            np.testing.assert_array_equal(z["ticks"], ticks)
            np.testing.assert_array_equal(z["images"], changed)
    else:
        atomic_npz(path, images=changed, ticks=ticks, input_sha256=image_hash, protocol_signature=signature)
    return key, changed, image_hash


def read_job(out, signature, family, key, index, ticks, input_hash, environment=None):
    path = out / f"predictions/{family}/{key}.npz"
    p = validate_prediction(path, signature, np.full(len(ticks), index, dtype=np.int64), input_hash, environment)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ticks"], ticks)
        return p, z["logits"].copy()


def collect(out, bundle, required_stages=("coarse", "scan"), include_refine=False):
    """Validate shared arrays and both models before any analysis/refinement."""
    config, signature, frozen, ids, images, labels, baseline, names = bundle
    collected = {}
    for pos, index in enumerate(ids):
        for axis, spec in config["axes"].items():
            ticks_all, probabilities = [], {f: [] for f in FAMILIES}
            for stage in (*required_stages, *(("refine",) if include_refine else ())):
                input_path = out / f"interventions/{axis}/sample_{index:05d}/{stage}.npz"
                if stage == "refine" and not input_path.exists():
                    continue
                with np.load(input_path, allow_pickle=False) as z:
                    ticks = z["ticks"]
                if stage != "refine":
                    np.testing.assert_array_equal(ticks, grid_ticks(spec, stage))
                key, changed, input_hash = input_job(out, signature, axis, int(index), stage, ticks, images[pos], spec, config["fill_rgb"])
                ticks_all.append(ticks)
                for family in FAMILIES:
                    audit = json.loads((out / f"predictions/{family}/reproducibility.json").read_text())
                    if audit["protocol_signature"] != signature:
                        raise ValueError("Stale reproducibility metadata")
                    p, _ = read_job(out, signature, family, key, int(index), ticks, input_hash, audit["environment_signature"])
                    probabilities[family].append(p)
            ticks = np.concatenate(ticks_all)
            if len(ticks) != len(np.unique(ticks)):
                raise ValueError("Repeated tick across stages")
            order = np.argsort(ticks)
            collected[axis, int(index)] = (ticks[order], {f: np.concatenate(a)[order] for f, a in probabilities.items()})
    return collected


def check_prior_geometry(out, bundle):
    """Equivalence to trusted E2 outputs where selected images overlap E2."""
    config, signature, frozen, ids, images, labels, baseline, names = bundle
    current = collect(out, bundle, required_stages=("coarse",))
    source = json.loads((ROOT / "results/stability/subset/manifest.json").read_text())
    lookup = {int(i): j for j, i in enumerate(source["sample_indices"])}
    records = []
    for axis in config["axes"]:
        levels = (-20, -10, -5, 5, 10, 20) if axis == "rotation" else (-3, -2, -1, 1, 2, 3)
        for level in levels:
            for family in FAMILIES:
                path = ROOT / f"results/stability/predictions/{family}/{axis}_{level:g}_r0.npz"
                with np.load(path, allow_pickle=False) as z:
                    p = z["probabilities"]
                    np.testing.assert_array_equal(z["sample_index"], source["sample_indices"])
                for pos, index in enumerate(ids):
                    if int(index) not in lookup:
                        continue
                    ticks, predicted = current[axis, int(index)]
                    tick = round(level/config["axes"][axis]["unit"])
                    where = np.flatnonzero(ticks == tick)
                    if not len(where):
                        continue
                    a, b = predicted[family][where[0]], p[lookup[int(index)]]
                    np.testing.assert_allclose(a, b, atol=config["baseline_atol"], rtol=config["baseline_rtol"])
                    if a.argmax() != b.argmax():
                        raise ValueError("Prior geometry decision mismatch")
                    records.append({"sample_index": int(index), "axis": axis, "value": level,
                                    "model": family, "max_probability_error": float(np.abs(a-b).max())})
    csv_dicts(out / "validation/prior_geometry_equivalence.csv", records)
    return records


def run(args, bundle):
    from inference import LocalPredictor
    config, signature, frozen, ids, images, labels, trusted, names = bundle
    save_json(args.output / "protocol.json", dict(signature=signature, **frozen))
    stage_counts = {}
    for stage in STAGES:
        if stage == "refine":
            scanned = collect(args.output, bundle)
        jobs = []
        for pos, index in enumerate(ids):
            for axis, spec in config["axes"].items():
                if stage == "refine":
                    tested, p = scanned[axis, int(index)]
                    ticks = refinement_ticks(tested, [p[f].argmax(1) for f in FAMILIES])
                else:
                    ticks = grid_ticks(spec, stage)
                if not len(ticks):
                    continue
                key, changed, h = input_job(args.output, signature, axis, int(index), stage, ticks, images[pos], spec, config["fill_rgb"])
                jobs.append((int(index), ticks, key, changed, h))
        stage_counts[stage] = sum(len(t) for _, t, _, _, _ in jobs)
        for family in FAMILIES:
            folder = args.output / f"predictions/{family}"
            audit_path = folder / "reproducibility.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
            if audit is not None and audit["protocol_signature"] != signature:
                raise ValueError("Stale audit")
            pending = []
            for job in jobs:
                index, ticks, key, changed, h = job
                if (folder / f"{key}.npz").exists():
                    if audit is None:
                        raise ValueError("Predictions exist without an environment audit")
                    read_job(args.output, signature, family, key, index, ticks, h, audit["environment_signature"])
                else:
                    pending.append(job)
            if not pending and audit is not None:
                print(f"{stage} {family}: all caches valid; no inference", flush=True)
                continue
            predictor = LocalPredictor(family, names, args.device, args.batch_size, config["seed"])
            try:
                import PIL
                import torch
                import transformers
                env = dict(predictor.info, numpy_version=np.__version__, pillow_version=PIL.__version__,
                           transformers_version=transformers.__version__, batch_size=args.batch_size,
                           device_name=torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else str(predictor.device))
                environment = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
                if audit is not None and audit["environment_signature"] != environment:
                    raise ValueError("Cannot mix hardware/software/batch environments during resume")
                started = time.perf_counter()
                # Exact raw-input memoization across axes AND stages avoids repeated
                # model calls for byte-identical rasterizations. Both logits and
                # probabilities are reused; the finite tick grid is unchanged.
                memo = {}
                if audit is None:
                    logits, base = predictor.predict(images)
                    np.testing.assert_allclose(base, trusted[family], atol=config["baseline_atol"], rtol=config["baseline_rtol"])
                    np.testing.assert_array_equal(base.argmax(1), trusted[family].argmax(1))
                    repeats = np.stack([base, *[predictor.predict(images)[1] for _ in range(2)]])
                    np.testing.assert_allclose(repeats, np.broadcast_to(base, repeats.shape), atol=2e-5, rtol=1e-4)
                    for image, logit, p in zip(images, logits, base):
                        memo[image.tobytes()] = (logit, p)
                    audit = dict(env, protocol_signature=signature, environment_signature=environment,
                        baseline_max_probability_error=float(np.abs(base-trusted[family]).max()),
                        repeat_max_probability_range=float(np.ptp(repeats, axis=0).max()),
                        initial_original_evaluations=8, extra_repeat_evaluations=16, stages={})
                    save_json(audit_path, audit)
                # Validated completed jobs provide the cross-stage memo.
                for path in sorted(folder.rglob("*.npz")):
                    key = path.relative_to(folder).with_suffix("").as_posix()
                    input_path = args.output / f"interventions/{key}.npz"
                    with np.load(input_path, allow_pickle=False) as z:
                        cached_ticks, cached_images = z["ticks"], z["images"]
                        if str(z["protocol_signature"]) != signature or digest(cached_images) != str(z["input_sha256"]):
                            raise ValueError("Invalid memo input")
                    index = int(key.split("/")[1].removeprefix("sample_"))
                    p, logits = read_job(args.output, signature, family, key, index, cached_ticks, digest(cached_images), environment)
                    for image, logit, probability in zip(cached_images, logits, p):
                        memo[image.tobytes()] = logit, probability
                unique = {}
                for _, ticks, key, changed, _ in pending:
                    for image in changed:
                        raw = image.tobytes()
                        if raw not in memo:
                            unique.setdefault(raw, image)
                if unique:
                    raw_keys = list(unique)
                    logits, p = predictor.predict(np.stack(list(unique.values())))
                    for raw, logit, probability in zip(raw_keys, logits, p):
                        memo[raw] = logit, probability
                for index, ticks, key, changed, h in pending:
                    values = [memo[image.tobytes()] for image in changed]
                    atomic_npz(folder / f"{key}.npz", logits=np.stack([v[0] for v in values]),
                        probabilities=np.stack([v[1] for v in values]), sample_index=np.full(len(ticks), index, dtype=np.int64),
                        ticks=ticks, protocol_signature=signature, input_sha256=h, environment_signature=environment)
                    read_job(args.output, signature, family, key, index, ticks, h, environment)
                old = audit["stages"].get(stage, {"unique_inferred_inputs": 0, "elapsed_seconds": 0.})
                audit["stages"][stage] = {"unique_inferred_inputs": old["unique_inferred_inputs"]+len(unique),
                                          "elapsed_seconds": old["elapsed_seconds"]+time.perf_counter()-started}
                audit["warnings"] = sorted(set(audit.get("warnings", [])) | predictor.warning_messages)
                save_json(audit_path, audit)
                print(f"{stage} {family}: {sum(len(j[1]) for j in pending)} tick evaluations, {len(unique)} unique new RGB inputs", flush=True)
            finally:
                predictor.close()
        if stage == "coarse":
            equivalent = check_prior_geometry(args.output, bundle)
            print(f"Prior Experiment 2 equivalence: {len(equivalent)} matching image/model/geometry checks", flush=True)
    save_json(args.output / "validation/stage_counts.json", stage_counts)
    report(args.output, bundle)


def plot_curves(out, axis, index, ticks, probabilities, base_classes, spec):
    plt = plotting()
    values = ticks*spec["unit"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), height_ratios=[4, 1], constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for col, family in enumerate(FAMILIES):
        for cls in range(8):
            axes[0, col].plot(values, probabilities[family][:, cls], color=cmap(cls),
                lw=2.2 if cls == base_classes[family] else .85, alpha=1 if cls == base_classes[family] else .75,
                label=f"{cls}: {SHORT_NAMES[cls]}")
        axes[0, col].axvline(0, color="black", linestyle=":", lw=1)
        axes[0, col].set(title=f"{family.upper()} · original class {base_classes[family]}", ylabel="Class probability", ylim=(-.02, 1.02))
        classes = probabilities[family].argmax(1)
        axes[1, col].scatter(values, np.zeros(len(values)), c=[cmap(int(c)) for c in classes], marker="|", s=180)
        axes[1, col].set(xlabel=f"Signed {axis} ({spec['unit_name']})", yticks=[], title="Top-1 class at every tested value", ylim=(-.2, .2))
        axes[0, col].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Experiment 5 · #{index} · {axis}\nAll 8 probabilities; lines only connect tested points. Complete global scan plus all observed transition brackets refined.")
    path = out / f"figures/probability_paths/{axis}_sample_{index:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def plot_counterfactual(out, axis, index, image, true_label, ticks, probabilities, base_classes, spec, fill):
    plt = plotting()
    choices = [0]
    titles = [f"Original #{index} · true={true_label}"]
    for family in FAMILIES:
        switched = ticks[probabilities[family].argmax(1) != base_classes[family]]
        if len(switched):
            tick = int(sorted(switched.tolist(), key=lambda t: (abs(t), t))[0])
            choices.append(tick)
            titles.append(f"Nearest observed {family.upper()} flip\n{tick*spec['unit']:+.4g} {spec['unit_name']}")
        else:
            choices.append(0)
            titles.append(f"{family.upper()}: no flip observed\nin tested ±{spec['max_tick']*spec['unit']:g} {spec['unit_name']}")
    modified = render(image, axis, choices, spec, fill)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    x = np.arange(8)
    for col, (tick, changed, title) in enumerate(zip(choices, modified, titles)):
        axes[0, col].imshow(changed); axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        lines = []
        where = int(np.flatnonzero(ticks == tick)[0])
        for j, family in enumerate(FAMILIES):
            p = probabilities[family][where]; pred = int(p.argmax())
            axes[1, col].bar(x+(-.19 if j == 0 else .19), p, .38, label=family.upper())
            lines.append(f"{family}: class {pred}, p(pred)={p[pred]:.3f}, p(original)={p[base_classes[family]]:.3f}")
        stats = intervention_stats(image, changed, np.ones((28, 28), bool))
        axes[0, col].set_title(title + f"\nChanged pixels={stats['actually_changed_pixels']}; RGB RMS={stats['rms_rgb_change_whole_image']:.2f}", fontsize=10)
        axes[0, col].set_xlabel("\n".join(lines), fontsize=9)
        axes[1, col].set(xticks=x, xticklabels=[f"{i}: {n}" for i, n in enumerate(SHORT_NAMES)], ylabel="Probability", ylim=(0, 1.03))
        plt.setp(axes[1, col].get_xticklabels(), rotation=55, ha="right", fontsize=8)
        axes[1, col].legend()
    fig.suptitle(f"5 · {axis} counterfactuals: each candidate is evaluated by BOTH models\nMinimum over tested signed values only; interpolation/uint8 rasterization is part of the intervention.", fontsize=13)
    path = out / f"figures/counterfactuals/{axis}_sample_{index:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def report(out, bundle, render_figures=True):
    config, signature, frozen, ids, images, labels, trusted, names = bundle
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["signature"] != signature:
        raise ValueError("Protocol mismatch")
    data = collect(out, bundle, include_refine=True)
    # Verify the complete global guard grid and the exact union-refinement policy.
    for axis, spec in config["axes"].items():
        scan_grid = np.union1d(grid_ticks(spec, "coarse"), grid_ticks(spec, "scan"))
        for index in ids:
            ticks, p = data[axis, int(index)]
            if not set(scan_grid.tolist()) <= set(ticks.tolist()):
                raise ValueError("Global scan is incomplete")
            pos = np.searchsorted(ticks, scan_grid)
            expected = refinement_ticks(scan_grid, [p[f][pos].argmax(1) for f in FAMILIES])
            np.testing.assert_array_equal(ticks, np.union1d(scan_grid, expected))
    curve_rows, transition_rows, boundary_rows, nearest_rows = [], [], [], []
    minimums = {}
    for pos, index in enumerate(ids):
        for axis, spec in config["axes"].items():
            ticks, probabilities = data[axis, int(index)]
            zero = int(np.flatnonzero(ticks == 0)[0])
            base_classes = {}
            for family in FAMILIES:
                base = probabilities[family][zero]
                np.testing.assert_allclose(base, trusted[family][pos], atol=config["baseline_atol"], rtol=config["baseline_rtol"])
                base_class = int(base.argmax()); base_classes[family] = base_class
                if base_class != trusted[family][pos].argmax():
                    raise ValueError("Original decision mismatch")
                # This same original must also agree exactly across both axes.
                other_axis = next(a for a in config["axes"] if a != axis)
                other_ticks, other_probs = data[other_axis, int(index)]
                np.testing.assert_array_equal(base, other_probs[family][np.flatnonzero(other_ticks == 0)[0]])
                changed = render(images[pos], axis, ticks, spec, config["fill_rgb"])
                for tick, image, p in zip(ticks, changed, probabilities[family]):
                    stats = intervention_stats(images[pos], image, np.ones((28, 28), bool))
                    curve_rows.append({"sample_index": int(index), "true_label": int(labels[pos]), "axis": axis,
                        "tick": int(tick), "value": round(float(tick)*spec["unit"], 8), "model": family,
                        "baseline_prediction": base_class, "prediction": int(p.argmax()),
                        "label_flipped": bool(p.argmax() != base_class),
                        "baseline_class_probability": float(p[base_class]), "true_class_probability": float(p[labels[pos]]),
                        "probabilities": json.dumps([float(v) for v in p]), **stats})
                for sign in (-1, 1):
                    ray = ray_summary(ticks, probabilities[family], base_class, spec, sign)
                    first = ray["first_switch_tick"]
                    boundary_rows.append({"sample_index": int(index), "true_label": int(labels[pos]), "axis": axis,
                        "model": family, "direction": sign, "baseline_prediction": base_class,
                        "observed_switch": first is not None,
                        "nearest_switch_tick": first,
                        "nearest_switch_value": None if first is None else round(first*spec["unit"], 8),
                        "previous_tested_same_value": None if first is None else round(ray["previous_same_tick"]*spec["unit"], 8),
                        "bracket_width": None if first is None else round(abs(first-ray["previous_same_tick"])*spec["unit"], 8),
                        "tested_limit": spec["max_tick"]*spec["unit"], "final_refinement_unit": spec["unit"],
                        "global_scan_spacing": spec["scan_stride"]*spec["unit"],
                        "observed_class_transitions": len(ray["transitions"]), "observed_returns_to_baseline": ray["return_count"],
                        "interpretation": "Nearest observed flip on this signed finite grid; no continuous minimum or no-switch certificate."})
                    for transition in ray["transitions"]:
                        transition_rows.append(dict(sample_index=int(index), axis=axis, model=family, direction=sign,
                                                    baseline_prediction=base_class, **transition))
                switched = ticks[probabilities[family].argmax(1) != base_class]
                tick = None if not len(switched) else sorted(switched.tolist(), key=lambda t: (abs(t), t))[0]
                minimums[axis, int(index), family] = None if tick is None else abs(tick)*spec["unit"]
                nearest_rows.append({"sample_index": int(index), "true_label": int(labels[pos]), "axis": axis, "model": family,
                    "baseline_prediction": base_class, "nearest_tick": tick,
                    "signed_value": None if tick is None else round(tick*spec["unit"], 8),
                    "absolute_value": None if tick is None else round(abs(tick)*spec["unit"], 8),
                    "switched_class": None if tick is None else int(probabilities[family][np.flatnonzero(ticks == tick)[0]].argmax()),
                    "no_observed_switch_is_censored": tick is None})
            if render_figures:
                plot_curves(out, axis, int(index), ticks, probabilities, base_classes, spec)
                plot_counterfactual(out, axis, int(index), images[pos], int(labels[pos]), ticks, probabilities, base_classes, spec, config["fill_rgb"])
    csv_dicts(out / "tables/probability_paths.csv", curve_rows)
    csv_dicts(out / "tables/directional_boundaries.csv", boundary_rows)
    csv_dicts(out / "tables/nearest_observed_switches.csv", nearest_rows)
    fields = ["sample_index", "axis", "model", "direction", "baseline_prediction", "previous_tick", "switch_tick",
              "previous_value", "switch_value", "bracket_width", "from_class", "to_class", "returns_to_baseline"]
    write_csv(out / "tables/all_observed_transitions.csv", fields, ([r[k] for k in fields] for r in transition_rows))
    if render_figures:
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        for ax, (axis, spec) in zip(axes, config["axes"].items()):
            for j, index in enumerate(ids):
                values = [minimums[axis, int(index), family] for family in FAMILIES]
                if all(v is not None for v in values): ax.plot(values, [j, j], color="gray", lw=1)
                for value, family, color, marker in zip(values, FAMILIES, ("#2678ad", "#c25a7c"), ("o", "s")):
                    if value is None:
                        ax.annotate(f"{family}: no observed flip", (spec["max_tick"]*spec["unit"], j), fontsize=7, ha="right")
                    else:
                        ax.scatter(value, j, c=color, marker=marker, label=family.upper() if j == 0 else None)
            ax.set(yticks=np.arange(8), yticklabels=[f"#{i}" for i in ids], xlabel=f"Smallest observed |{axis}| ({spec['unit_name']})",
                   title=f"{axis.title()}: min across both signs", xlim=(-.02, spec["max_tick"]*spec["unit"]*1.04))
            ax.legend(); ax.grid(axis="x", alpha=.2)
        fig.suptitle("5 · Finite-grid counterfactuals, eight deliberately selected images\nDifferent signs may produce each model's minimum; not comparable across transformation families or globally minimal perturbations")
        fig.savefig(out / "figures/nearest_switch_comparison.png", dpi=140); plt.close(fig)
    audits = {f: json.loads((out / f"predictions/{f}/reproducibility.json").read_text()) for f in FAMILIES}
    summary = {"protocol_signature": signature, "n_images": len(ids), "sample_indices": ids.tolist(),
        "axes": config["axes"], "evaluated_image_axis_ticks_per_model": sum(len(t) for t, _ in data.values()),
        "observed_transitions": len(transition_rows), "observed_returns_to_baseline": sum(r["returns_to_baseline"] for r in transition_rows),
        "directional_searches": len(boundary_rows), "directional_no_observed_switches": sum(not r["observed_switch"] for r in boundary_rows),
        "nearest_observed_switches": nearest_rows,
        "unique_inference_inputs_excluding_originals_and_repeats": {f: sum(s["unique_inferred_inputs"] for s in a["stages"].values()) for f, a in audits.items()},
        "original_evaluations": 16, "extra_repeat_evaluations": 32,
        "limitations": ["Finite global guard grid can miss narrower same-endpoint class islands; refinements cover every observed transition, not all continuous points.",
            "Nearest observed flips are specific to the signed axis, tested range, rasterization, interpolation, gray fill and model preprocessing.",
            "No observed switch is range/grid censoring, not a robustness certificate.",
            "The eight images were deliberately selected from prior experiments, not representative or an independent benchmark.",
            "Changes include interpolation and border-fill effects and may not preserve diagnostic content.",
            "A switch may fix an originally wrong prediction. Stability is not correctness.",
            "Rotation degrees, translation pixels and mathematical perturbation norms are different notions; no global minimum is established."]}
    save_json(out / "summary.json", summary)
    print(f"Experiment 5 complete: {len(ids)} images, {len(transition_rows)} observed transitions, {summary['observed_returns_to_baseline']} observed returns.", flush=True)


def prepare(args, bundle):
    config, signature, frozen, ids, images, labels, baseline, names = bundle
    plt = plotting()
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for pos, (ax, index) in enumerate(zip(axes.flat, ids)):
        ax.imshow(images[pos]); ax.set_xticks([]); ax.set_yticks([])
        selection = frozen["manifest"]["selection"][pos]
        ax.set_title(f"#{index} · true={labels[pos]} · selected from E{selection['experiment']}\n"
                     f"ResNet={baseline['resnet'][pos].argmax()} · ViT={baseline['vit'][pos].argmax()}", fontsize=10)
    fig.suptitle("Experiment 5: selection fixed using Experiments 1–4, before new geometric inference\nAll eight images belong to the existing 48-image explanation set; not a representative sample")
    path = args.output / "pilot/selected_images.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    for r in frozen["manifest"]["selection"]:
        print(f"#{r['sample_index']} E{r['experiment']}: {r['selection_criterion']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/counterfactual")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "cuda:0", "cuda:1"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("Batch size must be positive")
    bundle = setup(args)
    if args.stage == "prepare": prepare(args, bundle)
    elif args.stage == "run": run(args, bundle)
    else: report(args.output, bundle)


if __name__ == "__main__":
    main()
