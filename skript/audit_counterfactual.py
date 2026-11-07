#!/usr/bin/env python3
"""Repeat all observed transition endpoints; generate a selected local zoom.

python skript/audit_counterfactual.py --device cuda:0 --zoom-index 2878

This is an explicit small validation pass, not a rerun of the search. The zoom
is a post-search illustration; it does not change the original eight-image set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import counterfactual as e
from inference import LocalPredictor


def zoom(out, bundle, data, index):
    config, signature, frozen, ids, images, labels, trusted, names = bundle
    pos = int(np.flatnonzero(ids == index)[0])
    axis = "rotation"; spec = config["axes"][axis]
    ticks, p = data[axis, index]
    zero = int(np.flatnonzero(ticks == 0)[0])
    base = {f: int(p[f][zero].argmax()) for f in e.FAMILIES}
    ray = e.ray_summary(ticks, p["resnet"], base["resnet"], spec, 1)
    changes = ray["transitions"]
    if len(changes) < 3 or not changes[1]["returns_to_baseline"]:
        raise ValueError("Requested zoom needs an initial ResNet switch, return and subsequent switch")
    chosen_ticks = [0, *[r["switch_tick"] for r in changes[:3]]]
    selected_images = e.render(images[pos], axis, chosen_ticks, spec, config["fill_rgb"])
    plt = e.plotting()
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = fig.add_gridspec(2, 4)
    for j, (tick, image) in enumerate(zip(chosen_ticks, selected_images)):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
        where = int(np.flatnonzero(ticks == tick)[0])
        title = "Original" if tick == 0 else f"Rotation {tick*spec['unit']:+.4g} degrees"
        lines = []
        for family in e.FAMILIES:
            prob = p[family][where]
            lines.append(f"{family}: class {prob.argmax()}, p(original)={prob[base[family]]:.5f}")
        ax.set_title(title, fontsize=11); ax.set_xlabel("\n".join(lines), fontsize=9)
    lo = changes[0]["switch_value"]-.3; hi = max(r["switch_value"] for r in changes[:3])+.7
    chosen = (ticks*spec["unit"] >= lo) & (ticks*spec["unit"] <= hi)
    for col, family in enumerate(e.FAMILIES):
        ax = fig.add_subplot(gs[1, col*2:(col+1)*2])
        for cls in range(8):
            ax.plot(ticks[chosen]*spec["unit"], p[family][chosen, cls], "o-", ms=2.5,
                    lw=1.7 if cls in (base["resnet"], changes[0]["to_class"]) else .6,
                    label=f"{cls}: {e.SHORT_NAMES[cls]}")
        for r in changes[:3]: ax.axvline(r["switch_value"], color="black", alpha=.4, linestyle=":")
        ax.set(xlabel="Positive rotation (degrees)", ylabel="Class probability", title=family.upper(), ylim=(-.02, 1.02))
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
    fig.suptitle(f"Experiment 5 · #{index} · local switch/return illustration selected after search\nSame 28x28 rendering; all transition endpoints audited. Points are tested values; interpolation/quantization can produce narrow reentrant decisions.")
    path = out / f"figures/boundary_zoom_sample_{index:05d}.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return {"sample_index": index, "axis": axis, "chosen_ticks": chosen_ticks,
            "selection": "Post-search illustrative ResNet positive-ray switch, return and next departure; not a new image-selection procedure.",
            "figure": str(path.relative_to(out))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results/counterfactual")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--zoom-index", type=int, default=2878)
    args = parser.parse_args()
    bundle = e.setup(args)
    config, signature, frozen, ids, images, labels, trusted, names = bundle
    data = e.collect(args.output, bundle, include_refine=True)
    transitions = e.read_csv(args.output / "tables/all_observed_transitions.csv")
    endpoints = sorted({(r["axis"], int(r["sample_index"]), int(r[k]))
                        for r in transitions for k in ("previous_tick", "switch_tick")})
    all_images, expected = [], {f: [] for f in e.FAMILIES}
    for axis, index, tick in endpoints:
        pos = int(np.flatnonzero(ids == index)[0])
        all_images.append(e.render(images[pos], axis, [tick], config["axes"][axis], config["fill_rgb"])[0])
        ticks, p = data[axis, index]
        where = int(np.flatnonzero(ticks == tick)[0])
        for family in e.FAMILIES: expected[family].append(p[family][where])
    all_images = np.stack(all_images)
    records, audits = [], {}
    for family in e.FAMILIES:
        predictor = LocalPredictor(family, names, args.device, args.batch_size, config["seed"])
        try:
            baseline_audit = json.loads((args.output / f"predictions/{family}/reproducibility.json").read_text())
            for key in ("device", "torch_version", "seed", "preprocessing"):
                if predictor.info[key] != baseline_audit[key]:
                    raise ValueError(f"Endpoint audit environment mismatch: {key}")
            import PIL
            import torch
            import transformers
            for key, value in (("numpy_version", np.__version__), ("pillow_version", PIL.__version__),
                               ("transformers_version", transformers.__version__), ("batch_size", args.batch_size),
                               ("device_name", torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else str(predictor.device))):
                if value != baseline_audit[key]: raise ValueError(f"Endpoint audit environment mismatch: {key}")
            repeats = np.stack([predictor.predict(all_images)[1] for _ in range(3)])
            saved = np.stack(expected[family])
            np.testing.assert_allclose(repeats, np.broadcast_to(saved, repeats.shape), atol=2e-5, rtol=1e-4)
            np.testing.assert_array_equal(repeats.argmax(-1), np.broadcast_to(saved.argmax(-1), repeats.shape[:2]))
            margin = np.sort(saved, axis=1)[:, -1] - np.sort(saved, axis=1)[:, -2]
            for j, (axis, index, tick) in enumerate(endpoints):
                records.append({"sample_index": index, "axis": axis, "tick": tick,
                    "value": tick*config["axes"][axis]["unit"], "model": family,
                    "saved_prediction": int(saved[j].argmax()), "top2_probability_margin": float(margin[j]),
                    "max_probability_error": float(np.abs(repeats[:, j]-saved[j]).max()),
                    "repeat_probability_range": float(np.ptp(repeats[:, j], axis=0).max()),
                    "all_three_argmax_match": True})
            audits[family] = {"endpoints": len(endpoints), "repeats": 3,
                "max_abs_error_from_saved": float(np.abs(repeats-saved).max()),
                "max_repeat_probability_range": float(np.ptp(repeats, axis=0).max()),
                "min_top2_probability_margin": float(margin.min()), "argmax_mismatches": 0,
                "warnings": sorted(predictor.warning_messages)}
        finally:
            predictor.close()
    e.csv_dicts(args.output / "validation/transition_endpoint_audit.csv", records)
    example = zoom(args.output, bundle, data, args.zoom_index)
    report = {"protocol_signature": signature, "audit_code_sha256": e.file_digest(Path(__file__)),
        "raw_inputs_sha256": e.digest(all_images), "endpoint_input_count": len(endpoints),
        "additional_inference_evaluations": len(endpoints)*3*2,
        "scope": "Union of both models' observed transition endpoints, repeated three times for each model. Not all unsampled points or a cross-device guarantee.",
        "models": audits, "local_zoom": example}
    e.save_json(args.output / "validation/transition_endpoint_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
