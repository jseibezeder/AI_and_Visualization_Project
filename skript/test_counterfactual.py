"""Focused tests; synthetic probability paths never touch production results."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import counterfactual as e
import stability


class CounterfactualTests(unittest.TestCase):
    def test_exact_equivalence_to_prior_geometry(self):
        image = np.random.default_rng(5).integers(0, 256, (28, 28, 3), dtype=np.uint8)
        original = image.copy()
        for axis, levels in (("rotation", [-20, -10, -5, 0, 5, 10, 20]),
                             ("translation", [-3, -2, -1, 0, 1, 2, 3])):
            spec = e.DEFAULT_AXES[axis]
            ticks = np.array([round(v/spec["unit"]) for v in levels])
            changed = e.render(image, axis, ticks, spec, [128]*3)
            for value, actual in zip(levels, changed):
                expected = stability.transform(image[None], {"kind": axis, "level": value, "realization": 0}, 45, [128]*3)[0]
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(changed[levels.index(0)], original)
        np.testing.assert_array_equal(image, original)

    def test_translation_sign_border_and_fractional_inputs(self):
        image = np.full((28, 28, 3), 100, np.uint8)
        image[:, 10:14] = 220
        spec = e.DEFAULT_AXES["translation"]
        changed = e.render(image, "translation", np.array([-80, 0, 1, 80]), spec, [128]*3)
        np.testing.assert_array_equal(changed[3, :, 11:15], image[:, 10:14])
        np.testing.assert_array_equal(changed[3, :, 0], np.full((28, 3), 128))
        np.testing.assert_array_equal(changed[0, :, -1], np.full((28, 3), 128))
        self.assertEqual(changed.dtype, np.uint8)
        self.assertEqual(changed.shape, (4, 28, 28, 3))

    def test_global_scan_detects_coarse_same_endpoint_island(self):
        spec = {"max_tick": 20, "coarse_stride": 10, "scan_stride": 2}
        coarse = e.grid_ticks(spec, "coarse")
        scan = np.union1d(coarse, e.grid_ticks(spec, "scan"))
        # A pulse between coarse values 0 and 10: same labels at endpoints.
        coarse_labels = ((coarse >= 3) & (coarse <= 7)).astype(int)
        np.testing.assert_array_equal(coarse_labels, 0)
        labels = ((scan >= 3) & (scan <= 7)).astype(int)
        refined = e.refinement_ticks(scan, [labels, labels])
        self.assertIn(3, refined)
        self.assertIn(7, refined)
        self.assertNotIn(5, refined)  # same-endpoint scan cells are not certified!
        # Another model with a distinct transition adds its bracket to the union.
        other = (scan >= 11).astype(int)
        union = e.refinement_ticks(scan, [labels, other])
        self.assertIn(11, union)
        np.testing.assert_array_equal(union, e.refinement_ticks(scan[::-1], [labels[::-1], other[::-1]]))

    def test_non_monotonic_signed_rays_and_censoring(self):
        ticks = np.array([-20, -10, 0, 1, 2, 3, 4, 5, 10, 20])
        labels = np.array([0, 2, 0, 0, 1, 2, 0, 1, 0, 0])
        p = np.eye(8)[labels]
        spec = {"unit": .25}
        positive = e.ray_summary(ticks, p, 0, spec, 1)
        self.assertEqual(positive["first_switch_tick"], 2)
        self.assertEqual(positive["previous_same_tick"], 1)
        self.assertEqual(positive["return_count"], 2)
        self.assertEqual(len(positive["transitions"]), 5)
        negative = e.ray_summary(ticks, p, 0, spec, -1)
        self.assertEqual(negative["first_switch_tick"], -10)
        self.assertEqual(negative["return_count"], 1)
        none = e.ray_summary(ticks, np.repeat(np.eye(8)[0:1], len(ticks), axis=0), 0, spec, 1)
        self.assertIsNone(none["first_switch_tick"])
        self.assertEqual(none["transitions"], [])

    def test_selection_uses_all_prior_experiments_and_existing_set(self):
        records = e.select_examples()
        self.assertEqual(len(records), 8)
        self.assertEqual(len({r["sample_index"] for r in records}), 8)
        self.assertEqual({x: sum(r["experiment"] == x for r in records) for x in ("1", "2", "3", "4")}, {x: 2 for x in ("1", "2", "3", "4")})
        original = json.loads((e.ROOT / e.SOURCE_FILES[0]).read_text())
        self.assertTrue({r["sample_index"] for r in records} <= set(original["sample_indices"]))
        self.assertEqual(records, e.select_examples())

    def test_synthetic_full_report_refinement_policy_and_cache_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); args = argparse.Namespace(output=out)
            bundle = e.setup(args)
            config, signature, frozen, ids, images, labels, base, names = bundle
            e.save_json(out / "protocol.json", {"signature": signature})
            for family in e.FAMILIES:
                e.save_json(out / f"predictions/{family}/reproducibility.json", {
                    "protocol_signature": signature, "environment_signature": "synthetic_only",
                    "stages": {"coarse": {"unique_inferred_inputs": 0}, "scan": {"unique_inferred_inputs": 0}}})
            for stage in e.STAGES:
                scanned = e.collect(out, bundle) if stage == "refine" else None
                for pos, index in enumerate(ids):
                    for axis, spec in config["axes"].items():
                        if stage == "refine":
                            tested, predictions = scanned[axis, int(index)]
                            ticks = e.refinement_ticks(tested, [p.argmax(1) for p in predictions.values()])
                        else:
                            ticks = e.grid_ticks(spec, stage)
                        if not len(ticks): continue
                        key, changed, h = e.input_job(out, signature, axis, int(index), stage, ticks, images[pos], spec, config["fill_rgb"])
                        for family in e.FAMILIES:
                            p = np.repeat(base[family][pos:pos+1], len(ticks), axis=0)
                            island = (ticks >= 23) & (ticks <= 47)
                            p[island] = np.roll(p[island], 1, axis=1)
                            e.atomic_npz(out / f"predictions/{family}/{key}.npz", logits=np.log(np.maximum(p, 1e-30)),
                                probabilities=p, ticks=ticks, sample_index=np.full(len(ticks), index, np.int64),
                                input_sha256=h, protocol_signature=signature, environment_signature="synthetic_only")
            e.report(out, bundle, render_figures=False)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["observed_returns_to_baseline"], 32)
            self.assertEqual(summary["directional_no_observed_switches"], 32)
            self.assertTrue(all(r["nearest_tick"] == 23 for r in summary["nearest_observed_switches"]))
            data = e.collect(out, bundle, include_refine=True)
            index = int(ids[0]); ticks, p = data["rotation", index]
            baseline_classes = {f: int(base[f][0].argmax()) for f in e.FAMILIES}
            e.plot_curves(out, "rotation", index, ticks, p, baseline_classes, config["axes"]["rotation"])
            e.plot_counterfactual(out, "rotation", index, images[0], int(labels[0]), ticks, p, baseline_classes, config["axes"]["rotation"], config["fill_rgb"])
            self.assertEqual(len(list((out / "figures").rglob("*.png"))), 2)
            # Truncating a refined stage must fail instead of silently reporting
            # a coarser/monotonic first boundary.
            path = out / f"interventions/rotation/sample_{index:05d}/refine.npz"
            with np.load(path, allow_pickle=False) as z:
                t = z["ticks"]; im = z["images"]
            e.atomic_npz(path, images=im[:-1], ticks=t[:-1], input_sha256=e.digest(im[:-1]), protocol_signature=signature)
            with self.assertRaises((AssertionError, ValueError)):
                e.report(out, bundle, render_figures=False)
            c = json.loads((out / "config.json").read_text()); c["seed"] += 1
            e.save_json(out / "config.json", c)
            with self.assertRaisesRegex(ValueError, "Frozen"):
                e.setup(args)


if __name__ == "__main__":
    unittest.main()
