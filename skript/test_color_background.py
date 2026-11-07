"""Focused Experiment 4 tests; synthetic predictions remain in temporary folders."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageEnhance

import color_background as e
import stability


class ColorBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.images = np.random.default_rng(45).integers(20, 230, (3, 28, 28, 3), dtype=np.uint8)
        self.config = {"seed": 44, "region_blur_sigma": 2., "matched_deltas": [-16, -8, 8, 16],
                       "matched_max_pixels": 64, "matched_min_pixels": 24, "matched_repeats": 3}

    def test_identity_grayscale_and_input_immutability(self):
        original = self.images.copy()
        for condition in e.color_conditions():
            changed = e.color_transform(self.images, condition)
            self.assertEqual(changed.dtype, np.uint8)
            self.assertEqual(changed.shape, self.images.shape)
            if condition["kind"] == "identity":
                np.testing.assert_array_equal(changed, original)
            if condition["key"] == "grayscale":
                np.testing.assert_array_equal(changed[..., 0], changed[..., 1])
                np.testing.assert_array_equal(changed[..., 1], changed[..., 2])
        np.testing.assert_array_equal(self.images, original)

    def test_equivalence_to_existing_stability_transforms(self):
        # Reusing execution mechanics must not change existing RGB semantics.
        for kind in ("red_scale", "blue_scale", "brightness"):
            for level in (.9, .95, 1.05, 1.1):
                condition = {"kind": kind, "level": level, "realization": 0}
                expected = stability.transform(self.images, condition, 44, [128]*3)
                np.testing.assert_array_equal(e.color_transform(self.images, condition), expected)
        for level in (0, .5, .8, 1.2, 1.5):
            expected = np.stack([np.asarray(ImageEnhance.Color(Image.fromarray(i)).enhance(level)) for i in self.images])
            np.testing.assert_array_equal(e.color_transform(self.images, {"kind": "saturation", "level": level}), expected)

    def test_matched_region_budgets_and_outside_invariance(self):
        mask = e.ellipse_mask((14, 14, 8, 9)); original = self.images[0].copy()
        versions, rows, masks = e.region_versions(original, mask, 42, self.config)
        self.assertEqual(versions.shape, (28, 28, 28, 3))
        for changed, row, selected in zip(versions, rows, masks):
            np.testing.assert_array_equal(changed[~selected], original[~selected])
            self.assertEqual(row["outside_selected_changed_pixels"], 0)
            if row["design"] == "matched":
                self.assertEqual(row["selected_pixels"], 64)
                self.assertEqual(row["actually_changed_pixels"], 64)
                self.assertEqual(row["l1_rgb_total"], 64*3*abs(row["delta"]))
                np.testing.assert_array_equal(changed[selected].astype(int)-original[selected].astype(int),
                                              np.full((64, 3), row["delta"]))
        for repeat in range(3):
            for delta in self.config["matched_deltas"]:
                pair = [r for r in rows if r["repeat"] == repeat and r["delta"] == delta]
                self.assertEqual(len(pair), 2)
                for key in ("selected_pixels", "actually_changed_pixels", "l1_rgb_total", "rms_rgb_change_whole_image"):
                    self.assertEqual(pair[0][key], pair[1][key])
        np.testing.assert_array_equal(self.images[0], original)
        second = e.region_versions(original, mask, 42, self.config)
        np.testing.assert_array_equal(versions, second[0]); np.testing.assert_array_equal(masks, second[2])
        other = e.region_versions(original, mask, 43, self.config)
        self.assertFalse(np.array_equal(masks[4:], other[2][4:]))

    def test_insufficient_eligible_pixels_rejected(self):
        image = np.zeros((28, 28, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "eligible"):
            e.region_versions(image, e.ellipse_mask((14, 14, 8, 8)), 1, self.config)

    def test_production_subset_and_synthetic_report_cache_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(output=out)
            bundle = e.setup(args)
            config, signature, frozen, ids, images, labels, baseline, names, review, masks = bundle
            original_manifest = json.loads((e.ROOT / config["source_manifest"]).read_text())
            self.assertEqual(ids.tolist(), original_manifest["sample_indices"])
            np.testing.assert_array_equal(np.bincount(labels), [6]*8)
            region_labels = labels[[int(np.flatnonzero(ids == index)[0]) for index in masks]]
            np.testing.assert_array_equal(np.bincount(region_labels), [2]*8)
            e.save_json(out / "protocol.json", {"signature": signature})
            for family in e.FAMILIES:
                e.save_json(out / f"predictions/{family}/reproducibility.json", {
                    "protocol_signature": signature, "environment_signature": "synthetic_test_only"})
            for key, changed, sample_ids, metadata, selections in e.jobs(config, ids, images, masks):
                h = e.cache_inputs(out, signature, key, changed, sample_ids, selections)
                for family in e.FAMILIES:
                    lookup = {int(index): i for i, index in enumerate(ids)}
                    p = baseline[family][[lookup[int(index)] for index in sample_ids]]
                    e.atomic_npz(out / f"predictions/{family}/{key}.npz", probabilities=p,
                                 logits=np.log(np.maximum(p, 1e-30)), sample_index=sample_ids,
                                 protocol_signature=signature, input_sha256=h,
                                 environment_signature="synthetic_test_only")
            e.report(out, bundle)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["mild_one_model_only_flip_cases"], 0)
            self.assertEqual(summary["intervention_evaluations"], 2144)
            self.assertTrue(all(r["mean_probability_change"] == 0 for r in summary["color_results"]))
            self.assertEqual(len(list((out / "figures").glob("*.png"))), 8)
            changed = images.copy(); changed[0, 0, 0, 0] ^= 1
            with self.assertRaisesRegex(ValueError, "Stale/corrupt"):
                e.cache_inputs(out, signature, "color/original", changed, ids)
            with self.assertRaisesRegex(ValueError, "environment"):
                e.validate_prediction(out / "predictions/resnet/color/original.npz", signature, ids,
                                      e.digest(images), "wrong_environment")
            config_path = out / "config.json"
            modified = json.loads(config_path.read_text()); modified["seed"] += 1
            e.save_json(config_path, modified)
            with self.assertRaisesRegex(ValueError, "Frozen protocol"):
                e.setup(args)


if __name__ == "__main__":
    unittest.main()
