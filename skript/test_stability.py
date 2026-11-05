"""Run: python -m unittest discover -s skript -p 'test_stability.py' -v"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import stability as s


class TransformationTests(unittest.TestCase):
    def setUp(self):
        self.images = np.random.default_rng(1).integers(0, 256, (4, 28, 28, 3), dtype=np.uint8)

    def apply(self, kind, level, repeat=0):
        return s.transform(self.images, {"kind": kind, "level": level, "realization": repeat}, 42, [128]*3)

    def test_identity_transforms_do_not_mutate_original(self):
        original = self.images.copy()
        for kind, level in (("rotation", 0), ("translation", 0), ("brightness", 1),
                            ("blur", 0), ("noise", 0), ("red_scale", 1),
                            ("green_scale", 1), ("blue_scale", 1)):
            np.testing.assert_array_equal(self.apply(kind, level), original)
        np.testing.assert_array_equal(self.images, original)

    def test_translation_direction_and_no_wrap(self):
        right = self.apply("translation", 3)
        np.testing.assert_array_equal(right[:, :, 3:], self.images[:, :, :-3])
        np.testing.assert_array_equal(right[:, :, :3], 128)
        left = self.apply("translation", -2)
        np.testing.assert_array_equal(left[:, :, :-2], self.images[:, :, 2:])
        np.testing.assert_array_equal(left[:, :, -2:], 128)

    def test_noise_is_reproducible_but_realizations_are_distinct(self):
        a = self.apply("noise", .05, 0)
        np.testing.assert_array_equal(a, self.apply("noise", .05, 0))
        self.assertFalse(np.array_equal(a, self.apply("noise", .05, 1)))
        self.assertFalse(np.array_equal(a, self.apply("noise", .1, 0)))
        self.assertEqual(a.dtype, np.uint8)

    def test_channel_scaling_only_changes_selected_channel(self):
        modified = self.apply("red_scale", 1.1)
        np.testing.assert_array_equal(modified[..., 1:], self.images[..., 1:])
        self.assertTrue(np.all(modified[..., 0] >= self.images[..., 0]))

    def test_condition_count(self):
        config = {"levels": s.DEFAULT_LEVELS, "noise_realizations": 3}
        cs = list(s.conditions(config))
        self.assertEqual(len(cs), 38)
        self.assertEqual(len({c["key"] for c in cs}), len(cs))
        self.assertEqual(sum(c["kind"] == "noise" for c in cs), 12)


class IntegrationTests(unittest.TestCase):
    def test_subset_and_report_in_temporary_directory(self):
        # Saved baselines are reused, not recomputed. Synthetic transformed
        # predictions below are ONLY for testing; the TemporaryDirectory is removed.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = argparse.Namespace(output=out)
            config, signature, images, labels, indices, baseline = s.setup(args)
            self.assertEqual(images.shape, (320, 28, 28, 3))
            np.testing.assert_array_equal(np.bincount(labels), np.full(8, 40))
            again = s.setup(args)
            np.testing.assert_array_equal(indices, again[4])
            self.assertEqual(signature, again[1])
            # Reduce only the levels to keep the test fast; still test every kind.
            config["levels"] = {k: [v[0]] for k, v in config["levels"].items()}
            for c in s.conditions(config):
                changed = s.cached_images(out, config, signature, images, c)
                np.testing.assert_array_equal(changed, s.cached_images(out, config, signature, images, c))
                for family in s.FAMILIES:
                    path = out / "predictions" / family / f"{c['key']}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, probabilities=baseline[family], sample_index=indices,
                                        protocol_signature=signature, transformed_sha256=s.digest(changed))
            s.report(out, config, signature, images, labels, indices, baseline)
            self.assertTrue((out / "figures/behavioral_fingerprints.png").exists())
            self.assertEqual(len(list((out / "figures/curves").glob("*.png"))), 8)
            self.assertEqual(len(list((out / "figures/transition_grids").glob("*.png"))), 16)
            import csv
            with (out / "tables/stability_metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(float(r["flip_fraction"]) == 0 for r in rows))
            self.assertTrue(all(abs(float(r["probability_change"])) < 1e-6 for r in rows))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["n_images"], 320)


if __name__ == "__main__":
    unittest.main()
