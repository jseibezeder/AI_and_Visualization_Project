"""Run: python -m unittest discover -s skript -p 'test_occlusion.py' -v"""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import occlusion as o


class OcclusionTests(unittest.TestCase):
    def test_grid_and_only_patch_changes(self):
        coords = o.positions()
        self.assertEqual(coords.shape, (169, 2))
        np.testing.assert_array_equal(coords[0], [0, 0])
        np.testing.assert_array_equal(coords[-1], [24, 24])
        self.assertEqual(len({tuple(x) for x in coords}), 169)
        image = np.random.default_rng(4).integers(0, 256, (28, 28, 3), dtype=np.uint8)
        original = image.copy()
        modified = o.occluded_versions(image, 2)
        self.assertEqual(modified.shape, (169, 28, 28, 3))
        for version, (top, left) in zip(modified, coords):
            outside = np.ones((28, 28), dtype=bool)
            outside[top:top+4, left:left+4] = False
            np.testing.assert_array_equal(version[outside], image[outside])
        np.testing.assert_array_equal(image, original)

    def test_constant_image_is_invariant(self):
        image = np.full((28, 28, 3), 100, dtype=np.uint8)
        np.testing.assert_array_equal(o.occluded_versions(image, 2), np.repeat(image[None], 169, axis=0))

    def test_targets_are_explicit_and_shared(self):
        self.assertEqual(o.targets(2, 2, 2), [('true_class', 2)])
        self.assertEqual(o.targets(2, 3, 3), [('true_class', 2), ('shared_preferred', 3)])
        self.assertEqual(o.targets(2, 2, 4), [('true_class', 2), ('resnet_preferred', 2), ('vit_preferred', 4)])

    def test_full_report_with_synthetic_predictions_in_temporary_directory(self):
        # Synthetic probabilities are used ONLY in this temporary directory.
        # No production predictions or model inference are touched.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = argparse.Namespace(output=out)
            config, signature, indices, images, labels, baseline, names = o.prepare(args)
            self.assertEqual(len(indices), 48)
            np.testing.assert_array_equal(np.bincount(labels), [6]*8)
            for i, index in enumerate(indices):
                changed = o.cached_interventions(out, signature, images[i], int(index), 2)
                for family in o.FAMILIES:
                    path = out / 'predictions' / family / f'sample_{index:05d}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, probabilities=np.repeat(baseline[family][i:i+1], 169, axis=0),
                                        positions=o.positions(), sample_index=index,
                                        protocol_signature=signature, intervention_sha256=o.digest(changed))
            o.report(out, config, signature, indices, images, labels, baseline, names)
            self.assertEqual(len(list((out/'figures/sensitivity_maps').glob('*.png'))), 48)
            self.assertEqual(len(list((out/'maps').glob('*.npz'))), 48)
            with (out/'tables/occlusion_sensitivity.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(float(row['importance']) == 0 for row in rows))
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['total_intervention_evaluations'], 16224)
            self.assertEqual(summary['global_color_scale'], [-1e-6, 1e-6])


if __name__ == '__main__':
    unittest.main()
