"""Run `python image_sandbox.py` and open http://127.0.0.1:8765.

Real checkpoint inference for edited TEST images; no training or public network binding.
The same engine/preprocessing is used by the targeted controls in archive.ipynb.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision.models import resnet18
from transformers import ViTForImageClassification, ViTImageProcessor
from models.resnet18.models import ResNet50

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ['resnet18', 'resnet50', 'vit']
RESAMPLING = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR,
              'bicubic': Image.Resampling.BICUBIC}


def transform_image(image, kind='none', level=1, *, test_id=0, seed=42, repeat=0):
    """Original raw-28 protocol; noise uses original test ID, exact level, seed and repeat."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if kind == 'none':
        return np.asarray(pil).copy()
    if kind == 'rotation':
        out = pil.rotate(float(level), resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    elif kind == 'translation':
        out = pil.transform(pil.size, Image.Transform.AFFINE, (1, 0, -float(level), 0, 1, 0),
                            resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    elif kind in ('brightness', 'saturation', 'contrast'):
        enhancer = {'brightness': ImageEnhance.Brightness, 'saturation': ImageEnhance.Color,
                    'contrast': ImageEnhance.Contrast}[kind]
        out = enhancer(pil).enhance(float(level))
    elif kind == 'blur':
        out = pil.filter(ImageFilter.GaussianBlur(float(level)))
    elif kind == 'noise':
        key = f'{seed}:{test_id}:{level}:{repeat}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
        out = np.rint(np.clip(image / 255. + rng.normal(0, level, image.shape), 0, 1) * 255).astype(np.uint8)
    elif kind in ('red_scale', 'green_scale', 'blue_scale'):
        out = image.astype(float).copy()
        out[..., {'red_scale': 0, 'green_scale': 1, 'blue_scale': 2}[kind]] *= level
        out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    else:
        raise ValueError('Unknown transformation')
    return np.asarray(out).copy()


class InferenceEngine:
    def __init__(self, batch_size=32):
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size, self.lock, self.calls = batch_size, threading.Lock(), 0
        files = ['weights_bloodmnist/resnet18_224_1.pth', 'weights_bloodmnist/resnet50_28_1.pth',
                 'models/vit/model.safetensors', 'models/vit/config.json', 'models/vit/preprocessor_config.json']
        self.weight_hashes = [hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files]
        with np.load(ROOT / 'predictions/occlusion_test.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(self.weight_hashes, saved['weight_hashes'])
        self.models = [resnet18(weights=None, num_classes=8), ResNet50(in_channels=3, num_classes=8)]
        self.checkpoints = ['weights_bloodmnist/resnet18_224_1.pth', 'weights_bloodmnist/resnet50_28_1.pth', 'models/vit']
        for model, path in zip(self.models, self.checkpoints):
            model.load_state_dict(torch.load(ROOT / path, map_location='cpu', weights_only=True)['net'])
        self.models.append(ViTForImageClassification.from_pretrained(ROOT / 'models/vit', local_files_only=True))
        self.models = [model.to(self.device).eval() for model in self.models]
        self.processor = ViTImageProcessor.from_pretrained(ROOT / 'models/vit', local_files_only=True)

    @torch.inference_mode()
    def predict_one(self, m, images, interpolation=None):
        """Optional common PIL interpolation controls apply to the two 224px pipelines only."""
        if interpolation is not None and interpolation not in RESAMPLING:
            raise ValueError('Unknown interpolation')
        probabilities, scores = [], []
        for start in range(0, len(images), self.batch_size):
            batch = np.asarray(images[start:start + self.batch_size], dtype=np.uint8)
            if m != 1 and interpolation is not None:
                batch = np.stack([np.asarray(Image.fromarray(im).resize((224, 224), RESAMPLING[interpolation])) for im in batch])
            if m != 2:
                x = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).to(self.device).float()
                if m == 0 and x.shape[-2:] != (224, 224):
                    x = torch.nn.functional.interpolate(x, (224, 224), mode='nearest')
                logits = self.models[m]((x / 255 - .5) / .5)
            else:
                inputs = self.processor(images=list(batch), return_tensors='pt',
                                        **({'do_resize': False} if interpolation is not None else {})).to(self.device)
                logits = self.models[m](**inputs).logits
            probabilities.append(logits.softmax(-1).cpu().numpy())
            scores.append(logits.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(scores)

    def predict(self, images):
        with self.lock:
            result = [self.predict_one(m, images) for m in range(3)]
            self.calls += 1
        return np.stack([x[0] for x in result]), np.stack([x[1] for x in result])


def png_url(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def serve(port):
    with np.load(ROOT / 'data/bloodmnist.npz', allow_pickle=False) as saved:
        images, labels = saved['test_images'], saved['test_labels'].ravel()
    with np.load(ROOT / 'predictions/high_confidence_error_analysis.npz', allow_pickle=False) as saved:
        baseline, classes = saved['baseline_probabilities'], saved['class_names'].tolist()
    assert len(images) == 3421 and baseline.shape == (3, 3421, 8)
    if (ROOT / 'predictions/targeted_followup.npz').exists():
        with np.load(ROOT / 'predictions/targeted_followup.npz', allow_pickle=False) as saved:
            assert str(saved['input_sha256']) == hashlib.sha256(images.tobytes()).hexdigest()
    engine = InferenceEngine()
    p, _ = engine.predict(images[:32])
    np.testing.assert_allclose(p, baseline[:, :32], atol=1e-5, rtol=1e-4)
    predictions = baseline.argmax(-1)
    hc_wrong = (predictions != labels) & (baseline.max(-1) >= .9)
    groups = {'all': np.arange(len(labels)), 'errors': np.flatnonzero(hc_wrong.any(0)),
              'ig_neut': np.flatnonzero((hc_wrong & (((labels == 3) & (predictions == 6)) |
                                                        ((labels == 6) & (predictions == 3)))).any(0))}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[0]).startswith('POST'):
                super().log_message(format, *args)

        def reply(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                data = (ROOT / 'visualizations/image_sandbox.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == '/health':
                self.reply(dict(app='bloodmnist-image-sandbox', ready=True, device=str(engine.device),
                                test_images=len(images), inference_calls=engine.calls))
            elif parsed.path == '/api/image':
                query = parse_qs(parsed.query)
                try:
                    pool = groups[query.get('group', ['all'])[0]]
                    i = int(query['id'][0]) if 'id' in query else int(pool[secrets.randbelow(len(pool))])
                    if not 0 <= i < len(images):
                        raise ValueError('Invalid test index')
                    self.reply(dict(id=i, label=int(labels[i]), classes=classes, image=png_url(images[i]),
                                    baseline=baseline[:, i].tolist()))
                except (ValueError, KeyError, IndexError):
                    self.reply({'error': 'Unknown group or test index'}, 400)
            else:
                self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            if self.path != '/api/predict':
                return self.reply({'error': 'Not found'}, 404)
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                return self.reply({'error': 'Cross-origin requests are not allowed'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 100_000 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected small JSON request')
                request = json.loads(self.rfile.read(length))
                im = Image.open(io.BytesIO(base64.b64decode(request['image'].split(',', 1)[-1], validate=True)))
                if im.size != (28, 28):
                    raise ValueError('Edited inputs must remain 28×28')
                image = np.asarray(im.convert('RGB'))
                kind, level = request.get('kind', 'none'), float(request.get('level', 1))
                limits = {'none': (0, 1), 'rotation': (-45, 45), 'translation': (-6, 6), 'brightness': (0, 2),
                          'saturation': (0, 2), 'contrast': (0, 2), 'blur': (0, 3), 'noise': (0, .3),
                          'red_scale': (0, 2), 'green_scale': (0, 2), 'blue_scale': (0, 2)}
                if kind not in limits or not np.isfinite(level) or not limits[kind][0] <= level <= limits[kind][1]:
                    raise ValueError('Transformation outside permitted range')
                test_id = int(request.get('id', 0))
                if not 0 <= test_id < len(images):
                    raise ValueError('Invalid test index')
                changed = transform_image(image, kind, level, test_id=test_id, seed=int(request.get('seed', 42)))
                started = time.perf_counter()
                probabilities, _ = engine.predict(changed[None])
                self.reply(dict(probabilities=probabilities[:, 0].tolist(), image=png_url(changed),
                                seconds=time.perf_counter() - started, device=str(engine.device)))
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                self.reply({'error': str(exc)}, 400)

    print(f'Ready: http://127.0.0.1:{port} · {engine.device} · baseline equivalence passed', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
