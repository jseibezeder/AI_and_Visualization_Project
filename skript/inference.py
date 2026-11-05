"""Local black-box inference shared by new experiments (no training/downloads)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class LocalPredictor:
    def __init__(self, family, class_names, device=None, batch_size=32, seed=42):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        ))
        self.batch_size = batch_size
        self.family = family
        self.processor = None
        self.warning_messages = set()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if family == "resnet":
            from models.resnet18.models import ResNet18
            self.model = ResNet18(in_channels=3, num_classes=8)
            checkpoint = torch.load(ROOT / "models/resnet18/resnet18_28_1.pth",
                                    map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint["net"], strict=True)
        elif family == "vit":
            from transformers import ViTForImageClassification, ViTImageProcessor
            self.processor = ViTImageProcessor.from_pretrained(ROOT / "models/vit", local_files_only=True)
            self.model = ViTForImageClassification.from_pretrained(ROOT / "models/vit", local_files_only=True)
            if [self.model.config.id2label[i] for i in range(8)] != class_names:
                raise ValueError("ViT class mapping differs from dataset")
        else:
            raise ValueError(family)
        self.model = self.model.to(self.device).eval()
        self.info = {"device": str(self.device), "torch_version": str(torch.__version__),
                     "evaluation_mode": True, "deterministic_algorithms": True,
                     "deterministic_warn_only": True, "seed": seed,
                     "preprocessing": "RGB 28x28, /255, mean/std=0.5" if family == "resnet"
                     else "Local ViTImageProcessor: RGB resize224 bilinear, /255, mean/std=0.5"}

    def predict(self, images):
        torch = self.torch
        if images.dtype != np.uint8 or images.shape[1:] != (28, 28, 3) or len(images) == 0:
            raise ValueError("Expected nonempty uint8 RGB batch at original 28x28 resolution")
        outputs = []
        with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start+self.batch_size]
                if self.processor is None:
                    x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float()
                    logits = self.model(((x / 255 - .5) / .5).to(self.device))
                else:
                    inputs = self.processor(images=[Image.fromarray(x) for x in batch], return_tensors="pt")
                    if tuple(inputs["pixel_values"].shape[1:]) != (3, 224, 224):
                        raise ValueError("Unexpected ViT input resolution")
                    logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
                outputs.append(logits.float().cpu())
            logits = torch.cat(outputs)
            probabilities = logits.softmax(-1)
            self.warning_messages.update(str(w.message) for w in caught)
        return logits.numpy(), probabilities.numpy()

    def close(self):
        del self.model
        if self.device.type == "mps":
            self.torch.mps.empty_cache()
        elif self.device.type == "cuda":
            self.torch.cuda.empty_cache()
