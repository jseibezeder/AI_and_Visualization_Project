"""Verify supplied assets; optionally run offline CPU inference for both models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": 11959, "val": 1712, "test": 3421}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_hashes() -> None:
    manifest = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    for asset in manifest["files"]:
        path = ROOT / asset["path"]
        require(path.is_file(), f"Missing {asset['path']}; try 'git lfs pull'.")
        require(
            path.stat().st_size == asset["bytes"],
            f"Wrong size for {asset['path']}; try 'git lfs pull' for LFS pointers.",
        )
        with path.open("rb") as stream:
            # Streaming I/O avoids loading checkpoint files into memory.
            checksum = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else file_sha256(stream)
        require(checksum == asset["sha256"], f"SHA-256 mismatch: {asset['path']}")
    print(f"OK: {len(manifest['files'])} upstream files match their SHA-256 hashes.")


def file_sha256(stream) -> str:
    """Python 3.10 fallback, also using bounded-memory binary reads."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_dataset() -> np.ndarray:
    metadata = json.loads((ROOT / "data/metadata.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "models/vit/config.json").read_text(encoding="utf-8"))
    require(metadata["n_samples"] == SPLITS, "Unexpected split sizes in metadata.")
    require(metadata["image_shape"] == [28, 28, 3], "Unexpected image shape in metadata.")
    require(config["id2label"] == metadata["label"], "ViT / dataset class mappings differ.")
    require(set(metadata["label"]) == {str(i) for i in range(8)}, "Expected eight classes.")
    with np.load(ROOT / "data/bloodmnist.npz", allow_pickle=False) as dataset:
        expected_keys = {f"{split}_{kind}" for split in SPLITS for kind in ("images", "labels")}
        require(set(dataset.files) == expected_keys, "Unexpected NPZ arrays.")
        for split, count in SPLITS.items():
            # Each compressed array is materialized just once, not per sample.
            images = dataset[f"{split}_images"]
            labels = dataset[f"{split}_labels"]
            require(images.shape == (count, 28, 28, 3), f"Wrong image shape: {split}")
            require(images.dtype == np.uint8, f"Expected uint8 images: {split}")
            require(labels.shape == (count, 1), f"Wrong label shape: {split}")
            require(np.issubdtype(labels.dtype, np.integer), f"Non-integer labels: {split}")
            require(np.array_equal(np.unique(labels), np.arange(8)), f"Unexpected classes: {split}")
            if split == "test":
                # One fixed example per class, rather than eight adjacent images.
                indices = np.unique(labels.reshape(-1), return_index=True)[1]
                sample = images[indices].copy()
    print("OK: 17,092 RGB images, official splits and eight matching class labels.")
    return sample


def smoke_inference(images: np.ndarray) -> None:
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification, ViTImageProcessor

    sys.path.insert(0, str(ROOT))
    from models.resnet18.models import ResNet18

    # Small batches are faster without excessive CPU thread oversubscription.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cnn = ResNet18(in_channels=3, num_classes=8)
    checkpoint = torch.load(
        ROOT / "models/resnet18/resnet18_28_1.pth",
        map_location="cpu",
        weights_only=True,
    )
    cnn.load_state_dict(checkpoint["net"], strict=True)
    cnn.eval()
    inputs = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float()
    inputs = (inputs / 255.0 - 0.5) / 0.5
    with torch.inference_mode():
        logits = cnn(inputs)
    require(logits.shape == (8, 8), "Unexpected CNN logits shape.")
    require(torch.isfinite(logits).all().item(), "Non-finite CNN logits.")
    print("OK: ResNet-18 loaded strictly; finite (8, 8) logits.")

    directory = ROOT / "models/vit"
    processor = ViTImageProcessor.from_pretrained(directory, local_files_only=True)
    vit = ViTForImageClassification.from_pretrained(directory, local_files_only=True).eval()
    inputs = processor(images=[Image.fromarray(image) for image in images], return_tensors="pt")
    require(inputs["pixel_values"].shape == (8, 3, 224, 224), "Unexpected ViT inputs.")
    with torch.inference_mode():
        logits = vit(**inputs).logits
    require(logits.shape == (8, 8), "Unexpected ViT logits shape.")
    require(torch.isfinite(logits).all().item(), "Non-finite ViT logits.")
    print("OK: ViT loaded locally; finite (8, 8) logits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="also run offline CPU inference")
    args = parser.parse_args()
    verify_hashes()
    images = verify_dataset()
    if args.smoke:
        smoke_inference(images)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
