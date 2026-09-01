"""Inference and export utilities for layered single-image representations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class Layer:
    """One semantically grouped, depth-ordered RGBA layer."""

    name: str
    semantic_label: str
    confidence: float
    mask: np.ndarray
    depth: float


def _safe_name(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_") or "layer"


def _to_mask(mask: object, size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        return np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) > 127
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape[::-1] != size:
        array = np.asarray(Image.fromarray(array.astype(np.uint8)).resize(size, Image.Resampling.NEAREST))
    return array > 0


def _rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    return Image.fromarray(np.dstack((rgb, alpha.astype(np.uint8))), "RGBA")


class LayeredRepresentationPipeline:
    """Build semantic RGBA layers, rank them by depth, and export a manifest.

    The ``transformers`` backend uses pretrained Mask2Former and Depth Anything
    models. ``heuristic`` is an offline smoke-test backend; it does not claim
    semantic recognition and is intentionally labelled as a baseline.
    """

    def __init__(self, backend: str = "transformers", min_area: float = 0.01) -> None:
        if backend not in {"transformers", "heuristic"}:
            raise ValueError("backend must be 'transformers' or 'heuristic'")
        self.backend = backend
        self.min_area = min_area

    def _transformer_layers(self, image: Image.Image) -> tuple[list[Layer], np.ndarray]:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before using the transformers backend.") from exc

        segmenter = pipeline(
            "image-segmentation",
            model="facebook/mask2former-swin-small-ade-semantic",
            device=-1,
        )
        estimator = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=-1,
        )
        results = segmenter(image)
        predicted = estimator(image)["predicted_depth"]
        depth = np.asarray(predicted.squeeze().detach().cpu(), dtype=np.float32)
        depth = np.asarray(Image.fromarray(depth).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32)
        depth = (depth - depth.min()) / (np.ptp(depth) + 1e-8)

        layers: list[Layer] = []
        min_pixels = self.min_area * image.width * image.height
        for item in results:
            mask = _to_mask(item["mask"], image.size)
            if mask.sum() < min_pixels:
                continue
            label = str(item.get("label", "unknown"))
            score = float(item.get("score", 0.0))
            layers.append(Layer(_safe_name(label), label, score, mask, float(depth[mask].mean())))
        if not layers:
            raise RuntimeError("No layer passed the minimum-area threshold. Try --min-area 0.001.")
        return self._make_disjoint(layers), depth

    def _heuristic_layers(self, image: Image.Image) -> tuple[list[Layer], np.ndarray]:
        """A deterministic, dependency-light baseline for repository testing."""
        rgb = np.asarray(image, dtype=np.float32)
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        # Four spatial/color clusters, with lower image positions treated as nearer.
        features = np.dstack((rgb / 255.0, xx[..., None] / max(w, 1), yy[..., None] / max(h, 1))).reshape(-1, 5)
        centers = features[[0, w - 1, (h - 1) * w, h * w - 1]].copy()
        labels = np.zeros(features.shape[0], dtype=np.int32)
        for _ in range(12):
            distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for k in range(4):
                selected = features[labels == k]
                if len(selected):
                    centers[k] = selected.mean(axis=0)
        label_map = labels.reshape(h, w)
        depth = yy.astype(np.float32) / max(h - 1, 1)
        layers = []
        for k in range(4):
            mask = label_map == k
            if mask.sum() >= self.min_area * h * w:
                layers.append(Layer(f"region_{k + 1}", f"color_region_{k + 1}", 0.0, mask, float(depth[mask].mean())))
        return self._make_disjoint(layers), depth

    @staticmethod
    def _make_disjoint(layers: Iterable[Layer]) -> list[Layer]:
        """Assign overlapping masks to the highest-confidence layer once only."""
        assigned: np.ndarray | None = None
        output: list[Layer] = []
        for layer in sorted(layers, key=lambda item: item.confidence, reverse=True):
            if assigned is None:
                assigned = np.zeros_like(layer.mask, dtype=bool)
            mask = layer.mask & ~assigned
            assigned |= mask
            if mask.any():
                output.append(Layer(layer.name, layer.semantic_label, layer.confidence, mask, layer.depth))
        return output

    @staticmethod
    def _appearance_split(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple intrinsic-image proxy: chromatic albedo and luminance shading."""
        rgb_f = rgb.astype(np.float32) / 255.0
        luminance = np.clip(0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2], 0.08, 1.0)
        albedo = np.clip(rgb_f / luminance[..., None] * 0.55, 0.0, 1.0)
        albedo_u8 = (albedo * 255).astype(np.uint8)
        shading_u8 = (luminance * 255).astype(np.uint8)
        shading_rgb = np.repeat(shading_u8[..., None], 3, axis=2)
        return albedo_u8, shading_rgb

    def run(self, input_path: str | Path, output_dir: str | Path) -> dict:
        input_path, output_dir = Path(input_path), Path(output_dir)
        image = Image.open(input_path).convert("RGB")
        rgb = np.asarray(image)
        layers, depth = self._transformer_layers(image) if self.backend == "transformers" else self._heuristic_layers(image)
        layers.sort(key=lambda item: item.depth, reverse=True)  # near -> far
        output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for order, layer in enumerate(layers, start=1):
            layer_dir = output_dir / f"{order:02d}_{layer.name}"
            layer_dir.mkdir(exist_ok=True)
            alpha = layer.mask.astype(np.uint8) * 255
            rgba = _rgba(rgb, alpha)
            albedo, shading = self._appearance_split(rgb, layer.mask)
            rgba.save(layer_dir / "rgba.png")
            _rgba(albedo, alpha).save(layer_dir / "albedo_proxy.png")
            _rgba(shading, alpha).save(layer_dir / "shading_proxy.png")
            records.append({
                "order_near_to_far": order,
                "semantic_label": layer.semantic_label,
                "confidence": round(layer.confidence, 4),
                "mean_relative_depth": round(layer.depth, 4),
                "pixels": int(layer.mask.sum()),
                "files": {"rgba": str((layer_dir / "rgba.png").relative_to(output_dir)),
                          "albedo_proxy": str((layer_dir / "albedo_proxy.png").relative_to(output_dir)),
                          "shading_proxy": str((layer_dir / "shading_proxy.png").relative_to(output_dir))},
            })

        composite = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for layer in reversed(layers):  # far -> near for alpha compositing
            composite.alpha_composite(_rgba(rgb, layer.mask.astype(np.uint8) * 255))
        composite.save(output_dir / "composite.png")
        Image.fromarray((depth * 255).astype(np.uint8)).save(output_dir / "relative_depth.png")
        manifest = {
            "input": str(input_path),
            "backend": self.backend,
            "depth_convention": "higher relative depth is nearer; layers are listed near to far",
            "appearance_note": "Albedo and shading are luminance-based proxy layers, not ground-truth intrinsic decomposition.",
            "layers": records,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
