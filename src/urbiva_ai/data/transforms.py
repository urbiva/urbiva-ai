"""Image preprocessing in Pillow and NumPy, identical for training and serving."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PIXELS = 40_000_000


def load_image(data: bytes) -> Image.Image:
    """Decode bytes, apply EXIF orientation, drop metadata and convert to RGB."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    return clean


def letterbox(im: Image.Image, size: int, fill: tuple[int, int, int] = (114, 114, 114)) -> Image.Image:
    """Resize keeping aspect ratio, then pad to a size x size square."""
    if size <= 0:
        raise ValueError("size must be positive")
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_array(im: Image.Image) -> np.ndarray:
    """HWC uint8 to normalised CHW float32."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def preprocess(data: bytes, size: int) -> np.ndarray:
    return to_array(letterbox(load_image(data), size))


def augment(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Light, label-preserving augmentation. No vertical flips: stairs have a direction."""
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    if rng.random() < 0.8:
        from PIL import ImageEnhance

        im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.75, 1.25)))
        im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.75, 1.25)))
        im = ImageEnhance.Color(im).enhance(float(rng.uniform(0.8, 1.2)))
    if rng.random() < 0.5:
        w, h = im.size
        s = float(rng.uniform(0.8, 1.0))
        cw, ch = max(1, int(w * s)), max(1, int(h * s))
        x0 = int(rng.integers(0, w - cw + 1))
        y0 = int(rng.integers(0, h - ch + 1))
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
    return im
