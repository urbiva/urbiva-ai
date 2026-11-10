"""The /api/vision response contract shared with urbiva-app.

A local model must return exactly the shape the application already accepts,
so it can replace the external provider without any change in the app.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass

import numpy as np

from urbiva_ai.labels import CATEGORIES, NONE_CATEGORY

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")

_MAGIC = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
}

CONFIDENCE_THRESHOLDS = (0.55, 0.8)
"""Top-1 probability bounds for 'medium' and 'high'. Set on validation data before use."""


class ContractError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class VisionResult:
    category: str
    confidence: str
    rationale: str
    requiresHumanReview: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def confidence_level(p: float, thresholds: tuple[float, float] = CONFIDENCE_THRESHOLDS) -> str:
    lo, hi = thresholds
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= medium < high <= 1")
    if p >= hi:
        return "high"
    if p >= lo:
        return "medium"
    return "low"


def result_from_probs(probs, thresholds: tuple[float, float] = CONFIDENCE_THRESHOLDS) -> VisionResult:
    """Map class probabilities to the app contract. A person always confirms."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != len(CATEGORIES):
        raise ValueError(f"expected {len(CATEGORIES)} probabilities")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-3):
        raise ValueError("probabilities must be non-negative and sum to 1")
    idx = int(p.argmax())
    category = CATEGORIES[idx]
    level = confidence_level(float(p[idx]), thresholds)
    if category == NONE_CATEGORY:
        rationale = "No barrier category scored above the others. Please check the photo yourself."
    else:
        rationale = (
            f"The local model estimates '{category.replace('_', ' ')}' "
            f"with {level} confidence. Please confirm or correct it."
        )
    return VisionResult(category=category, confidence=level, rationale=rationale)


def decode_request(payload: dict) -> tuple[bytes, str]:
    """Validate a request body {imageBase64, mediaType} like the app's handler does."""
    if not isinstance(payload, dict):
        raise ContractError(400, "Request body must be a JSON object.")
    media = payload.get("mediaType")
    data = payload.get("imageBase64")
    if media not in ALLOWED_MEDIA_TYPES:
        raise ContractError(415, "Only JPEG, PNG or WebP images are accepted.")
    if not isinstance(data, str) or not data:
        raise ContractError(400, "imageBase64 is missing.")
    if len(data) > (MAX_BYTES * 4) // 3 + 4:
        raise ContractError(413, "The image is larger than 5 MB.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError(400, "imageBase64 is not valid base64.") from exc
    if len(raw) > MAX_BYTES:
        raise ContractError(413, "The image is larger than 5 MB.")
    if media == "image/webp":
        ok = raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    else:
        ok = any(raw.startswith(m) for m in _MAGIC[media])
    if not ok:
        raise ContractError(415, "The file content does not match the media type.")
    return raw, media
