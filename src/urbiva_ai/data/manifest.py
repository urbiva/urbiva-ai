"""Dataset manifests and provenance validation.

Every image used for training or evaluation is listed in a JSON Lines manifest.
A record is accepted only if its source is registered with a licence that
permits the intended use, and if privacy processing has been applied.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from urbiva_ai.labels import CATEGORIES

REQUIRED_FIELDS = ("id", "path", "label", "source_id", "location_group", "faces_blurred", "plates_blurred")

ALLOWED_LICENSES = frozenset(
    {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "ODbL-1.0", "own-capture-consented", "partner-agreement"}
)
"""Licences accepted without additional review. Anything else needs a legal decision."""


class ManifestError(ValueError):
    """Raised when a manifest or a provenance record is invalid."""


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    license: str
    commercial_use: bool
    url: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Record:
    id: str
    path: str
    label: str
    source_id: str
    location_group: str
    faces_blurred: bool
    plates_blurred: bool
    sha256: str = ""


def load_sources(path: str | Path) -> dict[str, Source]:
    """Load the source registry (JSON list) and check licences."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ManifestError("source registry must be a JSON list")
    out: dict[str, Source] = {}
    for i, item in enumerate(raw):
        try:
            src = Source(**item)
        except TypeError as exc:
            raise ManifestError(f"source #{i}: {exc}") from exc
        if src.source_id in out:
            raise ManifestError(f"duplicate source_id {src.source_id!r}")
        if src.license not in ALLOWED_LICENSES:
            raise ManifestError(f"source {src.source_id!r}: licence {src.license!r} needs legal review")
        if not src.commercial_use:
            raise ManifestError(f"source {src.source_id!r}: commercial use is not permitted")
        out[src.source_id] = src
    return out


def parse_record(obj: dict, line: int = 0) -> Record:
    missing = [k for k in REQUIRED_FIELDS if k not in obj]
    if missing:
        raise ManifestError(f"line {line}: missing fields {missing}")
    if obj["label"] not in CATEGORIES:
        raise ManifestError(f"line {line}: unknown label {obj['label']!r}")
    if not (obj["faces_blurred"] is True and obj["plates_blurred"] is True):
        raise ManifestError(f"line {line}: faces and licence plates must be blurred before use")
    p = str(obj["path"])
    if p.startswith("/") or ".." in Path(p).parts:
        raise ManifestError(f"line {line}: path must be relative and inside the image root")
    return Record(
        id=str(obj["id"]),
        path=p,
        label=obj["label"],
        source_id=str(obj["source_id"]),
        location_group=str(obj["location_group"]),
        faces_blurred=True,
        plates_blurred=True,
        sha256=str(obj.get("sha256", "")),
    )


def read_manifest(path: str | Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"line {n}: invalid JSON ({exc.msg})") from exc
            yield parse_record(obj, n)


def validate(records: Iterable[Record], sources: dict[str, Source]) -> Counter:
    """Check ids are unique and every record has a registered source. Returns label counts."""
    seen: set[str] = set()
    counts: Counter = Counter()
    for r in records:
        if r.id in seen:
            raise ManifestError(f"duplicate record id {r.id!r}")
        seen.add(r.id)
        if r.source_id not in sources:
            raise ManifestError(f"record {r.id!r}: source {r.source_id!r} is not registered")
        counts[r.label] += 1
    return counts


def split_of(location_group: str, val_fraction: float = 0.1, test_fraction: float = 0.1, salt: str = "uva") -> str:
    """Deterministic split by location group, so one place never appears in two splits."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("fractions must be non-negative and sum to less than 1")
    h = int(hashlib.sha256(f"{salt}:{location_group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_fraction:
        return "test"
    if h < test_fraction + val_fraction:
        return "val"
    return "train"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
