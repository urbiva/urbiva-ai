import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urbiva_ai.data import manifest as mf
from urbiva_ai.data.transforms import letterbox, load_image, preprocess
from urbiva_ai.training.budget import ComputeBudget

ROOT = Path(__file__).resolve().parents[1]


def test_example_manifest_validates():
    sources = mf.load_sources(ROOT / "datasets" / "examples" / "sources.example.json")
    counts = mf.validate(mf.read_manifest(ROOT / "datasets" / "examples" / "manifest.example.jsonl"), sources)
    assert counts == {"stairs": 1, "none": 1}


@pytest.mark.parametrize("change,msg", [
    ({"label": "ramp"}, "unknown label"),
    ({"faces_blurred": False}, "blurred"),
    ({"path": "../x.jpg"}, "relative"),
])
def test_bad_records(change, msg):
    rec = {"id": "1", "path": "a.jpg", "label": "curb", "source_id": "s", "location_group": "g",
           "faces_blurred": True, "plates_blurred": True}
    rec.update(change)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.parse_record(rec)


def test_sources_reject_non_commercial(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"source_id": "x", "name": "x", "license": "CC-BY-4.0", "commercial_use": False}]))
    with pytest.raises(mf.ManifestError, match="commercial"):
        mf.load_sources(p)


def test_split_is_deterministic_and_grouped():
    groups = [f"g{i}" for i in range(2000)]
    s1 = [mf.split_of(g) for g in groups]
    assert s1 == [mf.split_of(g) for g in groups]
    share = {k: s1.count(k) / len(s1) for k in ("train", "val", "test")}
    assert 0.07 < share["val"] < 0.13 and 0.07 < share["test"] < 0.13


def test_preprocess_strips_and_normalises():
    buf = io.BytesIO()
    Image.new("RGB", (640, 320), (200, 100, 50)).save(buf, "JPEG")
    x = preprocess(buf.getvalue(), 320)
    assert x.shape == (3, 320, 320) and x.dtype == np.float32
    im = load_image(buf.getvalue())
    assert not im.info.get("exif")
    assert letterbox(im, 64).size == (64, 64)


def test_compute_budget():
    t = [0.0]
    b = ComputeBudget(2.0, num_gpus=2, safety_margin=0.0, clock=lambda: t[0])
    b.start()
    t[0] = 1800  # 0.5 h wall clock on 2 GPUs = 1 GPU-hour
    assert b.used_gpu_hours == pytest.approx(1.0)
    assert b.can_continue(0.4) and not b.can_continue(0.6)
