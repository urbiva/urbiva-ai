import base64
import json

import numpy as np
import pytest

from urbiva_ai.inference.contract import (ContractError, VisionResult, confidence_level, decode_request,
                                          result_from_probs)
from urbiva_ai.inference.server import RateLimiter, handle

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def body(data: bytes, media: str = "image/png") -> bytes:
    return json.dumps({"imageBase64": base64.b64encode(data).decode(), "mediaType": media}).encode()


def test_result_shape_matches_app():
    p = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    r = result_from_probs(p).to_dict()
    assert set(r) == {"category", "confidence", "rationale", "requiresHumanReview"}
    assert r["category"] == "stairs" and r["confidence"] == "high" and r["requiresHumanReview"] is True


def test_confidence_levels():
    assert confidence_level(0.3) == "low"
    assert confidence_level(0.6) == "medium"
    assert confidence_level(0.95) == "high"


@pytest.mark.parametrize("payload,status", [
    ({"imageBase64": "abc", "mediaType": "image/gif"}, 415),
    ({"mediaType": "image/png"}, 400),
    ({"imageBase64": "!!!", "mediaType": "image/png"}, 400),
    ({"imageBase64": base64.b64encode(b"not an image").decode(), "mediaType": "image/png"}, 415),
])
def test_request_validation(payload, status):
    with pytest.raises(ContractError) as e:
        decode_request(payload)
    assert e.value.status == status


def test_valid_png_decodes():
    raw, media = decode_request(json.loads(body(PNG)))
    assert raw == PNG and media == "image/png"


def test_server_without_model_is_not_configured():
    status, obj = handle(body(PNG), None, "m")
    assert status == 503 and obj["status"] == "not_configured"


def test_server_with_stub_model():
    stub = lambda raw: VisionResult("curb", "medium", "stub")  # noqa: E731
    status, obj = handle(body(PNG), stub, "m")
    assert status == 200 and obj["result"]["category"] == "curb" and obj["model"] == "m"
    assert handle(b"{bad", stub, "m")[0] == 400


def test_server_hides_model_errors():
    def boom(raw):
        raise RuntimeError("secret detail")
    status, obj = handle(body(PNG), boom, "m")
    assert status == 502 and "secret" not in obj["message"]


def test_rate_limiter():
    t = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: t[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    t[0] = 61
    assert rl.allow("a")
