"""Export a checkpoint to ONNX for local CPU serving. Requires the 'torch' and 'onnx' extras."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("urbiva_ai.inference.export requires PyTorch: pip install 'urbiva-ai[torch,onnx]'") from exc

from urbiva_ai.inference.predictor import Predictor


def export_onnx(checkpoint: str | Path, output: str | Path, opset: int = 17) -> Path:
    pred = Predictor(checkpoint, device="cpu")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, pred.size, pred.size)
    torch.onnx.export(
        pred.model, dummy, str(out), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
    )
    return out
