"""Load a trained checkpoint and produce app-compatible results. Requires PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("urbiva_ai.inference.predictor requires PyTorch: pip install 'urbiva-ai[torch]'") from exc

from urbiva_ai.config import from_dict
from urbiva_ai.data.transforms import preprocess
from urbiva_ai.inference.contract import VisionResult, result_from_probs
from urbiva_ai.models.uva_net import build_model


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        if not isinstance(state, dict) or "model" not in state or "config" not in state:
            raise ValueError("not a UVA-Net S10 checkpoint produced by urbiva_ai.training")
        self.cfg = from_dict(state["config"])
        m = self.cfg.model
        self.model = build_model(m.num_classes, m.width_multiplier, m.dropout, m.max_parameters)
        self.model.load_state_dict(state["model"])
        self.model.eval().to(device)
        self.device = torch.device(device)
        self.size = m.input_size

    @torch.no_grad()
    def probabilities(self, image_bytes: bytes) -> np.ndarray:
        x = torch.from_numpy(preprocess(image_bytes, self.size)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()

    def predict(self, image_bytes: bytes) -> VisionResult:
        return result_from_probs(self.probabilities(image_bytes))
