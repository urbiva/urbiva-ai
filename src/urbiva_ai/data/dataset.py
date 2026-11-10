"""PyTorch dataset over a validated manifest. Requires the 'torch' extra."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from urbiva_ai.data.manifest import Record
from urbiva_ai.data.transforms import augment, letterbox, load_image, to_array
from urbiva_ai.labels import index_of

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("urbiva_ai.data.dataset requires PyTorch: pip install 'urbiva-ai[torch]'") from exc


class BarrierDataset(Dataset):
    def __init__(self, records: Sequence[Record], image_root: str | Path, size: int, train: bool, seed: int = 0):
        self.records = list(records)
        self.root = Path(image_root)
        self.size = size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        im = load_image((self.root / r.path).read_bytes())
        if self.train:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            rng = np.random.default_rng((self.seed, i, wid, int(torch.initial_seed() % (2**31))))
            im = augment(im, rng)
        x = torch.from_numpy(to_array(letterbox(im, self.size)))
        return x, index_of(r.label)

    def class_weights(self, num_classes: int) -> torch.Tensor:
        """Inverse-frequency weights, normalised to mean 1, for imbalanced classes."""
        counts = np.bincount([index_of(r.label) for r in self.records], minlength=num_classes).astype(np.float64)
        w = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
        if w.sum() > 0:
            w = w / w[w > 0].mean()
        return torch.tensor(w, dtype=torch.float32)
