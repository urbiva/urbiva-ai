"""CPU latency benchmark for UVA-Net S10. Requires PyTorch.

Measures single-image forward latency with untrained weights. Latency does not
depend on weight values, so the result is valid before training. Accuracy is
never measured here.
"""

from __future__ import annotations

import platform
import statistics
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("urbiva_ai.models.benchmark requires PyTorch: pip install 'urbiva-ai[torch]'") from exc

from urbiva_ai.models.uva_net import build_model, count_parameters

LATENCY_TARGET_MS = 100.0
"""Target: p50 single-image latency on one CPU core at 320 x 320 input."""


@torch.no_grad()
def benchmark(width: float = 1.0, size: int = 320, iterations: int = 50, threads: int = 1, warmup: int = 5) -> dict:
    if iterations < 1 or warmup < 0 or threads < 1:
        raise ValueError("iterations and threads must be positive, warmup non-negative")
    torch.set_num_threads(threads)
    model = build_model(width_multiplier=width).eval()
    x = torch.zeros(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    return {
        "width_multiplier": width,
        "input_size": size,
        "parameters": count_parameters(model),
        "threads": threads,
        "iterations": iterations,
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(p95, 2),
        "target_ms_p50": LATENCY_TARGET_MS,
        "meets_target": p50 <= LATENCY_TARGET_MS,
        "machine": f"{platform.machine()} {platform.processor() or platform.system()}",
        "torch": torch.__version__,
    }
