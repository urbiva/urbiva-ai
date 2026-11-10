"""Command line interface: `urbiva-ai <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from urbiva_ai import __version__


def _cmd_validate_config(a: argparse.Namespace) -> int:
    from urbiva_ai.config import load

    cfg = load(a.config)
    print(json.dumps({"valid": True, "experiment_id": cfg.experiment_id}, indent=2))
    return 0


def _cmd_validate_manifest(a: argparse.Namespace) -> int:
    from urbiva_ai.data.manifest import load_sources, read_manifest, validate

    counts = validate(read_manifest(a.manifest), load_sources(a.sources))
    print(json.dumps({"valid": True, "records": sum(counts.values()), "labels": dict(counts)}, indent=2))
    return 0


def _cmd_params(a: argparse.Namespace) -> int:
    from urbiva_ai.models.uva_net import UVANetS10, count_parameters

    n = count_parameters(UVANetS10(width_multiplier=a.width))
    print(json.dumps({"width_multiplier": a.width, "parameters": n, "within_budget": n <= 10_000_000}, indent=2))
    return 0


def _cmd_benchmark(a: argparse.Namespace) -> int:
    from urbiva_ai.models.benchmark import benchmark

    print(json.dumps(benchmark(a.width, a.size, a.iterations, a.threads), indent=2))
    return 0


def _cmd_train(a: argparse.Namespace) -> int:
    from urbiva_ai.config import load, require_dataset
    from urbiva_ai.data.dataset import BarrierDataset
    from urbiva_ai.data.manifest import load_sources, read_manifest, validate
    from urbiva_ai.training.trainer import train

    cfg = load(a.config)
    require_dataset(cfg)
    sources = load_sources(a.sources)
    tr = list(read_manifest(cfg.data.manifest))
    va = list(read_manifest(cfg.data.val_manifest))
    validate(tr, sources)
    validate(va, sources)
    size = cfg.model.input_size
    run = train(cfg, BarrierDataset(tr, cfg.data.image_root, size, True, cfg.training.seed),
                BarrierDataset(va, cfg.data.image_root, size, False), device=a.device)
    print(f"run directory: {run}")
    return 0


def _cmd_evaluate(a: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from urbiva_ai.data.dataset import BarrierDataset
    from urbiva_ai.data.manifest import read_manifest
    from urbiva_ai.evaluation.report import build_report, write_report
    from urbiva_ai.inference.predictor import Predictor
    from urbiva_ai.training.trainer import predict

    p = Predictor(a.checkpoint, device=a.device)
    recs = list(read_manifest(a.manifest or p.cfg.data.test_manifest))
    ds = BarrierDataset(recs, p.cfg.data.image_root, p.size, False)
    probs, labels = predict(p.model, DataLoader(ds, batch_size=64), torch.device(a.device))
    rep = build_report(probs, np.asarray(labels), p.cfg.evaluation.target_false_positive_rate,
                       p.cfg.evaluation.calibration_bins)
    j, md = write_report(rep, a.out)
    print(f"wrote {j} and {md}")
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    from urbiva_ai.inference.export import export_onnx

    print(f"wrote {export_onnx(a.checkpoint, a.out)}")
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    from urbiva_ai.inference.server import serve

    serve(a.host, a.port, a.checkpoint)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="urbiva-ai", description="UVA-Net S10 research tools.")
    p.add_argument("--version", action="version", version=f"urbiva-ai {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("validate-config", help="validate an experiment configuration")
    s.add_argument("config", type=Path)
    s.set_defaults(func=_cmd_validate_config)

    s = sub.add_parser("validate-manifest", help="validate a manifest against the source registry")
    s.add_argument("manifest", type=Path)
    s.add_argument("--sources", type=Path, default=Path("datasets/sources.json"))
    s.set_defaults(func=_cmd_validate_manifest)

    s = sub.add_parser("params", help="count model parameters (requires torch)")
    s.add_argument("--width", type=float, default=1.0)
    s.set_defaults(func=_cmd_params)

    s = sub.add_parser("benchmark", help="measure CPU inference latency (requires torch)")
    s.add_argument("--width", type=float, default=1.0)
    s.add_argument("--size", type=int, default=320)
    s.add_argument("--iterations", type=int, default=50)
    s.add_argument("--threads", type=int, default=1)
    s.set_defaults(func=_cmd_benchmark)

    s = sub.add_parser("train", help="train UVA-Net S10 (requires torch and data)")
    s.add_argument("config", type=Path)
    s.add_argument("--sources", type=Path, default=Path("datasets/sources.json"))
    s.add_argument("--device", default=None)
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("evaluate", help="evaluate a checkpoint on a manifest")
    s.add_argument("checkpoint", type=Path)
    s.add_argument("--manifest", type=Path, default=None)
    s.add_argument("--out", type=Path, default=Path("reports"))
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("export", help="export a checkpoint to ONNX")
    s.add_argument("checkpoint", type=Path)
    s.add_argument("--out", type=Path, default=Path("models/uva-net-s10.onnx"))
    s.set_defaults(func=_cmd_export)

    s = sub.add_parser("serve", help="serve POST /api/vision locally")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--checkpoint", default=None)
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
