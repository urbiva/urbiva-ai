"""Training loop for UVA-Net S10. Requires PyTorch.

Features: deterministic seeding, AdamW with warmup and cosine decay, label
smoothing, class weights, mixed precision on CUDA, early stopping on the
validation false positive rate at the target operating point, checkpoints,
a JSON log, and a hard stop before the GPU-hour budget runs out.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("urbiva_ai.training.trainer requires PyTorch: pip install 'urbiva-ai[torch]'") from exc

from urbiva_ai.config import ExperimentConfig
from urbiva_ai.evaluation import metrics as M
from urbiva_ai.models.uva_net import build_model, count_parameters
from urbiva_ai.training.budget import ComputeBudget


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 1 % of the base rate."""
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
    if not probs:
        return np.zeros((0, 0)), np.zeros((0,), dtype=np.int64)
    return np.concatenate(probs), np.concatenate(labels)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_macro_f1: float
    val_fpr_at_threshold: float
    val_tpr_at_threshold: float
    threshold: float
    gpu_hours_used: float


def evaluate_split(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> tuple[float, float, float, float]:
    cm = M.confusion_matrix(labels, probs.argmax(1))
    macro = M.classification_report(cm).macro_f1
    scores = M.barrier_scores(probs)
    is_b = labels != M.NONE_INDEX
    thr = M.threshold_at_fpr(scores, is_b, target_fpr)
    pred_b = scores >= thr
    fp = int(np.sum(pred_b & ~is_b))
    tn = int(np.sum(~pred_b & ~is_b))
    tp = int(np.sum(pred_b & is_b))
    fn = int(np.sum(~pred_b & is_b))
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    return macro, fpr, tpr, thr


def train(cfg: ExperimentConfig, train_set, val_set, device: str | None = None) -> Path:
    """Train and return the run directory. Writes config.json, log.jsonl, best.pt and last.pt."""
    t, m, e = cfg.training, cfg.model, cfg.evaluation
    seed_everything(t.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = Path(t.output_dir) / f"{cfg.experiment_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    model = build_model(m.num_classes, m.width_multiplier, m.dropout, m.max_parameters).to(dev)
    g = torch.Generator().manual_seed(t.seed)
    tl = DataLoader(train_set, batch_size=t.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                    pin_memory=dev.type == "cuda", drop_last=True, generator=g)
    vl = DataLoader(val_set, batch_size=t.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    weights = train_set.class_weights(m.num_classes).to(dev) if hasattr(train_set, "class_weights") else None
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=t.label_smoothing)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=t.learning_rate)
    total_steps = t.epochs * max(1, len(tl))
    warmup_steps = t.warmup_epochs * max(1, len(tl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warmup_steps))
    use_amp = t.mixed_precision and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    budget = ComputeBudget(t.gpu_hour_budget, t.num_gpus)
    budget.start()

    best_fpr, best_tpr, patience = math.inf, -1.0, 0
    log = (run / "log.jsonl").open("a", encoding="utf-8")
    log.write(json.dumps({"event": "start", "parameters": count_parameters(model), "device": str(dev)}) + "\n")
    epoch_hours = 0.0
    try:
        for epoch in range(1, t.epochs + 1):
            if not budget.can_continue(epoch_hours):
                log.write(json.dumps({"event": "budget_stop", "epoch": epoch,
                                      "gpu_hours_used": budget.used_gpu_hours}) + "\n")
                break
            t0 = time.monotonic()
            model.train()
            running, n = 0.0, 0
            for x, y in tl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += float(loss.detach()) * x.size(0)
                n += x.size(0)
            probs, labels = predict(model, vl, dev)
            macro, fpr, tpr, thr = evaluate_split(probs, labels, e.target_false_positive_rate)
            epoch_hours = (time.monotonic() - t0) / 3600.0
            res = EpochResult(epoch, running / max(1, n), macro, fpr, tpr, thr, budget.used_gpu_hours)
            log.write(json.dumps({"event": "epoch", **res.__dict__}) + "\n")
            log.flush()
            state = {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, "threshold": thr}
            torch.save(state, run / "last.pt")
            improved = fpr < best_fpr - 1e-6 or (abs(fpr - best_fpr) <= 1e-6 and tpr > best_tpr)
            if improved:
                best_fpr, best_tpr, patience = fpr, tpr, 0
                torch.save(state, run / "best.pt")
            else:
                patience += 1
                if patience >= t.early_stopping_patience:
                    log.write(json.dumps({"event": "early_stop", "epoch": epoch}) + "\n")
                    break
    finally:
        log.write(json.dumps({"event": "end", "gpu_hours_used": budget.used_gpu_hours}) + "\n")
        log.close()
    return run
