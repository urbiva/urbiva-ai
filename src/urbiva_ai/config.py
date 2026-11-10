"""Experiment configuration with validation of the UVA-Net S10 constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from urbiva_ai.labels import NUM_CLASSES

MAX_PARAMETERS = 10_000_000
"""Hard limit on trainable parameters, including the classification head."""

TOTAL_GPU_HOURS = 10.0
"""Total GPU budget for the research phase (GH200 GPU-hours)."""


class ConfigError(ValueError):
    """Raised when a configuration violates a rule."""


@dataclass
class ModelConfig:
    name: str = "uva-net-s10"
    width_multiplier: float = 1.0
    input_size: int = 320
    dropout: float = 0.2
    num_classes: int = NUM_CLASSES
    max_parameters: int = MAX_PARAMETERS


@dataclass
class DataConfig:
    manifest: str = "datasets/manifests/train.jsonl"
    val_manifest: str = "datasets/manifests/val.jsonl"
    test_manifest: str = "datasets/manifests/test.jsonl"
    image_root: str = "datasets/processed"
    dataset_version: str = "unset"
    num_workers: int = 4


@dataclass
class TrainingConfig:
    seed: int = 1337
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    mixed_precision: bool = True
    early_stopping_patience: int = 6
    gpu_hour_budget: float = 2.0
    num_gpus: int = 1
    output_dir: str = "runs"


@dataclass
class EvaluationConfig:
    primary_metric: str = "false_positive_rate"
    target_false_positive_rate: float = 0.05
    calibration_bins: int = 15


@dataclass
class ExperimentConfig:
    experiment_id: str = "unset"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any] | None, section: str):
    raw = raw or {}
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown keys in '{section}': {sorted(unknown)}")
    return cls(**raw)


def from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build and validate an ExperimentConfig from a plain dictionary."""
    top = set(raw) - {"experiment_id", "model", "data", "training", "evaluation"}
    if top:
        raise ConfigError(f"unknown top-level keys: {sorted(top)}")
    cfg = ExperimentConfig(
        experiment_id=str(raw.get("experiment_id", "unset")),
        model=_build(ModelConfig, raw.get("model"), "model"),
        data=_build(DataConfig, raw.get("data"), "data"),
        training=_build(TrainingConfig, raw.get("training"), "training"),
        evaluation=_build(EvaluationConfig, raw.get("evaluation"), "evaluation"),
    )
    validate(cfg)
    return cfg


def load(path: str | Path) -> ExperimentConfig:
    """Load a YAML configuration file and validate it."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return from_dict(raw)


def validate(cfg: ExperimentConfig) -> None:
    """Raise ConfigError if any rule is broken."""
    m, t, e = cfg.model, cfg.training, cfg.evaluation
    if m.num_classes != NUM_CLASSES:
        raise ConfigError(f"model.num_classes must be {NUM_CLASSES} to match the app taxonomy")
    if m.max_parameters > MAX_PARAMETERS:
        raise ConfigError(f"model.max_parameters may not exceed {MAX_PARAMETERS:,}")
    if not 0.25 <= m.width_multiplier <= 2.0:
        raise ConfigError("model.width_multiplier must be between 0.25 and 2.0")
    if m.input_size < 96 or m.input_size > 640 or m.input_size % 32:
        raise ConfigError("model.input_size must be a multiple of 32 between 96 and 640")
    if not 0.0 <= m.dropout < 1.0:
        raise ConfigError("model.dropout must be in [0, 1)")
    if t.epochs < 1 or t.batch_size < 1:
        raise ConfigError("training.epochs and training.batch_size must be positive")
    if t.learning_rate <= 0 or t.weight_decay < 0:
        raise ConfigError("training.learning_rate must be positive and weight_decay non-negative")
    if not 0.0 <= t.label_smoothing < 0.5:
        raise ConfigError("training.label_smoothing must be in [0, 0.5)")
    if t.gpu_hour_budget <= 0 or t.gpu_hour_budget > TOTAL_GPU_HOURS:
        raise ConfigError(f"training.gpu_hour_budget must be in (0, {TOTAL_GPU_HOURS}] GPU-hours")
    if t.num_gpus < 1:
        raise ConfigError("training.num_gpus must be at least 1")
    if not 0.0 < e.target_false_positive_rate < 1.0:
        raise ConfigError("evaluation.target_false_positive_rate must be in (0, 1)")
    if e.calibration_bins < 2:
        raise ConfigError("evaluation.calibration_bins must be at least 2")
