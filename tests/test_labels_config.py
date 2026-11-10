from pathlib import Path

import pytest

from urbiva_ai import labels
from urbiva_ai.config import ConfigError, from_dict, load

ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_matches_app():
    assert labels.CATEGORIES == ("stairs", "blocked_path", "broken_surface", "curb", "obstruction", "other", "none")
    assert labels.NUM_CLASSES == 7
    assert labels.is_barrier("stairs") and not labels.is_barrier("none")
    with pytest.raises(ValueError):
        labels.index_of("ramp")


def test_template_config_is_valid():
    cfg = load(ROOT / "configs" / "uva_net_s10.template.yaml")
    assert cfg.model.max_parameters == 10_000_000
    assert cfg.model.num_classes == 7


@pytest.mark.parametrize("patch,msg", [
    ({"model": {"max_parameters": 10_000_001}}, "max_parameters"),
    ({"model": {"num_classes": 5}}, "num_classes"),
    ({"model": {"input_size": 300}}, "input_size"),
    ({"training": {"gpu_hour_budget": 11}}, "gpu_hour_budget"),
    ({"training": {"label_smoothing": 0.6}}, "label_smoothing"),
    ({"evaluation": {"target_false_positive_rate": 0}}, "target_false_positive_rate"),
    ({"model": {"unknown": 1}}, "unknown keys"),
    ({"extra": 1}, "unknown top-level"),
])
def test_invalid_configs_are_rejected(patch, msg):
    with pytest.raises(ConfigError, match=msg):
        from_dict(patch)


def test_training_requires_a_dataset_version():
    from urbiva_ai.config import require_dataset

    cfg = load(ROOT / "configs" / "uva_net_s10.template.yaml")
    with pytest.raises(ConfigError, match="dataset_version"):
        require_dataset(cfg)
    cfg.data.dataset_version = "zone-a-2027-01"
    require_dataset(cfg)


def test_experiment_id_format():
    with pytest.raises(ConfigError, match="experiment_id"):
        from_dict({"experiment_id": "Bad Id!"})
