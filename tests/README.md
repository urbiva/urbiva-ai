# Tests

Run `pytest` from the repository root.

| File | Covers | Needs torch |
| --- | --- | --- |
| `test_labels_config.py` | Taxonomy, template configuration, every configuration rule | No |
| `test_metrics.py` | Confusion matrix, per-class report, barrier rates, threshold at target FPR, calibration, report | No |
| `test_contract_server.py` | App response contract, request validation, server without and with a model, rate limiting | No |
| `test_data.py` | Manifest and provenance validation, grouped splits, preprocessing, compute budget | No |
| `test_model.py` | Parameter count, budget enforcement, forward pass, a training step | Yes (skipped without torch, always run in CI) |
