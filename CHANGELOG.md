# Changelog

All notable changes to this repository. Dates are ISO 8601.

## [0.1.1] - 2026-11-10

### Added
- `urbiva-ai benchmark`: CPU latency measurement against the 100 ms p50 target.
- Training refuses to start without a recorded dataset version.
- Experiment record template, concrete GPU and CPU allocation plan, licence acceptance rules.
- Dependabot configuration.

## [0.1.0] - 2026-11-10

### Added
- UVA-Net S10 architecture with a hard 10M parameter check (3,963,191 parameters at width 1.0).
- Configuration loader with validation of every research constraint.
- Dataset manifest and provenance validation, grouped deterministic splits, preprocessing that removes metadata.
- Training loop with seeding, warmup and cosine schedule, class weights, mixed precision, early stopping on false-positive rate, and a GPU-hour budget guard.
- Metrics: confusion matrix, per-class report, barrier rates, threshold at target false-positive rate, expected calibration error, evaluation report.
- Inference: app-compatible contract, predictor, ONNX export, local POST /api/vision server.
- `urbiva-ai` command line tool, test suite and CI.
- Urbiva Proprietary License 1.0 and NOTICE.

### Not included
- Datasets, trained weights and results.

## [0.0.0] - 2026-11-10

### Added
- Research documentation, provenance templates and repository structure.
