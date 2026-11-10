# Reproducibility standards

Every experiment must be reproducible from the repository alone, given access to the recorded dataset version.

## Required for each experiment

- A config file in `configs/` committed before the run.
- An experiment folder `experiments/<YYYY-MM-DD>-<short-name>/` with a README: goal, config path, dataset version, commit hash, hardware, GPU-hours used, results, conclusions.
- Fixed random seeds for Python, NumPy and the framework; note any non-determinism.
- Exact software environment (lock file or container image digest).

## Data versioning

- Datasets are identified by source, version and checksum, recorded in `datasets/`.
- Processed datasets are rebuilt from scripts; they are never edited by hand.

## Artifacts

- Model weights and large outputs are stored outside Git (object storage with access control) and referenced by checksum from the model card in `models/`.
- No weights, datasets or credentials are committed.
