# Experiments

One folder per run: `experiments/<YYYY-MM-DD>-<short-name>/`, containing a copy of the configuration and a `README.md` made from [TEMPLATE.md](TEMPLATE.md). The record is written before the job is submitted and completed when it ends, following [docs/reproducibility.md](../docs/reproducibility.md).

| Rule | Why |
| --- | --- |
| Commit the record and config before submitting | Every result can be traced to code and data |
| Copy numbers only from `report.json` | No hand-typed or estimated metrics |
| Record GPU-hours from the training log | Keeps the 10 GPU-hour allocation auditable |
