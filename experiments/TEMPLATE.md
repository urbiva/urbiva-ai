# Experiment record

| Field | Value |
| --- | --- |
| Experiment id | must match `experiment_id` in the config |
| Date | YYYY-MM-DD |
| Owner | GitHub handle |
| Goal | one sentence: what question this run answers |
| Config | `experiments/<folder>/config.yaml` |
| Code commit | full SHA |
| Dataset version and manifest SHA-256 | from `datasets/` |
| Hardware | GPU model and count |
| GPU-hour budget | from `training.gpu_hour_budget` |

## Result

Fill after the run, copying from `runs/<run>/log.jsonl` and `report.json` only.

| Field | Value |
| --- | --- |
| GPU-hours used | |
| Stopped by | early stopping, budget guard or epoch limit |
| Validation false-positive rate at threshold | |
| Validation true-positive rate at threshold | |
| Macro F1 | |
| Calibration error | |

## Conclusion

What was learned, and what the next run changes.
