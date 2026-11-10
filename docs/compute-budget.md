# Compute budget

Status: **allocation known, usage zero**.

| Resource | Amount | Notes |
| --- | --- | --- |
| GPU | about 10 GH200 GPU-hours | CSC allocation |
| CPU | about 20 to 30 CPU-hours | CSC allocation; plans use the lower bound of 20 CPU-hours, and the billing unit (core-hours or billing units) is confirmed with CSC before the first preprocessing job |

## Implications

Ten GPU-hours is small. The plan must assume:

- short training runs on a carefully prepared dataset, not large searches;
- transfer learning instead of training from scratch;
- preprocessing (resizing, blurring, deduplication) done once and cached, ideally on CPU;
- a small number of documented experiments, each with a fixed budget.

## Allocation plan

| Activity | GPU-hours | Config field |
| --- | --- | --- |
| Smoke test and throughput check on a small subset | 0.5 | `training.gpu_hour_budget: 0.5` |
| Baseline (width 1.0, 320 px) | 1.5 | `training.gpu_hour_budget: 1.5` |
| Main experiments (at most 3 runs of 1.5 GPU-hours) | 4.5 | one config per run |
| Final training of the selected configuration | 2.0 | `training.gpu_hour_budget: 2.0` |
| Evaluation on the held-out test set, once | 0.5 | `urbiva-ai evaluate` |
| Reserve for a failed run | 1.0 | used only with a written reason in the experiment record |
| **Total** | **10.0** | |

| Activity | CPU-hours |
| --- | --- |
| Blurring, resizing and deduplication, once, cached | 12 |
| Manifest building and validation | 2 |
| Evaluation reports and exports | 2 |
| Reserve | 4 |
| **Total (lower bound of the allocation)** | **20** |

The trainer enforces each run's share: it stops before `gpu_hour_budget` is used, with a 5 % safety margin. Record actual usage in every experiment record (`experiments/TEMPLATE.md`).

## Rules

- No job is submitted without a committed config and an experiment record.
- Log wall-clock time and GPU-hours for every run.
- Stop runs that exceed their budget; do not silently extend.
