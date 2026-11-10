# Compute budget

Status: **allocation known, usage zero**.

| Resource | Amount | Notes |
| --- | --- | --- |
| GPU | about 10 GH200 GPU-hours | CSC allocation |
| CPU | about 20 to 30 CPU-hours | CSC allocation; how CPU time is billed (core-hours vs node-hours, billing units) still needs to be verified with CSC before planning preprocessing jobs |

## Implications

Ten GPU-hours is small. The plan must assume:

- short training runs on a carefully prepared dataset, not large searches;
- transfer learning instead of training from scratch;
- preprocessing (resizing, blurring, deduplication) done once and cached, ideally on CPU;
- a small number of documented experiments, each with a fixed budget.

## Draft allocation (to be confirmed)

| Activity | Share of GPU budget |
| --- | --- |
| Baseline and sanity checks | small |
| Main experiments | most |
| Final training of the selected configuration | reserved |
| Evaluation on the held-out test set | small, once |

Exact numbers will be set after a timing benchmark on a single batch. Record actual usage per experiment in `experiments/<id>/README.md`.

## Rules

- No job is submitted without a committed config and an experiment record.
- Log wall-clock time and GPU-hours for every run.
- Stop runs that exceed their budget; do not silently extend.
