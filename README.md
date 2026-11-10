<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/images/header-urbiva-ai-light.png">
  <img alt="Urbiva AI Research: UVA-Net S10, planned and not trained" src="docs/images/header-urbiva-ai-dark.png" width="100%">
</picture>

# urbiva-ai

**Research and development of lightweight urban accessibility computer vision models.**

This repository is the home of **UVA-Net S10** (Urban Vision Accessibility Network, small, about 10 million parameters), a planned model that recognises accessibility barriers in street photos and runs locally, without depending on an external vision API.

> **Status: research setup only.** No model has been designed in code, trained or evaluated. There are no weights, no datasets and no results in this repository. Every number below is a constraint or a target, never a measurement.

## Why

Urbiva Vision in `urbiva-app` currently sends a re-encoded photo to an external vision model and always asks a person to confirm the result. A small local model would:

- keep photos on infrastructure we control,
- give predictable latency and cost,
- let us tune for a **low false-positive rate**, because a wrongly reported barrier sends people on longer routes.

## Task definition (draft)

Single-image classification of the barrier category on a sidewalk or crossing, aligned with the categories already used in the app (`src/lib/vision/categories.ts`):

`stairs` · `blocked_path` · `broken_surface` · `curb` · `obstruction` · `other` · `none`

Detection or segmentation heads are open research questions (see [docs/architecture-goals.md](docs/architecture-goals.md)).

## Constraints and targets

| Item | Value | Type |
| --- | --- | --- |
| Model size | at most 10 million parameters | Hard constraint |
| GPU budget | about 10 GH200 GPU-hours | Allocation (CSC) |
| CPU budget | about 20 to 30 CPU-hours | Allocation (CSC); billing interpretation still to be verified |
| Initial dataset | about 40,000 suitable annotated images | Target |
| Longer term | larger, multi-city dataset | Ambition |
| Quality | low false-positive rate; strong precision and recall | Goal, thresholds not yet set |
| Serving | fast local inference, no external vision API | Goal |

Details: [docs/compute-budget.md](docs/compute-budget.md).

## Repository structure

```
configs/        Experiment configuration files (templates, nothing has been run)
datasets/       Provenance and licensing records; no image data is committed
docs/           Architecture goals, compute budget, dataset strategy, evaluation, reproducibility, roadmap
experiments/    One folder per experiment with config, notes and results (empty)
models/         Model cards only; weights are never committed
notebooks/      Exploration notebooks (empty)
scripts/        Entry points for data preparation, training, evaluation (empty)
src/urbiva_ai/  Package skeleton: data, models, training, evaluation, inference
tests/          Unit tests (empty)
```

## Documentation

| Document | Purpose |
| --- | --- |
| [Architecture goals](docs/architecture-goals.md) | Design space for a sub-10M model and the questions to answer |
| [Compute budget](docs/compute-budget.md) | GPU and CPU allocation, how to spend it, what to verify |
| [Dataset strategy](docs/dataset-strategy.md) | Sources, annotation, privacy, splits |
| [Evaluation methodology](docs/evaluation.md) | Metrics, false-positive focus, test protocol |
| [Reproducibility standards](docs/reproducibility.md) | Seeds, configs, environment, reporting |
| [Research roadmap](docs/roadmap.md) | Milestones from data audit to integration |
| [Dataset provenance template](datasets/templates/PROVENANCE.md) | One record per dataset source |
| [Dataset licence register](datasets/LICENSES.md) | Licence of every source, checked before use |

## Research principles

- **No fabricated results.** Only report metrics produced by a committed config on a recorded dataset version.
- **Provenance first.** No image enters training without a provenance record and a compatible licence.
- **Privacy by design.** Faces and licence plates are blurred before annotation; location metadata is removed.
- **Human in the loop.** The app keeps human confirmation even after a local model exists.

## Status

| | |
| --- | --- |
| Phase | Repository and documentation setup |
| Code | Package skeleton only |
| Data | None collected |
| Training | Not started |

## Contributing

Use `feat/*`, `fix/*`, `docs/*` and `chore/*` branches and open pull requests into `main`. Never commit datasets, weights, credentials or personal data. See the organization-wide contributing guide in `urbiva/.github`.

## License

Private repository. All rights reserved by Urbiva. Third-party dataset licences are tracked in [datasets/LICENSES.md](datasets/LICENSES.md).
