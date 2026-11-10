# Dataset strategy

Status: **planned**. No data has been downloaded or collected.

## Target

About 40,000 suitable annotated street-level images for the first version, with a long-term goal of a larger, multi-city dataset.

## Candidate sources (to evaluate)

| Source type | Considerations |
| --- | --- |
| Open street-level imagery datasets | Licence for commercial use, geographic coverage, existing labels |
| Public accessibility datasets | Label taxonomy match, size, image quality |
| Urbiva field collection (consented) | Pilot zone coverage, privacy review, annotation cost |
| Synthetic or augmented data | Domain gap; only as a supplement |

Every source needs a record in `datasets/` based on [templates/PROVENANCE.md](../datasets/templates/PROVENANCE.md) and an entry in [LICENSES.md](../datasets/LICENSES.md) before any image is used.

## Annotation

- Label set aligned with the app categories: `stairs`, `blocked_path`, `broken_surface`, `curb`, `obstruction`, `other`, `none`.
- Written guidelines with positive and negative examples per class.
- Double annotation on a sample to measure agreement; disagreements resolved and recorded.

## Privacy

- Blur faces and licence plates before annotation.
- Strip EXIF and GPS metadata.
- Do not store images of private interiors or identifiable people beyond what the licence and consent allow.

## Splits

- Train, validation and test splits fixed once and versioned.
- Split by location or source, not by random image, to avoid near-duplicate leakage.
- The test set is used only for final reporting.
