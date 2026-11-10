# Model card: UVA-Net S10 (<experiment id>)

Fill every field from committed artifacts only. Leave a field as "not measured" rather than estimating.

| Field | Value |
| --- | --- |
| Experiment id | |
| Code commit | |
| Configuration | `experiments/<id>/config.yaml` |
| Dataset version and manifest checksums | |
| Parameters | |
| Training compute (GPU-hours) | |
| Weights location and SHA-256 | stored outside Git |

## Intended use

Suggest a barrier category for a street photo in Urbiva Vision. A person always confirms. Not for navigation, safety or enforcement decisions on its own.

## Evaluation

Paste `report.md` produced by `urbiva-ai evaluate` on the test split. Include the operating point (threshold, false-positive rate, true-positive rate) and calibration error.

## Limitations and risks

Known failure cases, under-represented conditions (night, rain, snow, unusual surfaces), and any group or area where performance is lower.

## Data and privacy

Sources and licences from `datasets/LICENSES.md`; confirmation that faces and licence plates were blurred.
