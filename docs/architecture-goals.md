# Architecture goals for UVA-Net S10

Status: **planned**. Nothing in this document has been implemented or measured.

## Requirements

1. At most 10 million trainable parameters, counted with the classification head.
2. Real-time capable inference on a single modern CPU core or a small GPU; exact latency targets will be set after a baseline exists.
3. Input: one RGB street-level photo, re-encoded by the app to JPEG with the long side at most 1280 px.
4. Output: a probability per barrier category plus a calibrated "no barrier" score, so the app can abstain and fall back to manual classification.
5. Calibrated confidence, because the product decision depends on keeping false positives low.

## Design space to explore

- Efficient convolutional backbones and small vision transformers that fit the parameter budget.
- Transfer learning from openly licensed pretrained weights, only if the licence allows commercial use and the provenance is recorded.
- Input resolution versus accuracy versus latency trade-offs.
- Class imbalance handling (barrier categories are rare compared with clear sidewalks).
- Optional auxiliary heads (localisation of the barrier) if they improve classification without breaking the budget.

## Questions to answer before implementation

- Which label taxonomy maps cleanly to the app's categories and to available datasets?
- What operating point (false-positive rate) do planners and residents accept?
- Is single-label classification enough, or do scenes need multiple labels?
- What hardware will serve the model in production?

## Non-goals for the first version

- Video, 3D or depth estimation.
- Measuring slopes or widths from a single photo (the app already tells users this is not reliable).
- Replacing human confirmation.
