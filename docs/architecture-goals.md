# Architecture goals for UVA-Net S10

Status: **implemented in code, not trained.** The design below exists in `src/urbiva_ai/models/uva_net.py`. Nothing has been trained or measured.

## Requirements

1. At most 10 million trainable parameters, counted with the classification head.
2. Inference latency at p50 of at most 100 ms per image on one modern CPU core at 320 x 320 input, measured with `urbiva-ai benchmark` (the result does not depend on training).
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

## Implemented design (version 0.1)

| Part | Choice | Why |
| --- | --- | --- |
| Stem | 3x3 convolution, stride 2, 16 channels, hard-swish | Cheap early downsampling |
| Body | 15 inverted residual blocks (kernels 3 and 5, expansion up to 6x, strides at 4 stages) | Strong accuracy per parameter on mobile hardware |
| Attention | Squeeze-and-excitation in 9 of 15 blocks | Channel reweighting at small cost |
| Head | 1x1 convolution to 6x channels, global pooling, 1024-unit layer, dropout, 7 logits | Compact classifier over the app taxonomy |
| Width multiplier | 0.25 to 1.5 within budget | One knob for the size and latency trade-off |
| Size check | `build_model` raises if parameters exceed 10,000,000 | The constraint cannot be broken by configuration |

Parameter counts computed from the architecture: width 0.5 gives 1,285,455; width 1.0 gives 3,963,191; width 1.5 gives 8,112,247. Width 1.75 gives 10,733,967 and is refused.

Training choices: AdamW with warmup and cosine decay, label smoothing 0.05, inverse-frequency class weights, mixed precision on GPU, gradient clipping, early stopping on the validation false-positive rate at the target operating point. Augmentation is horizontal flip, colour jitter and mild crops; no vertical flips, because stairs have a direction.
