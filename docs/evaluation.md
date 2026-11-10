# Evaluation methodology

Status: **planned**. No results exist.

## Primary metrics

- **False-positive rate** on images without a barrier: the most important number for the product.
- **Precision and recall per class**, plus macro-averaged F1.
- **Calibration** (expected calibration error) of the confidence scores.

## Secondary metrics

- Latency per image on the target serving hardware.
- Parameter count and model file size.

## Protocol

1. Fix the test split before any training.
2. Choose the operating threshold on the validation split.
3. Report test metrics once per final model, with the dataset version, config hash and commit.
4. Report confidence intervals (for example by bootstrap) and the number of test images per class.
5. Compare against a documented baseline, including the current external vision API on the same test set, if its terms allow.

## Reporting rules

- Never report metrics without the config, dataset version and commit that produced them.
- Never round in our favour or omit classes that perform poorly.
- Publish failure cases alongside headline numbers.
