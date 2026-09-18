# Binary reproducibility notes

## Confirmed behavior

- Inputs begin with an already summarized and labelled CSV plus separate stratum
  metadata. Upstream record acquisition, summarization, annotation, exclusions, and
  de-identification were not recovered.
- IDs are converted to strings, trimmed, uppercased, and deduplicated independently
  with first-row retention before an inner join.
- The two named schemes use a 10,000-trial randomized greedy count matcher with
  NumPy `RandomState(42)`.
- Standard encoders use eight 512-token chunks and average chunk logits. Longformer
  and BigBird use one sequence of up to 4096 tokens. BigBird uses `use_fast=false`.
- One checkpoint is trained for each label. Missing labels use sentinel `-100` and
  are masked from focal loss and metrics.
- The positive focal-loss alpha is one minus the observed positive proportion,
  clipped to `[0.01, 0.99]`; gamma is `2.0`.
- The pretrained backbone is loaded before classifier construction and then assigned
  to the classifier, matching the recovered initialization order.
- Validation selects saved checkpoints by AUC and thresholds by positive F1 on the
  grid `0.10, 0.15, ..., 0.90`; ties retain the first improving threshold.
- Evaluation applies a sigmoid, the saved per-label threshold, missing-label masks,
  and the recovered metric definitions. Probability columns and final tables round
  to four decimal places.
- AUC intervals use 1,000 replacement resamples with seed 42. One-class resamples
  are skipped, and indexed 2.5% and 97.5% bounds are reported.

## Configuration evidence

The historical profile records the five publication-facing encoder names, recovered
launcher batch sizes, context settings, and available per-run best-parameter JSON
artifacts for both schemes and all eight labels. It uses those fixed parameters
directly; it does not rerun parameter search. The GatorTron launcher used training
batch size 8 while the recovered evaluator used batch size 2, so both values are
preserved separately.

The staging command is a transparent reconstruction because the archived operation
that copied `dev.csv` to `validation.csv` was missing. Its manifest records file
hashes and row counts.

## Corrected evaluation boundary

Current inference omits labels from model forward calls and masks missing targets in
metric calculations. This avoids an invalid evaluation-loss path present in an older
snapshot while preserving probability decoding, thresholds, and metric formulas.
The exact BigBird environment, model revision, and historical rerun sequence remain
unresolved, so corrected outputs must not be presented as byte-identical historical
outputs.

## Remaining limitations

- No archived manifest proves the historical input bytes, split memberships, code
  revision, or full commands.
- The original parameter-search sampler was not explicitly seeded; its exact trial
  sequence and objective aggregation are unresolved.
- Most package, CUDA, driver, and hardware versions are unknown. Only
  `transformers==4.57.6` was recovered exactly from artifacts.
- Encoder revisions were not pinned to immutable provider commits and availability
  can change.
- Validation was used for checkpoint selection and threshold calibration.
- The split optimizer targets counts rather than label balance and depends on
  normalized input ordering.
- The semantics and governance of stratum values require custodian confirmation.
- Real training and evaluation have not been rerun for this public release.
- Publication citation, license, and disclosure approval remain pending.

