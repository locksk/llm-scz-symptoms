# Mental-LLM independent binary pipeline

## 1. Project overview

This repository is a concise public release for predicting eight binary SAPS/SANS
symptom indicators from already summarized and labelled text. It covers safe data
preparation, two split schemes, one-label-at-a-time model training, checkpoint
inference, prediction aggregation, metrics, bootstrap intervals, and CSV reports.
No clinical data, model weights, checkpoints, predictions, or historical results
are distributed.

## 2. Independent binary task formulation

Each configured symptom is a separate binary task with its own encoder checkpoint,
positive-class weight, validation threshold, and output directory. Missing labels
are encoded as `-100`, excluded from loss and metrics, and retained in prediction
files. Label probabilities are combined only after every one-label inference run.

## 3. Supported encoders

- `mental/mental-bert-base-uncased`
- `AIMH/mental-bert-large-uncased`
- `UFNLP/gatortron-base`
- `AIMH/mental-longformer-base-4096`
- `google/bigbird-roberta-large`

Model artifacts are not included. Access depends on provider availability, terms,
network access, and any credential supplied through the environment.

## 4. Data schemes

- `Scheme_1_S1_Train`: all S1 rows train the model; whole non-S1 `sumstrat`
  groups are assigned approximately two-thirds to development and one-third to test.
- `Scheme_2_S1_ValTest`: all non-S1 rows train the model; S1 IDs are assigned
  approximately 50/50 to development and test.

Both use the recovered 10,000-trial greedy count matcher with NumPy
`RandomState(42)`. Assignment targets row counts, not label balance.

## 5. Repository structure

```text
.
├── configs/{historical,portable,smoke}/
├── data/{schemas,samples}/
├── docs/reproducibility.md
├── scripts/
│   ├── prepare_data.sh
│   ├── train_binary.sh
│   ├── infer_binary.sh
│   ├── evaluate_binary.sh
│   └── run_binary_pipeline.sh
├── src/mental_llm/{data,binary,evaluation,reporting}/
└── tests/
```

## 6. Installation

Python 3.11 or 3.12 and Bash 4 or later are supported.

```bash
/path/to/repository/scripts/setup.sh --dev
source /path/to/repository/.venv/bin/activate
```

The setup command installs packages and therefore requires access to the selected
package indexes. Smoke and unit tests do not download encoders.

## 7. Private data requirements

Set a private runtime root outside this repository:

```bash
export MENTAL_LLM_DATA_ROOT=/path/to/private/runtime_data
```

The source CSV requires `Study_ID`, `Case_note_summary`, and all eight configured
labels as `0`, `1`, or missing. The stratum CSV requires `Study_ID` and `sumstrat`.
See `data/schemas/` for machine-readable contracts.

## 8. Data preparation

```bash
/path/to/repository/scripts/prepare_data.sh prepare \
  --config /path/to/repository/configs/portable/binary.json \
  --source "$MENTAL_LLM_DATA_ROOT/source/summarized_labelled.csv" \
  --strata "$MENTAL_LLM_DATA_ROOT/source/strata.csv" \
  --output-dir "$MENTAL_LLM_DATA_ROOT/prepared"
```

IDs are string-normalized, trimmed, uppercased, and deduplicated with first-row
retention in each input. Preparation inner-joins the inputs, validates labels and
partitions, and writes both schemes. Use `--validation-only` to write nothing.

## 9. Staging

```bash
/path/to/repository/scripts/prepare_data.sh stage \
  --config /path/to/repository/configs/portable/binary.json \
  --prepared-dir "$MENTAL_LLM_DATA_ROOT/prepared" \
  --staged-dir "$MENTAL_LLM_DATA_ROOT/staged"
```

Staging copies `dev.csv` to `validation.csv`, validates six split files, and writes
a row-count and SHA-256 manifest without record identifiers.

## 10. Binary training

Train all eight labels for one scheme and encoder:

```bash
/path/to/repository/scripts/train_binary.sh \
  --config /path/to/repository/configs/portable/binary.json \
  --scheme Scheme_1_S1_Train \
  --model-name mental/mental-bert-base-uncased \
  --all-labels
```

Use `--label SAPS_Global_Hals` instead to train one task. Training computes the
positive focal-loss alpha from observed training labels, masks missing targets,
selects the best saved epoch by validation AUC, and selects a positive-F1 threshold
from `0.10` through `0.90` in `0.05` increments.

## 11. Binary inference

```bash
/path/to/repository/scripts/infer_binary.sh \
  --config /path/to/repository/configs/portable/binary.json \
  --scheme Scheme_1_S1_Train \
  --model-name mental/mental-bert-base-uncased
```

Inference loads every label checkpoint separately and predicts the validation and
test splits. It never trains or changes checkpoints.

## 12. Per-label checkpoint structure

```text
$MENTAL_LLM_DATA_ROOT/
└── checkpoints/
    └── binary/
        └── <scheme>/
            └── <safe-model>/
                └── <label>/
                    └── final_model/
```

Each `final_model/` contains the model, tokenizer, and `thresholds.json`. Its parent
contains a private run manifest.

## 13. Prediction aggregation

Each label produces `per_label/<label>/{validation,test}_predictions.csv` with ID,
actual value, decoded prediction, and probability. Aggregation verifies identical ID
ordering, then creates one configured-label-order details file per split.

## 14. Evaluation

After inference has completed for all configured schemes and encoders:

```bash
/path/to/repository/scripts/evaluate_binary.sh \
  --config /path/to/repository/configs/portable/binary.json
```

Reports contain precision, sensitivity, specificity, positive F1, macro F1,
accuracy, balanced accuracy, AUC, MCC, predicted cases, actual cases, and N.

## 15. Bootstrap confidence intervals

AUC intervals use 1,000 replacement resamples and NumPy `RandomState(42)`. Samples
containing one observed class are skipped. The bounds use the recovered indexed
2.5th and 97.5th percentiles. A one-class label receives AUC and bounds of `0.5`.

## 16. Output files

```text
$MENTAL_LLM_DATA_ROOT/
├── prepared/<scheme>/{train,dev,test}.csv
├── staged/<scheme>/{train,validation,test}.csv
├── checkpoints/binary/<scheme>/<safe-model>/<label>/final_model/
├── outputs/binary/<scheme>/<safe-model>/
│   ├── per_label/<label>/{validation,test}_predictions.csv
│   ├── {validation,test}_predictions_details.csv
│   └── {validation,test}_metrics.csv
└── reports/binary/
    ├── per_model_reports/
    ├── <scheme>_leaderboard.csv
    ├── combined_leaderboard.csv
    └── binary_final_table.csv
```

All generated paths are private and may contain sensitive record-level values.

## 17. Synthetic smoke test

The offline smoke path uses only fabricated CSVs and dry-run model commands:

```bash
/path/to/repository/scripts/run_binary_pipeline.sh --smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/repository/src \
  python -m pytest -q /path/to/repository/tests
```

It does not download a model or run training or inference.

## 18. Historical configurations

`configs/historical/binary.json` preserves recovered context lengths, chunk counts,
batch sizes, BigBird `use_fast=false`, and the fixed best parameters found for each
scheme, encoder, and label. Longformer and BigBird use 4096 tokens. These values are
artifact-derived; they do not recreate the original search process.

## 19. Portable configurations

`configs/portable/binary.json` retains model-specific tokenization and sequence
lengths while reducing device batch sizes to one. It is a resource-conscious
execution profile, not a historical default. `configs/smoke/binary.json` is only for
offline validation.

## 20. Reproducibility limitations

The summarized/labelled source-creation stage, exact historical environment, model
revisions, input hashes, and complete command manifests were not recovered. The
historical parameter search was not explicitly seeded and is not claimed to be
reproducible. Training-launcher provenance drift and older BigBird evaluation
behavior remain uncertain. See `docs/reproducibility.md`.

## 21. Privacy and governance

Use only institutionally authorized data and storage. Do not publish record text,
study IDs, strata, prepared splits, predictions, logs, checkpoints, caches, or
populated environment files. Aggregate tables can also require disclosure review.
Users remain responsible for ethics approval, data-use agreements, retention rules,
access controls, and third-party model terms.

## 22. Citation

Citation information is pending. Replace this placeholder with confirmed publication
metadata and a versioned software archive identifier before release.

## 23. License status

License status is pending. No software license is supplied, and no permissions are
granted beyond those provided by applicable law. Confirm institutional ownership,
publication approval, and third-party terms before adding a license.

