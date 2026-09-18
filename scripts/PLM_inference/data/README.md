# Data contract

This repository contains no clinical data. Files under `samples/` are fabricated
fixtures and are not statistically meaningful.

`source.schema.json` describes the already summarized and labelled input CSV. It
requires `Study_ID`, `Case_note_summary`, and the eight configured binary-or-missing
labels. `strata.schema.json` describes the separate `Study_ID` and `sumstrat`
metadata. `prepared.schema.json` describes generated rows after the join.

Preparation normalizes IDs by converting them to strings, trimming whitespace,
and uppercasing. It keeps the first duplicate in each input, inner-joins the two
inputs, validates labels, and creates both schemes. Extra source columns are
preserved and must receive disclosure review.

Prepared files are:

```text
prepared/
├── Scheme_1_S1_Train/{train,dev,test}.csv
├── Scheme_2_S1_ValTest/{train,dev,test}.csv
├── split_statistics_detailed.csv
└── split_statistics_report.txt
```

Staging copies `dev.csv` to `validation.csv` and records hashes and row counts:

```text
staged/
├── Scheme_1_S1_Train/{train,validation,test}.csv
├── Scheme_2_S1_ValTest/{train,validation,test}.csv
└── staging_manifest.json
```

All real inputs and generated files remain restricted. Store them below
`MENTAL_LLM_DATA_ROOT` or another authorized private location. Never place real
record text, study IDs, prepared splits, predictions, or checkpoints under this
public `data/` directory.

