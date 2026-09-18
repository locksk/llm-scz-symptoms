"""Tokenization for standard and long-context encoders."""

from __future__ import annotations

import pandas as pd


def label_value(value) -> float:
    if value is None or value == "NA" or pd.isna(value):
        return -100.0
    return float(value)


def process_chunked(
    examples,
    tokenizer,
    text_column: str,
    label: str,
    max_length: int = 512,
    max_chunks: int = 8,
):
    input_batches: list[list[list[int]]] = []
    attention_batches: list[list[list[int]]] = []
    label_batches: list[list[float]] = []
    chunk_size = max_length - 2

    for row_index, text in enumerate(examples[text_column]):
        token_ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
            "input_ids"
        ]
        row_inputs: list[list[int]] = []
        row_masks: list[list[int]] = []
        for chunk_index in range(max_chunks):
            start = chunk_index * chunk_size
            prepared = tokenizer.prepare_for_model(
                token_ids[start : start + chunk_size],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            row_inputs.append(prepared["input_ids"])
            row_masks.append(prepared["attention_mask"])
        input_batches.append(row_inputs)
        attention_batches.append(row_masks)
        label_batches.append([label_value(examples[label][row_index])])

    return {
        "input_ids": input_batches,
        "attention_mask": attention_batches,
        "labels": label_batches,
    }


def process_long_context(
    examples,
    tokenizer,
    text_column: str,
    label: str,
    max_length: int = 4096,
):
    tokenized = tokenizer(
        examples[text_column],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        pad_to_multiple_of=64,
    )
    global_attention = [
        [1] + [0] * (len(input_ids) - 1) for input_ids in tokenized["input_ids"]
    ]
    labels = [[label_value(examples[label][index])] for index in range(len(examples[label]))]
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "global_attention_mask": global_attention,
        "labels": labels,
    }

