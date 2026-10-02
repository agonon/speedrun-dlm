"""Materialize the complete public HellaSwag validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import datasets

from ._hellaswag import write_jsonl


DATASET = "Rowan/hellaswag"
DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
HARNESS_REVISION = "4e7e0d47f33bc71070c1d38394bafbb52b117163"


def preprocess(text: str) -> str:
    """Match lm-evaluation-harness HellaSwag text preprocessing."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = datasets.load_dataset(
        DATASET,
        split="validation",
        revision=DATASET_REVISION,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    rows = []
    for index, raw in enumerate(split):
        context = str(raw["ctx_a"]) + " " + str(raw["ctx_b"]).capitalize()
        prompt = preprocess(str(raw["activity_label"]) + ": " + context)
        rows.append({
            "schema": "speedrun-dlm-hellaswag-data-v1",
            "example_id": f"hellaswag:validation:{index}",
            "prompt": prompt,
            "choices": [preprocess(str(value)) for value in raw["endings"]],
            "label": int(raw["label"]),
        })
    write_jsonl(args.output, rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "schema": "speedrun-dlm-hellaswag-manifest-v1",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "lm_evaluation_harness_revision": HARNESS_REVISION,
        "examples": len(rows),
        "sha256": digest,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
