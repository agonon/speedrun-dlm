"""Small data and reporting helpers for the HellaSwag finish-line evaluator."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .checkpoint_utils import tokenizer


REFERENCE_MC_BATCH_SIZE = 16


@dataclass(frozen=True)
class Candidate:
    example_index: int
    choice_index: int
    example_id: str
    ids: tuple[int, ...]
    target_start: int
    target_end: int
    normalization_text: str

    @property
    def target_tokens(self) -> int:
        return self.target_end - self.target_start

    @property
    def characters(self) -> int:
        return len(self.normalization_text)

    @property
    def utf8_bytes(self) -> int:
        return len(self.normalization_text.encode("utf-8"))


def stable_uint64(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    kwargs = {"encoding": "utf-8"}
    with opener(path, "rt", **kwargs) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for row in rows:
                    zipped.write((json.dumps(row, sort_keys=True) + "\n").encode())
    else:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def encode_continuation(
    prompt: str, continuation: str, normalization_text: str, max_length: int
) -> tuple[tuple[int, ...], int, int]:
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + continuation)
    common_prefix = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common_prefix += 1
    target_ids = full_ids[common_prefix:]
    if not target_ids:
        raise ValueError("continuation encoded to zero target tokens")
    if len(target_ids) > max_length - 1:
        raise ValueError("continuation is longer than the evaluation context")
    prompt_budget = max_length - len(target_ids) - 1
    retained_prompt = full_ids[:common_prefix]
    retained_prompt = retained_prompt[-prompt_budget:] if prompt_budget else []
    ids = (int(tokenizer.eot_token), *map(int, retained_prompt), *map(int, target_ids))
    if not normalization_text:
        raise ValueError("normalization text must be non-empty")
    return ids, 1 + len(retained_prompt), len(ids)


def load_rows(path: Path, start: int, limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if start < 0 or start > len(rows):
        raise ValueError(f"start={start} outside dataset with {len(rows)} rows")
    stop = len(rows) if limit <= 0 else min(len(rows), start + limit)
    selected = rows[start:stop]
    if not selected:
        raise ValueError("selected shard contains no examples")
    return selected


def build_candidates(
    rows: list[dict[str, Any]], max_length: int
) -> tuple[list[Candidate], list[list[int]], list[int]]:
    candidates: list[Candidate] = []
    groups: list[list[int]] = []
    labels: list[int] = []
    for example_index, row in enumerate(rows):
        choices = [str(value) for value in row["choices"]]
        label = int(row["label"])
        if label < 0 or label >= len(choices):
            raise ValueError(f"invalid label for {row['example_id']}")
        group = []
        for choice_index, choice in enumerate(choices):
            ids, target_start, target_end = encode_continuation(
                str(row["prompt"]), " " + choice, choice, max_length
            )
            group.append(len(candidates))
            candidates.append(Candidate(
                example_index=example_index,
                choice_index=choice_index,
                example_id=str(row["example_id"]),
                ids=ids,
                target_start=target_start,
                target_end=target_end,
                normalization_text=choice,
            ))
        groups.append(group)
        labels.append(label)
    return candidates, groups, labels


def reference_mask_counts(target_tokens: int, batch_size: int, start: int) -> np.ndarray:
    values = np.rint(np.linspace(
        float(start), start + (batch_size - 1) * (target_tokens / batch_size), num=batch_size
    )).astype(np.int64)
    return ((values - 1) % target_tokens) + 1


def summarize_multiple_choice(
    rows: list[dict[str, Any]],
    candidates: list[Candidate],
    groups: list[list[int]],
    labels: list[int],
    losses: np.ndarray,
    prefixes: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    example_results = []
    summaries = []
    normalizers = {
        "raw": np.ones(len(candidates), dtype=np.float64),
        "characters": np.asarray([c.characters for c in candidates], dtype=np.float64),
        "utf8_bytes": np.asarray([c.utf8_bytes for c in candidates], dtype=np.float64),
        "tokens": np.asarray([c.target_tokens for c in candidates], dtype=np.float64),
    }
    for prefix in prefixes:
        estimated_nll = losses[:, :prefix].mean(axis=1)
        correct = {name: 0 for name in normalizers}
        for row, group, label in zip(rows, groups, labels, strict=True):
            predictions = {}
            scores = {}
            for name, denominator in normalizers.items():
                values = -estimated_nll[group] / denominator[group]
                predictions[name] = int(np.argmax(values))
                scores[name] = [float(value) for value in values]
                correct[name] += int(predictions[name] == label)
            example_results.append({
                "example_id": str(row["example_id"]),
                "mc_samples": prefix,
                "label": label,
                "predictions": predictions,
                "scores": scores,
            })
        summaries.append({
            "mc_samples": prefix,
            "examples": len(rows),
            "chance_accuracy": 0.25,
            **{f"accuracy_{name}": correct[name] / len(rows) for name in normalizers},
        })
    return example_results, summaries
