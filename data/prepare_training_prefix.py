#!/usr/bin/env python3
"""Materialize batch-invariant FineWeb shards over one exact token prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
MAGIC = 20240520
VERSION = 1
MAX_HEADER_TOKEN_COUNT = 2**31 - 1


def inspect_shard(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(HEADER_BYTES), dtype="<i4").copy()
    if len(header) != HEADER_INTS or int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(f"invalid FineWeb token shard: {path}")
    count = int(header[2])
    if path.stat().st_size != HEADER_BYTES + count * 2:
        raise ValueError(f"token count/size mismatch in {path}")
    return header, count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_global_range(
    inspected: list[tuple[Path, np.ndarray, int]],
    start: int,
    count: int,
    destination: np.memmap,
) -> list[dict[str, int | str]]:
    """Copy ``[start, start + count)`` from the concatenated source stream."""
    end = start + count
    source_start = 0
    written = 0
    segments: list[dict[str, int | str]] = []
    for path, _, source_count in inspected:
        source_end = source_start + source_count
        overlap_start = max(start, source_start)
        overlap_end = min(end, source_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - source_start
            take = overlap_end - overlap_start
            source = np.memmap(path, mode="r", dtype="<u2", offset=HEADER_BYTES, shape=(source_count,))
            for chunk_start in range(0, take, 16_000_000):
                chunk_end = min(take, chunk_start + 16_000_000)
                destination[written + chunk_start : written + chunk_end] = source[
                    local_start + chunk_start : local_start + chunk_end
                ]
            del source
            segments.append(
                {
                    "path": str(path),
                    "source_token_start": local_start,
                    "tokens": take,
                }
            )
            written += take
        source_start = source_end
        if source_start >= end:
            break
    if written != count:
        raise RuntimeError(f"copied {written:,} tokens, expected {count:,}")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-output-dir", default="", help="Canonical directory recorded in the manifest.")
    parser.add_argument("--output-shards", type=int, default=2)
    parser.add_argument("--name-prefix", default="fineweb_train_consumed_3276800000")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.target_tokens < 1 or args.output_shards < 1:
        raise ValueError("target-tokens and output-shards must be positive")
    if args.target_tokens % args.output_shards:
        raise ValueError("target-tokens must divide evenly across output-shards")
    consumed_per_output = args.target_tokens // args.output_shards
    physical_per_output = consumed_per_output + 1
    if physical_per_output > MAX_HEADER_TOKEN_COUNT:
        raise ValueError("an output shard would exceed the signed-int32 header capacity")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {args.manifest}")

    inspected = [(path, *inspect_shard(path)) for path in args.shards]
    available = sum(count for _, _, count in inspected)
    # Each output shard retains one next-token lookahead. Adjacent output shards
    # therefore overlap at their boundary; only the final lookahead extends the
    # logical training prefix by one physical source token.
    if available < args.target_tokens + 1:
        raise ValueError(f"only {available:,} tokens are available, need {args.target_tokens + 1:,}")
    source_rows = [
        {"path": str(path), "tokens": count, "sha256": sha256(path)}
        for path, _, count in inspected
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logical_dir = Path(args.logical_output_dir) if args.logical_output_dir else args.output_dir
    output_rows = []
    for index in range(args.output_shards):
        path = args.output_dir / f"{args.name_prefix}_{index + 1:06d}.bin"
        if path.exists():
            raise FileExistsError(f"refusing to replace existing output: {path}")
        temporary = path.with_name(f".{path.name}.incoming.{os.getpid()}")
        header = inspected[0][1].copy()
        header[2] = physical_per_output
        with temporary.open("wb") as handle:
            handle.write(header.astype("<i4", copy=False).tobytes())
            handle.truncate(HEADER_BYTES + 2 * physical_per_output)
        destination = np.memmap(
            temporary,
            mode="r+",
            dtype="<u2",
            offset=HEADER_BYTES,
            shape=(physical_per_output,),
        )
        logical_start = index * consumed_per_output
        segments = copy_global_range(inspected, logical_start, physical_per_output, destination)
        destination.flush()
        del destination
        temporary.replace(path)
        output_rows.append(
            {
                "path": str(logical_dir / path.name),
                "materialized_path": str(path),
                "logical_source_start": logical_start,
                "consumed_tokens": consumed_per_output,
                "lookahead_tokens": 1,
                "physical_tokens": physical_per_output,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_segments": segments,
            }
        )

    payload = {
        "schema": "speedrun-dlm-training-prefix-v1",
        "purpose": (
            "make the ordered training token stream invariant to per-step batch size and shard-tail handling "
            "for both DLM and next-token AR loaders"
        ),
        "target_consumed_tokens": args.target_tokens,
        "output_shards": args.output_shards,
        "consumed_tokens_per_output_shard": consumed_per_output,
        "physical_tokens_per_output_shard": physical_per_output,
        "boundary_lookahead_policy": (
            "each shard stores one next-token lookahead; adjacent shards overlap by that boundary token"
        ),
        "outputs": output_rows,
        "sources": source_rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_name(f".{args.manifest.name}.incoming.{os.getpid()}")
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)
    print(
        json.dumps(
            {
                "consumed_tokens": args.target_tokens,
                "physical_tokens": physical_per_output * args.output_shards,
                "output_shards": args.output_shards,
                "outputs": [{"path": row["path"], "sha256": row["sha256"]} for row in output_rows],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
