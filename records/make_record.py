#!/usr/bin/env python3
"""Create a minimal benchmark record folder.

The script copies machine-readable outputs into stable filenames:

- training_runs.jsonl: one TRAIN_RESULT JSON object per seed.
- gate_passes.tsv: one quality-gate summary row per seed.
- auxiliary_metrics.json: inference-cost profiler output plus diversity metrics.
- commands.jsonl: command templates used to train, gate, and profile the entry.
- environment.json: training-time hardware and software version snapshot.
- recipe_snapshot.txt: source files that define the trainer, sampler, and gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# Support the documented `python3 records/make_record.py` invocation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speedrun_dlm.score_generation_quality import (  # noqa: E402
    SEED_PROTOCOL,
    derive_evaluation_seed,
    load_verified_sample_manifest,
    sample_pass,
    sha256_file,
)

AR_REFERENCE_TFLOPS = 95.863037506048
AR_REFERENCE_TOTAL_PARAMETERS = 124_318_464
DEFAULT_SOURCE_FILES = [
    "Dockerfile",
    "constraints.txt",
    "requirements.txt",
    "generation_quality_rule.json",
    "prepare_data.sh",
    "data/cached_fineweb10B.py",
    "run_ar.sh",
    "run_dlm.sh",
    "records/capture_environment.py",
    "speedrun_dlm/train_ar.py",
    "speedrun_dlm/train_dlm.py",
    "speedrun_dlm/checkpoint_utils.py",
    "speedrun_dlm/sample_text.py",
    "speedrun_dlm/_quality_metrics.py",
    "speedrun_dlm/score_generation_quality.py",
    "speedrun_dlm/measure_inference_cost.py",
]


RECORDS_CSV_FIELDS = [
    "record_id",
    "entry",
    "trainer",
    "fixed_budget_steps",
    "parameters_total",
    "non_embedding_parameters",
    "seeds",
    "passes",
    "mean_timed_seconds",
    "mean_wall_seconds",
    "mean_pass_rate",
    "sampler",
    "nmc",
    "inference_tflops_per_1024_token_sample",
    "torch_profiler_tflops_per_1024_token_sample",
    "attention_tflops_added_by_formula",
    "inference_tflops_vs_ar",
    "parameter_normalized_inference_tflops_vs_ar",
    "diversity_unigram_entropy_bits_mean",
    "diversity_distinct_2_mean",
    "method_reference",
    "recipe_snapshot",
    "benchmark_label",
    "command_log",
    "record_path",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty TSV")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def binom_tail_ge(total: int, observed: int, threshold: float) -> float:
    """P[X >= observed] for X~Binomial(total, threshold)."""
    prob = 0.0
    for k in range(observed, total + 1):
        prob += math.comb(total, k) * (threshold**k) * ((1.0 - threshold) ** (total - k))
    return prob


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def official_training_seeds(rule: dict[str, Any]) -> tuple[int, ...]:
    start = int(rule["official_training_seed_start"])
    end = int(rule["official_training_seed_end"])
    if start > end:
        raise ValueError("official training seed range is empty")
    return tuple(range(start, end + 1))


def parse_gate_dir(raw_path: str, rule: dict[str, Any], require_significance: bool) -> dict[str, Any]:
    gate_dir = Path(raw_path)
    manifest_path = gate_dir / "sample_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing sample manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"missing evaluation metadata in {manifest_path}")
    checkpoint = Path(str(evaluation.get("checkpoint_path", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint recorded by {manifest_path} is unavailable: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    manifest, sample_payload = load_verified_sample_manifest(
        manifest_path,
        checkpoint_sha256,
        int(rule["official_sampling_batch_size"]),
    )
    evaluation = manifest["evaluation"]
    seed = int(evaluation["training_seed"])
    expected_seeds = evaluation["seeds"]

    expected_scoring = {
        "rule_version": rule["version"],
        "rule_sha256": sha256_file(ROOT / "generation_quality_rule.json"),
        "reference_model": rule["candidate_reference_model"],
        "pinned_reference_model": rule["pinned_reference_model"],
        "dtype": "bfloat16",
        "max_eval_tokens": 4096,
        "score_batch_size": 4,
        "reference_device_map": "",
        "allow_fineweb_fallback": False,
        "significance_required": True,
    }
    scoring = manifest.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError(f"missing scoring provenance in {manifest_path}")
    for key, expected in expected_scoring.items():
        if scoring.get(key) != expected:
            raise ValueError(f"official scorer setting {key!r} is invalid in {manifest_path}")

    sample_passes = gate_dir / "sample_passes.csv"
    if not sample_passes.exists():
        raise FileNotFoundError(f"missing {sample_passes}")
    sample_scores = gate_dir / "scores" / "sample_scores.csv"
    if not sample_scores.exists():
        raise FileNotFoundError(f"missing {sample_scores}")
    if scoring.get("sample_scores_sha256") != sha256_file(sample_scores):
        raise ValueError(f"sample-score hash does not match {manifest_path}")
    if scoring.get("sample_passes_sha256") != sha256_file(sample_passes):
        raise ValueError(f"sample-pass hash does not match {manifest_path}")
    sample_rows = sample_payload.get("samples")
    if len(sample_rows) != 128:
        raise ValueError("official quality gates must contain exactly 128 generated samples")

    with sample_passes.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with sample_scores.open(newline="") as handle:
        score_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("reference_model") == rule["candidate_reference_model"]
            and row.get("protocol") == "unconditional"
        ]
    samples = len(rows)
    if samples != len(sample_rows) or len(score_rows) != samples:
        raise ValueError(f"sample-pass count does not match generated sample count in {gate_dir}")
    for index, (row, score_row) in enumerate(zip(rows, score_rows, strict=True)):
        score_sample_id = score_row.get("sample_id", score_row.get("sample_index", ""))
        if row.get("sample_id") != score_sample_id:
            raise ValueError(f"sample {index} is misaligned between scores and passes in {gate_dir}")
        if truthy(row.get("pass", 0)) != sample_pass(score_row, rule):
            raise ValueError(f"sample {index} pass was not produced by the official rule in {gate_dir}")
    passing = sum(sample_pass(row, rule) for row in score_rows)
    pass_rate = passing / samples if samples else 0.0
    threshold = float(rule["pass_rate_threshold"])
    p_value = binom_tail_ge(samples, passing, threshold)
    overall_pass = pass_rate >= threshold and (
        not require_significance or p_value < float(rule["significance_p_value"])
    )
    return {
        "seed": seed,
        "sampling_seed": expected_seeds["sampling"],
        "checkpoint_sha256": checkpoint_sha256,
        "seed_protocol": SEED_PROTOCOL,
        "samples": samples,
        "passing_samples": passing,
        "pass_rate": f"{pass_rate:.4f}",
        "exact_binomial_p_value_vs_threshold": f"{p_value:.6g}",
        "significance_required": str(bool(require_significance)),
        "overall_pass": str(bool(overall_pass)),
    }


def train_result_from_log(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("TRAIN_RESULT "):
            return json.loads(line[len("TRAIN_RESULT ") :])
    raise ValueError(f"no TRAIN_RESULT line found in {path}")


def checkpoint_tag(path: str) -> str:
    name = Path(path).name
    step_match = re.search(r"_step(\d+)\.pt$", name)
    if step_match:
        return f"step{step_match.group(1)}"
    if name.endswith("_final.pt"):
        return "final"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)


def checkpoint_uri(record_name: str, seed: Any, path: str, tag: str | None = None) -> str:
    if path.startswith("records://"):
        return path
    seed_label = f"seed-{seed}" if seed is not None else "seed-unknown"
    tag_label = tag or checkpoint_tag(path)
    return f"records://{record_name}/checkpoints/{seed_label}/{tag_label}.pt"


def sanitize_train_result(record_name: str, row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    seed = row.get("seed")
    if isinstance(row.get("checkpoint_path"), str):
        row["checkpoint_path"] = checkpoint_uri(record_name, seed, row["checkpoint_path"])
    if isinstance(row.get("checkpoint_paths"), list):
        row["checkpoint_paths"] = [
            checkpoint_uri(record_name, seed, path) if isinstance(path, str) else path
            for path in row["checkpoint_paths"]
        ]
    if isinstance(row.get("checkpoint_records"), list):
        row["checkpoint_records"] = [dict(item) if isinstance(item, dict) else item for item in row["checkpoint_records"]]
        for item in row["checkpoint_records"]:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                tag = str(item.get("tag")) if item.get("tag") is not None else None
                item["path"] = checkpoint_uri(record_name, seed, item["path"], tag)
    return row


def sanitize_inference_cost(record_name: str, row: dict[str, Any], profiled_train_seed: int | None) -> dict[str, Any]:
    row = dict(row)
    row.pop("checkpoint_path", None)
    if isinstance(row.get("snapshot"), str):
        row["snapshot"] = checkpoint_uri(record_name, profiled_train_seed, row["snapshot"])
    return row


def copy_training_runs(
    output: Path,
    jsonl: Path | None,
    json_paths: list[str],
    log_paths: list[str],
) -> list[dict[str, Any]]:
    destination = output / "training_runs.jsonl"
    if jsonl:
        rows = read_jsonl(jsonl)
    else:
        rows = [json.loads(Path(path).read_text()) for path in json_paths]
        rows.extend(train_result_from_log(Path(path)) for path in log_paths)
    rows = [sanitize_train_result(output.name, row) for row in rows]
    rows.sort(key=lambda row: int(row.get("seed", 0)))
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return read_jsonl(destination)


def write_recipe_snapshot(output: Path, source_files: list[str], note: str) -> None:
    lines = ["# Recipe snapshot", "", note.strip(), ""]
    for rel in source_files:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        lines.extend([f"===== {rel} =====", path.read_text(), ""])
    (output / "recipe_snapshot.txt").write_text("\n".join(lines))


def write_commands(output: Path, commands: list[str]) -> None:
    if not commands:
        return
    rows = []
    for index, raw in enumerate(commands, start=1):
        if "=" in raw:
            kind, command = raw.split("=", 1)
        else:
            kind, command = f"command_{index}", raw
        rows.append({"kind": kind, "command": command})
    (output / "commands.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def copy_environment_json(output: Path, environment_json: Path | None) -> None:
    if environment_json is None:
        raise ValueError("a training-time environment snapshot is required")
    # Validate the training-time snapshot before publishing it as a record artifact.
    environment = json.loads(environment_json.read_text())
    if not isinstance(environment, dict):
        raise ValueError(f"{environment_json} must contain a JSON object")
    destination = output / "environment.json"
    if environment_json.resolve() != destination.resolve():
        shutil.copyfile(environment_json, destination)


def write_auxiliary_metrics(
    output: Path,
    inference_cost_json: Path,
    profiled_train_seed: int | None,
    diversity_unigram_entropy_bits: float | None,
    diversity_distinct_2: float | None,
) -> dict[str, Any]:
    raw_inference = json.loads(inference_cost_json.read_text())
    if raw_inference.get("seed_protocol") != SEED_PROTOCOL:
        raise ValueError(f"{inference_cost_json} does not use {SEED_PROTOCOL}")
    checkpoint = Path(str(raw_inference.get("checkpoint_path", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"profiled checkpoint is unavailable: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != raw_inference.get("checkpoint_sha256"):
        raise ValueError(f"profiled checkpoint hash does not match {inference_cost_json}")
    expected_seed = derive_evaluation_seed(checkpoint_sha256, "inference_cost")
    reported_seed = raw_inference.get("seed")
    if type(reported_seed) is not int or reported_seed != expected_seed:
        raise ValueError(
            f"{inference_cost_json} reports inference-cost seed {reported_seed!r}; "
            f"the official checkpoint-derived seed is {expected_seed}"
        )
    if profiled_train_seed is None or int(raw_inference.get("training_seed")) != profiled_train_seed:
        raise ValueError("--profiled-train-seed must match the training seed stored in the profiled checkpoint")
    inference = sanitize_inference_cost(output.name, raw_inference, profiled_train_seed)
    auxiliary = {
        "inference_cost": inference,
        "diversity": {
            "unigram_entropy_bits_mean": diversity_unigram_entropy_bits,
            "distinct_2_mean": diversity_distinct_2,
        },
    }
    (output / "auxiliary_metrics.json").write_text(json.dumps(auxiliary, indent=2, sort_keys=True) + "\n")
    return auxiliary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def append_records_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORDS_CSV_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RECORDS_CSV_FIELDS})


def validate_record_rows(
    training_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, str]],
    rule: dict[str, Any],
    min_passes: int | None,
) -> None:
    if not training_rows:
        raise SystemExit("training_runs.jsonl is empty")
    if not gate_rows:
        raise SystemExit("gate_passes.tsv is empty")
    if len(training_rows) != len(gate_rows):
        raise SystemExit(
            f"expected one gate row per training seed, got {len(training_rows)} training rows "
            f"and {len(gate_rows)} gate rows"
        )
    expected_seeds = list(official_training_seeds(rule))
    seed_range = f"{expected_seeds[0]} through {expected_seeds[-1]}"
    train_seed_list = sorted(int(row["seed"]) for row in training_rows)
    gate_seed_list = sorted(int(row["seed"]) for row in gate_rows)
    if train_seed_list != expected_seeds:
        raise SystemExit(f"official records must contain every training seed from {seed_range} exactly once")
    if gate_seed_list != expected_seeds:
        raise SystemExit(f"official records must contain one verified gate output for every seed from {seed_range}")
    train_seeds = {str(row.get("seed")) for row in training_rows if row.get("seed") is not None}
    gate_seeds = {str(row.get("seed")) for row in gate_rows if row.get("seed")}
    if train_seeds and gate_seeds and train_seeds != gate_seeds:
        raise SystemExit(
            "training and gate seed sets differ: "
            f"training={sorted(train_seeds, key=int)} gate={sorted(gate_seeds, key=int)}"
        )
    passes = 0
    for row in gate_rows:
        if row.get("seed_protocol") != SEED_PROTOCOL or not row.get("checkpoint_sha256"):
            seed = row.get("seed", "<unknown>")
            raise SystemExit(f"seed {seed} does not have verified checkpoint-derived evaluation seeds")
        if not truthy(row.get("significance_required", "")):
            seed = row.get("seed", "<unknown>")
            raise SystemExit(f"seed {seed} was not checked with the required significance rule")
        passes += int(truthy(row.get("overall_pass", "")))
    required = len(gate_rows) if min_passes is None else min_passes
    if passes < required:
        raise SystemExit(f"expected at least {required} passing seeds, got {passes}/{len(gate_rows)}")


def model_calls_for_gate_sample(args: argparse.Namespace, inference: dict[str, Any]) -> int:
    if args.nmc is not None:
        return args.nmc
    if args.trainer == "dlm" and inference.get("dlm_sampling_steps"):
        return int(float(inference["dlm_sampling_steps"]))
    if args.trainer == "ar" and inference.get("tokens_per_sample"):
        return int(float(inference["tokens_per_sample"]))
    return int(float(inference["forward_calls_per_sample"]))


def build_records_row(
    args: argparse.Namespace,
    record_dir: Path,
    training_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, str]],
    auxiliary: dict[str, Any],
) -> dict[str, Any]:
    inference = auxiliary["inference_cost"]
    first_train = training_rows[0]
    passes = sum(str(row.get("overall_pass", "")).lower() == "true" for row in gate_rows)
    pass_rates = [float(row["pass_rate"]) for row in gate_rows]
    parameters_total = first_train.get("num_parameters") or inference.get("num_parameters", "")
    non_embedding_parameters = first_train.get("non_embedding_parameters") or inference.get("non_embedding_parameters", "")
    tflops = float(
        inference.get("total_tflops_per_1024_token_sample")
        or inference["profiler_tflops_per_1024_token_sample"]
    )
    profiler_tflops = float(inference["profiler_tflops_per_1024_token_sample"])
    attention_tflops = float(
        inference.get("attention_tflops_added_by_formula_per_1024_token_sample")
        or 0.0
    )
    tflops_vs_ar = tflops / args.ar_reference_tflops
    parameter_normalized_tflops_vs_ar = tflops_vs_ar * (
        args.ar_reference_total_parameters / float(parameters_total)
    )
    timed = [float(row["timed_training_seconds"]) for row in training_rows if row.get("timed_training_seconds") is not None]
    wall = [
        float(row["total_wallclock_seconds"])
        for row in training_rows
        if row.get("total_wallclock_seconds") is not None
    ]
    diversity = auxiliary["diversity"]
    return {
        "record_id": args.record_id,
        "entry": args.entry,
        "trainer": args.trainer,
        "fixed_budget_steps": first_train.get("num_iterations", ""),
        "parameters_total": parameters_total,
        "non_embedding_parameters": non_embedding_parameters,
        "seeds": len(training_rows),
        "passes": passes,
        "mean_timed_seconds": f"{mean(timed):.1f}",
        "mean_wall_seconds": f"{mean(wall):.1f}",
        "mean_pass_rate": f"{mean(pass_rates):.4f}",
        "sampler": args.sampler,
        "nmc": model_calls_for_gate_sample(args, inference),
        "inference_tflops_per_1024_token_sample": f"{tflops:.4f}",
        "torch_profiler_tflops_per_1024_token_sample": f"{profiler_tflops:.4f}",
        "attention_tflops_added_by_formula": f"{attention_tflops:.4f}",
        "inference_tflops_vs_ar": f"{tflops_vs_ar:.4f}",
        "parameter_normalized_inference_tflops_vs_ar": f"{parameter_normalized_tflops_vs_ar:.4f}",
        "diversity_unigram_entropy_bits_mean": diversity["unigram_entropy_bits_mean"],
        "diversity_distinct_2_mean": diversity["distinct_2_mean"],
        "method_reference": args.method_reference,
        "recipe_snapshot": "recipe_snapshot.txt",
        "benchmark_label": args.benchmark_label,
        "command_log": "commands.jsonl" if (record_dir / "commands.jsonl").exists() else "",
        "record_path": f"{record_dir.as_posix().rstrip('/')}/",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal speedrun-dlm benchmark record.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--trainer", choices=("ar", "dlm"), required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--benchmark-label", default="")
    parser.add_argument("--method-reference", default="")
    parser.add_argument("--min-passes", type=int, help="Required passing seeds. Defaults to all included seeds.")
    parser.add_argument("--nmc", type=int, help="Model calls used to generate one 1024-token gate sample.")
    parser.add_argument("--ar-reference-tflops", type=float, default=AR_REFERENCE_TFLOPS)
    parser.add_argument(
        "--ar-reference-total-parameters",
        "--ar-reference-parameters",
        dest="ar_reference_total_parameters",
        type=float,
        default=AR_REFERENCE_TOTAL_PARAMETERS,
    )
    parser.add_argument("--training-runs-jsonl", type=Path)
    parser.add_argument("--train-result-json", action="append", default=[])
    parser.add_argument("--train-log", action="append", default=[], help="Repeat with logs containing TRAIN_RESULT lines.")
    parser.add_argument(
        "--gate-dir",
        action="append",
        default=[],
        help="Repeat with each scorer output directory; its training seed is read from sample_manifest.json.",
    )
    parser.add_argument("--inference-cost-json", type=Path, required=True)
    parser.add_argument(
        "--profiled-train-seed",
        type=int,
        required=True,
        help="Training seed of the checkpoint used for inference-cost profiling.",
    )
    parser.add_argument("--diversity-unigram-entropy-bits", type=float)
    parser.add_argument("--diversity-distinct-2", type=float)
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--snapshot-note", default="Source files defining the trainer, sampler, and gate.")
    parser.add_argument(
        "--environment-json",
        type=Path,
        required=True,
        help="Training-time hardware and software version snapshot to copy into the record.",
    )
    parser.add_argument("--command", action="append", default=[], help="Repeat as KIND=command to write commands.jsonl.")
    parser.add_argument("--records-csv", type=Path, help="Append one row to this leaderboard CSV.")
    args = parser.parse_args()

    if not args.training_runs_jsonl and not args.train_result_json and not args.train_log:
        raise SystemExit("Provide --training-runs-jsonl, --train-result-json, or --train-log.")
    if not args.gate_dir:
        raise SystemExit("Provide one --gate-dir for each official training seed.")

    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    training_rows = copy_training_runs(record_dir, args.training_runs_jsonl, args.train_result_json, args.train_log)

    rule = json.loads((ROOT / "generation_quality_rule.json").read_text())
    write_tsv(record_dir / "gate_passes.tsv", [parse_gate_dir(spec, rule, True) for spec in args.gate_dir])
    gate_rows = read_tsv(record_dir / "gate_passes.tsv")
    validate_record_rows(training_rows, gate_rows, rule, args.min_passes)

    auxiliary = write_auxiliary_metrics(
        record_dir,
        args.inference_cost_json,
        profiled_train_seed=args.profiled_train_seed,
        diversity_unigram_entropy_bits=args.diversity_unigram_entropy_bits,
        diversity_distinct_2=args.diversity_distinct_2,
    )
    write_commands(record_dir, args.command)
    copy_environment_json(record_dir, args.environment_json)
    write_recipe_snapshot(record_dir, args.source_file or DEFAULT_SOURCE_FILES, args.snapshot_note)

    row = build_records_row(args, record_dir, training_rows, gate_rows, auxiliary)
    if args.records_csv:
        append_records_csv(args.records_csv, row)

    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
