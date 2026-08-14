#!/usr/bin/env python3
"""Create a minimal benchmark record folder.

The script copies machine-readable outputs into stable filenames:

- training_runs.jsonl: one TRAIN_RESULT JSON object per seed.
- quality_gate.json: one decision from all sample outcomes.
- auxiliary_metrics.json: mean inference cost across the checkpoint profiles plus diversity metrics.
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

INFERENCE_PROFILE_SHARED_FIELDS = (
    "trainer",
    "checkpoint_variant",
    "num_parameters",
    "non_embedding_parameters",
    "tokens_per_sample",
    "measured_samples",
    "warmup_samples",
    "temperature",
    "top_k",
    "dlm_sampler",
    "dlm_sampling_steps",
    "sampling_eps",
    "amp_dtype",
    "profiler",
    "cost_accounting_version",
    "attention_flops_formula",
    "profiler_caveat",
)
INFERENCE_PROFILE_MEAN_FIELDS = (
    "forward_calls_per_sample",
    "profiler_flops_per_1024_token_sample",
    "profiler_tflops_per_1024_token_sample",
    "attention_flops_measured_by_torch_per_1024_token_sample",
    "attention_tflops_measured_by_torch_per_1024_token_sample",
    "attention_flops_by_formula_per_1024_token_sample",
    "attention_tflops_by_formula_per_1024_token_sample",
    "attention_flops_added_by_formula_per_1024_token_sample",
    "attention_tflops_added_by_formula_per_1024_token_sample",
    "total_flops_per_1024_token_sample",
    "total_tflops_per_1024_token_sample",
)
INFERENCE_GATE_SHARED_FIELDS = (
    "trainer",
    "checkpoint_variant",
    "tokens_per_sample",
    "temperature",
    "top_k",
    "dlm_sampler",
    "dlm_sampling_steps",
    "sampling_eps",
)
INFERENCE_PROFILE_PROTOCOL = {
    "measured_samples": 1,
    "warmup_samples": 1,
    "profiler": "torch.profiler.profile(with_flops=True)",
    "cost_accounting_version": "v0.2-torch-profiler-plus-sdpa-estimate",
}


RECORDS_CSV_FIELDS = [
    "record_id",
    "entry",
    "trainer",
    "fixed_budget_steps",
    "parameters_total",
    "non_embedding_parameters",
    "seeds",
    "passing_samples",
    "samples",
    "pass_rate",
    "quality_gate_pass",
    "mean_timed_seconds",
    "mean_wall_seconds",
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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def official_training_seeds(rule: dict[str, Any]) -> tuple[int, ...]:
    start = int(rule["official_training_seed_start"])
    end = int(rule["official_training_seed_end"])
    if start > end:
        raise ValueError("official training seed range is empty")
    return tuple(range(start, end + 1))


def parse_gate_dir(
    raw_path: str,
    rule: dict[str, Any],
    for_official_submission: bool,
) -> dict[str, Any]:
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
    }
    scoring = manifest.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError(f"missing scoring provenance in {manifest_path}")
    for key, expected in expected_scoring.items():
        if scoring.get(key) != expected:
            raise ValueError(f"official scorer setting {key!r} is invalid in {manifest_path}")

    sample_outcomes = gate_dir / "sample_outcomes.csv"
    if not sample_outcomes.exists():
        raise FileNotFoundError(f"missing {sample_outcomes}")
    sample_scores = gate_dir / "scores" / "sample_scores.csv"
    if not sample_scores.exists():
        raise FileNotFoundError(f"missing {sample_scores}")
    if scoring.get("sample_scores_sha256") != sha256_file(sample_scores):
        raise ValueError(f"sample-score hash does not match {manifest_path}")
    if scoring.get("sample_outcomes_sha256") != sha256_file(sample_outcomes):
        raise ValueError(f"sample-outcome hash does not match {manifest_path}")
    sample_rows = sample_payload.get("samples")
    expected_samples = int(rule["official_samples_per_training_seed"])
    if for_official_submission and len(sample_rows) != expected_samples:
        raise ValueError(
            "quality gates for official submission must contain exactly "
            f"{expected_samples} generated samples"
        )
    if not sample_rows:
        raise ValueError(f"quality gate contains no generated samples: {gate_dir}")
    first_sample = sample_rows[0]
    sampling = {
        "trainer": first_sample.get("trainer"),
        "checkpoint_variant": first_sample.get("checkpoint_variant"),
        "tokens_per_sample": first_sample.get("max_new_tokens"),
        "temperature": first_sample.get("temperature"),
        "top_k": first_sample.get("top_k"),
        "dlm_sampler": first_sample.get("dlm_sampler"),
        "dlm_sampling_steps": first_sample.get("num_sampling_steps"),
        "sampling_eps": first_sample.get("sampling_eps"),
    }
    for index, sample in enumerate(sample_rows):
        current = {
            "trainer": sample.get("trainer"),
            "checkpoint_variant": sample.get("checkpoint_variant"),
            "tokens_per_sample": sample.get("max_new_tokens"),
            "temperature": sample.get("temperature"),
            "top_k": sample.get("top_k"),
            "dlm_sampler": sample.get("dlm_sampler"),
            "dlm_sampling_steps": sample.get("num_sampling_steps"),
            "sampling_eps": sample.get("sampling_eps"),
        }
        if current != sampling:
            raise ValueError(f"sample {index} uses different sampling settings in {gate_dir}")

    with sample_outcomes.open(newline="") as handle:
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
        raise ValueError(f"sample-outcome count does not match generated sample count in {gate_dir}")
    for index, (row, score_row) in enumerate(zip(rows, score_rows, strict=True)):
        score_sample_id = score_row.get("sample_id", score_row.get("sample_index", ""))
        if row.get("sample_id") != score_sample_id:
            raise ValueError(f"sample {index} is misaligned between scores and outcomes in {gate_dir}")
        if truthy(row.get("pass", 0)) != sample_pass(score_row, rule):
            raise ValueError(f"sample {index} outcome was not produced by the official rule in {gate_dir}")
    passing = sum(sample_pass(row, rule) for row in score_rows)
    pass_rate = passing / samples if samples else 0.0
    return {
        "seed": seed,
        "sampling_seed": expected_seeds["sampling"],
        "checkpoint_sha256": checkpoint_sha256,
        "seed_protocol": SEED_PROTOCOL,
        "sample_manifest_sha256": sha256_file(manifest_path),
        "sample_sha256": manifest["sample_sets"][0]["sample_sha256"],
        "sample_scores_sha256": scoring["sample_scores_sha256"],
        "sample_outcomes_sha256": scoring["sample_outcomes_sha256"],
        "sampling": sampling,
        "samples": samples,
        "passing_samples": passing,
        "pass_rate": pass_rate,
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


def sanitize_inference_cost(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("checkpoint_path", None)
    row.pop("snapshot", None)
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
    inference_cost_jsons: list[Path],
    gate_rows: list[dict[str, Any]],
    expected_trainer: str,
    diversity_unigram_entropy_bits: float | None,
    diversity_distinct_2: float | None,
) -> dict[str, Any]:
    gates_by_seed = {int(row["seed"]): row for row in gate_rows}
    expected_hashes = {seed: row["checkpoint_sha256"] for seed, row in gates_by_seed.items()}
    profiles_by_seed: dict[int, dict[str, Any]] = {}
    for inference_cost_json in inference_cost_jsons:
        raw_inference = json.loads(inference_cost_json.read_text())
        if raw_inference.get("seed_protocol") != SEED_PROTOCOL:
            raise ValueError(f"{inference_cost_json} does not use {SEED_PROTOCOL}")
        training_seed = raw_inference.get("training_seed")
        if type(training_seed) is not int:
            raise ValueError(f"{inference_cost_json} does not contain an integer training_seed")
        if training_seed in profiles_by_seed:
            raise ValueError(f"duplicate inference-cost profile for training seed {training_seed}")
        if training_seed not in expected_hashes:
            raise ValueError(f"inference-cost profile has unexpected training seed {training_seed}")
        checkpoint_sha256 = raw_inference.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str):
            raise ValueError(f"{inference_cost_json} does not contain a checkpoint hash")
        if checkpoint_sha256 != expected_hashes[training_seed]:
            raise ValueError(
                f"inference-cost profile and quality gate use different checkpoints for seed {training_seed}"
            )
        expected_seed = derive_evaluation_seed(checkpoint_sha256, "inference_cost")
        reported_seed = raw_inference.get("seed")
        if type(reported_seed) is not int or reported_seed != expected_seed:
            raise ValueError(
                f"{inference_cost_json} reports inference-cost seed {reported_seed!r}; "
                f"the official checkpoint-derived seed is {expected_seed}"
            )
        if raw_inference.get("trainer") != expected_trainer:
            raise ValueError(
                f"{inference_cost_json} profiles trainer {raw_inference.get('trainer')!r}, "
                f"expected {expected_trainer!r}"
            )
        gate_sampling = gates_by_seed[training_seed]["sampling"]
        for field in INFERENCE_GATE_SHARED_FIELDS:
            if raw_inference.get(field) != gate_sampling[field]:
                raise ValueError(
                    f"inference-cost profile and quality gate use different {field} "
                    f"for seed {training_seed}"
                )
        for field, expected in INFERENCE_PROFILE_PROTOCOL.items():
            if raw_inference.get(field) != expected:
                raise ValueError(
                    f"inference-cost profile for seed {training_seed} must use {field}={expected!r}"
                )
        profiles_by_seed[training_seed] = sanitize_inference_cost(raw_inference)

    if set(profiles_by_seed) != set(expected_hashes):
        missing = sorted(set(expected_hashes) - set(profiles_by_seed))
        raise ValueError(f"missing inference-cost profiles for training seeds {missing}")

    profiles = [profiles_by_seed[seed] for seed in sorted(profiles_by_seed)]
    for field in INFERENCE_PROFILE_SHARED_FIELDS:
        if field not in profiles[0]:
            raise ValueError(f"inference-cost profiles do not contain {field}")
    shared = {field: profiles[0].get(field) for field in INFERENCE_PROFILE_SHARED_FIELDS}
    for profile in profiles[1:]:
        for field, expected in shared.items():
            if profile.get(field) != expected:
                raise ValueError(
                    f"inference-cost profiles use different {field}: "
                    f"seed {profiles[0]['training_seed']} has {expected!r}, "
                    f"seed {profile['training_seed']} has {profile.get(field)!r}"
                )
    means = {}
    for field in INFERENCE_PROFILE_MEAN_FIELDS:
        values = [profile.get(field) for profile in profiles]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f"inference-cost profiles must contain finite numeric {field}")
        means[field] = mean(values)
    published_profiles = [
        {
            "training_seed": profile["training_seed"],
            "checkpoint_sha256": profile["checkpoint_sha256"],
            "seed": profile["seed"],
            **{field: profile[field] for field in INFERENCE_PROFILE_MEAN_FIELDS},
        }
        for profile in profiles
    ]

    inference = {
        "version": 1,
        "aggregation": "mean-across-training-seeds",
        "seed_protocol": SEED_PROTOCOL,
        "profile_count": len(profiles),
        "training_seeds": sorted(profiles_by_seed),
        "shared": shared,
        "mean": means,
        "profiles": published_profiles,
    }
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
    if exists:
        with path.open(newline="") as handle:
            existing_fields = next(csv.reader(handle), [])
        if existing_fields != RECORDS_CSV_FIELDS:
            raise ValueError(f"{path} does not use the current quality-gate leaderboard columns")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORDS_CSV_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RECORDS_CSV_FIELDS})


def validate_record_rows(
    training_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    rule: dict[str, Any],
    experimental: bool,
) -> None:
    if not training_rows:
        raise SystemExit("training_runs.jsonl is empty")
    if not gate_rows:
        raise SystemExit("no quality-gate outputs were provided")
    if len(training_rows) != len(gate_rows):
        raise SystemExit(
            f"expected one gate row per training seed, got {len(training_rows)} training rows "
            f"and {len(gate_rows)} gate rows"
        )
    expected_seeds = list(official_training_seeds(rule))
    seed_range = f"{expected_seeds[0]} through {expected_seeds[-1]}"
    train_seed_list = sorted(int(row["seed"]) for row in training_rows)
    gate_seed_list = sorted(int(row["seed"]) for row in gate_rows)
    if len(train_seed_list) != len(set(train_seed_list)):
        raise SystemExit("training seed list contains duplicates")
    if len(gate_seed_list) != len(set(gate_seed_list)):
        raise SystemExit("quality-gate seed list contains duplicates")
    if train_seed_list != gate_seed_list:
        raise SystemExit(
            "training and gate seed sets differ: "
            f"training={train_seed_list} gate={gate_seed_list}"
        )
    if not experimental and train_seed_list != expected_seeds:
        raise SystemExit(f"official records must contain every training seed from {seed_range} exactly once")
    if not experimental:
        for field in ("num_iterations", "total_tokens_processed"):
            values_by_seed = {}
            for row in training_rows:
                value = row.get(field)
                if type(value) is not int or value <= 0:
                    raise SystemExit(
                        f"official training row for seed {row.get('seed', '<unknown>')} "
                        f"must contain a positive integer {field}"
                    )
                values_by_seed[int(row["seed"])] = value
            if len(set(values_by_seed.values())) != 1:
                raise SystemExit(
                    f"official training runs must use one common {field}; got {values_by_seed}"
                )
    for row in gate_rows:
        if row.get("seed_protocol") != SEED_PROTOCOL or not row.get("checkpoint_sha256"):
            seed = row.get("seed", "<unknown>")
            raise SystemExit(f"seed {seed} does not have verified checkpoint-derived evaluation seeds")


def aggregate_quality_gate(
    gate_rows: list[dict[str, Any]],
    rule: dict[str, Any],
    for_official_submission: bool,
) -> dict[str, Any]:
    gate_rows = sorted(gate_rows, key=lambda row: int(row["seed"]))
    total_samples = sum(int(row["samples"]) for row in gate_rows)
    passing_samples = sum(int(row["passing_samples"]) for row in gate_rows)
    if total_samples <= 0:
        raise ValueError("cannot evaluate an empty sample set")
    pass_rate = passing_samples / total_samples
    minimum_passing_samples = int(rule["official_min_passing_samples"])
    official_sample_count = len(official_training_seeds(rule)) * int(
        rule["official_samples_per_training_seed"]
    )
    if not 1 <= minimum_passing_samples <= official_sample_count:
        raise ValueError("official minimum passing-sample count is outside the official sample range")
    if for_official_submission and total_samples != official_sample_count:
        raise ValueError(
            f"official submissions must contain exactly {official_sample_count} samples, "
            f"got {total_samples}"
        )
    passes = passing_samples >= minimum_passing_samples if for_official_submission else None
    return {
        "version": 1,
        "aggregation": "all-samples",
        "for_official_submission": for_official_submission,
        "rule_version": rule["version"],
        "rule_sha256": sha256_file(ROOT / "generation_quality_rule.json"),
        "evaluation_seed_protocol": SEED_PROTOCOL,
        "training_seeds": [int(row["seed"]) for row in gate_rows],
        "runs": gate_rows,
        "samples": total_samples,
        "passing_samples": passing_samples,
        "pass_rate": pass_rate,
        "minimum_passing_samples": minimum_passing_samples,
        "passes": passes,
    }


def write_quality_gate(path: Path, quality_gate: dict[str, Any]) -> None:
    path.write_text(json.dumps(quality_gate, indent=2, sort_keys=True) + "\n")


def build_records_row(
    args: argparse.Namespace,
    record_dir: Path,
    training_rows: list[dict[str, Any]],
    quality_gate: dict[str, Any],
    auxiliary: dict[str, Any],
) -> dict[str, Any]:
    inference = auxiliary["inference_cost"]
    inference_shared = inference["shared"]
    inference_mean = inference["mean"]
    first_train = training_rows[0]
    parameters_total = first_train.get("num_parameters") or inference_shared.get("num_parameters", "")
    non_embedding_parameters = first_train.get("non_embedding_parameters") or inference_shared.get("non_embedding_parameters", "")
    tflops = float(
        inference_mean.get("total_tflops_per_1024_token_sample")
        or inference_mean["profiler_tflops_per_1024_token_sample"]
    )
    profiler_tflops = float(inference_mean["profiler_tflops_per_1024_token_sample"])
    attention_tflops = float(
        inference_mean.get("attention_tflops_added_by_formula_per_1024_token_sample")
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
        "passing_samples": quality_gate["passing_samples"],
        "samples": quality_gate["samples"],
        "pass_rate": f"{quality_gate['pass_rate']:.6f}",
        "quality_gate_pass": quality_gate["passes"],
        "mean_timed_seconds": f"{mean(timed):.1f}",
        "mean_wall_seconds": f"{mean(wall):.1f}",
        "sampler": args.sampler,
        "nmc": f"{inference_mean['forward_calls_per_sample']:.2f}",
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
    parser.add_argument(
        "--experimental",
        action="store_true",
        help=(
            "Allow any nonempty matching subset of training and gate seeds, and any positive sample count, "
            "for development. The result is marked as not for official submission and cannot be appended "
            "to the leaderboard CSV."
        ),
    )
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
    parser.add_argument(
        "--inference-cost-json",
        type=Path,
        action="append",
        required=True,
        help="Repeat with one checkpoint-derived inference-cost profile per training seed.",
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
        raise SystemExit("Provide at least one --gate-dir.")
    if args.experimental and args.records_csv:
        raise SystemExit("experimental results cannot be appended to the leaderboard CSV")

    record_dir = Path(args.record_dir)
    if record_dir.exists() and any(record_dir.iterdir()):
        raise SystemExit(f"record directory must be new or empty: {record_dir}")
    record_dir.mkdir(parents=True, exist_ok=True)
    training_rows = copy_training_runs(record_dir, args.training_runs_jsonl, args.train_result_json, args.train_log)

    rule = json.loads((ROOT / "generation_quality_rule.json").read_text())
    gate_rows = [
        parse_gate_dir(spec, rule, for_official_submission=not args.experimental)
        for spec in args.gate_dir
    ]
    gate_rows.sort(key=lambda row: int(row["seed"]))
    validate_record_rows(training_rows, gate_rows, rule, experimental=args.experimental)
    quality_gate = aggregate_quality_gate(
        gate_rows,
        rule,
        for_official_submission=not args.experimental,
    )
    write_quality_gate(record_dir / "quality_gate.json", quality_gate)
    if not args.experimental and not quality_gate["passes"]:
        raise SystemExit(
            "quality gate failed: "
            f"{quality_gate['passing_samples']}/{quality_gate['samples']} samples passed "
            f"(need at least {quality_gate['minimum_passing_samples']})"
        )

    auxiliary = write_auxiliary_metrics(
        record_dir,
        args.inference_cost_json,
        gate_rows,
        expected_trainer=args.trainer,
        diversity_unigram_entropy_bits=args.diversity_unigram_entropy_bits,
        diversity_distinct_2=args.diversity_distinct_2,
    )
    mean_inference_tflops = auxiliary["inference_cost"]["mean"][
        "total_tflops_per_1024_token_sample"
    ]
    if not args.experimental and args.trainer == "dlm" and mean_inference_tflops >= args.ar_reference_tflops:
        raise SystemExit(
            f"mean inference cost is {mean_inference_tflops:.6f} TFLOPs per 1024-token sample; "
            f"official DLM records must remain below {args.ar_reference_tflops:.6f} TFLOPs"
        )
    write_commands(record_dir, args.command)
    copy_environment_json(record_dir, args.environment_json)
    write_recipe_snapshot(record_dir, args.source_file or DEFAULT_SOURCE_FILES, args.snapshot_note)

    row = build_records_row(args, record_dir, training_rows, quality_gate, auxiliary)
    if args.records_csv:
        append_records_csv(args.records_csv, row)

    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
