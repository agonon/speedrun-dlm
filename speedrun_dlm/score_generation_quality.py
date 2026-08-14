import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SEED_PROTOCOL = "checkpoint-sha256-v1"
SEED_PURPOSES = (
    "sampling",
    "inference_cost",
    "order_contrast",
    "fineweb_controls",
    "document_coherence",
)
EVAL_SEED_MODULUS = 2**63

# Release gate settings passed through to _quality_metrics.
ORDER_CONTRAST = {
    "pairs_per_sample": 8,
    "distractors": 4,
    "context_sentences": 3,
    "min_sentence_tokens": 5,
    "max_sentence_tokens": 80,
    "distractor_fineweb_count": 128,
}
DOCUMENT_COHERENCE = {
    "min_sentences": 5,
    "max_sentences": 12,
    "max_sentence_tokens": 80,
    "shuffles": 1,
    "swaps": 1,
    "replacements": 1,
    "block_replacements": 1,
    "tail_splices": 1,
    "replacement_sentences": 2,
    "block_sentences": 3,
}

UNCONDITIONAL_PRESET: dict[str, float] = {
    "min_tokens": 500,
    "max_repeat_3gram": 0.12,
    "max_repeat_4gram": 0.08,
    "min_unique": 0.25,
    "min_compression": 0.30,
    "max_compression": 0.75,
    "max_token_run": 6,
    "max_eot": 6,
    "max_mask": 0,
    "max_bad": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_evaluation_seed(checkpoint_sha256: str, purpose: str) -> int:
    if purpose not in SEED_PURPOSES:
        raise ValueError(f"Unknown evaluation seed purpose: {purpose!r}")
    payload = f"{SEED_PROTOCOL}\0{checkpoint_sha256}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % EVAL_SEED_MODULUS


def checkpoint_evaluation_seeds(checkpoint_sha256: str) -> dict[str, int]:
    return {purpose: derive_evaluation_seed(checkpoint_sha256, purpose) for purpose in SEED_PURPOSES}


def load_verified_sample_manifest(
    manifest_path: Path,
    checkpoint_sha256: str,
    official_sampling_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a sample manifest and verify its checkpoint-derived evaluation metadata."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported sample-manifest version in {manifest_path}")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Missing evaluation metadata in {manifest_path}")
    if evaluation.get("seed_protocol") != SEED_PROTOCOL:
        raise ValueError(f"{manifest_path} does not use {SEED_PROTOCOL}")
    if evaluation.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Checkpoint hash does not match {manifest_path}")

    expected_seeds = checkpoint_evaluation_seeds(checkpoint_sha256)
    if evaluation.get("seeds") != expected_seeds:
        raise ValueError(f"Evaluation seeds were not derived from the checkpoint in {manifest_path}")
    if int(evaluation.get("sampling_batch_size", -1)) != official_sampling_batch_size:
        raise ValueError(
            f"{manifest_path} must use official sampling batch size {official_sampling_batch_size}"
        )

    sample_sets = manifest.get("sample_sets")
    if not isinstance(sample_sets, list) or len(sample_sets) != 1:
        raise ValueError(f"{manifest_path} must contain exactly one sample set")
    sample_path = manifest_path.parent / str(sample_sets[0].get("sample_path", ""))
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample file referenced by {manifest_path} is unavailable: {sample_path}")
    if sample_sets[0].get("sample_sha256") != sha256_file(sample_path):
        raise ValueError(f"Generated-sample hash does not match {manifest_path}")
    sample_payload = json.loads(sample_path.read_text())
    try:
        training_seed = int(evaluation["training_seed"])
        sample_training_seed = int(sample_payload["training_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing integer training seed in {manifest_path} or {sample_path}") from exc
    if sample_training_seed != training_seed:
        raise ValueError(f"Training seed mismatch between {manifest_path} and {sample_path}")

    sample_rows = sample_payload.get("samples")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise ValueError(f"Missing samples in {sample_path}")
    trainer = sample_payload.get("trainer")
    if trainer not in {"ar", "dlm"}:
        raise ValueError(f"Unsupported trainer in {sample_path}: {trainer!r}")
    for index, row in enumerate(sample_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {index} in {sample_path} is not an object")
        if int(row.get("sample_index", -1)) != index:
            raise ValueError(f"Sample indices are missing, duplicated, or reordered in {sample_path}")
        if trainer == "dlm":
            batch_start = (index // official_sampling_batch_size) * official_sampling_batch_size
            expected_seed = expected_seeds["sampling"] + batch_start
            expected_batch_size = min(official_sampling_batch_size, len(sample_rows) - batch_start)
        else:
            expected_seed = expected_seeds["sampling"] + index
            expected_batch_size = 1
        if int(row.get("seed", -1)) != expected_seed:
            raise ValueError(f"Sample {index} does not record its actual checkpoint-derived RNG seed")
        if int(row.get("sampling_batch_size", -1)) != expected_batch_size:
            raise ValueError(f"Sample {index} does not use the official sampling batch size")
    return manifest, sample_payload


def materialize_reused_samples(source_manifest: Path, output_dir: Path) -> Path:
    """Copy reused generated samples so the new scorer output remains self-contained."""
    manifest = json.loads(source_manifest.read_text())
    sample_sets = manifest.get("sample_sets")
    if not isinstance(sample_sets, list) or len(sample_sets) != 1:
        raise ValueError(f"{source_manifest} must contain exactly one sample set")
    relative_sample_path = Path(str(sample_sets[0].get("sample_path", "")))
    if relative_sample_path.is_absolute() or ".." in relative_sample_path.parts:
        raise ValueError(f"Sample path in {source_manifest} must stay inside its scorer output directory")
    source_sample = source_manifest.parent / relative_sample_path
    destination_sample = output_dir / relative_sample_path
    destination_sample.parent.mkdir(parents=True, exist_ok=True)
    if source_sample.resolve() != destination_sample.resolve():
        shutil.copyfile(source_sample, destination_sample)

    destination_manifest = output_dir / "sample_manifest.json"
    if source_manifest.resolve() != destination_manifest.resolve():
        shutil.copyfile(source_manifest, destination_manifest)
    source_markdown = source_manifest.parent / "samples.md"
    destination_markdown = output_dir / "samples.md"
    if source_markdown.is_file() and source_markdown.resolve() != destination_markdown.resolve():
        shutil.copyfile(source_markdown, destination_markdown)
    return destination_manifest


def record_scoring_provenance(
    manifest_path: Path,
    rule_path: Path,
    rule: dict[str, Any],
    reference_model: str,
    pinned_reference_model: str,
    args: argparse.Namespace,
) -> None:
    """Bind the generated samples to the exact official scoring configuration and outputs."""
    manifest = json.loads(manifest_path.read_text())
    output_dir = manifest_path.parent
    manifest["scoring"] = {
        "rule_version": rule["version"],
        "rule_sha256": sha256_file(rule_path),
        "reference_model": reference_model,
        "pinned_reference_model": pinned_reference_model,
        "dtype": args.dtype,
        "max_eval_tokens": args.max_eval_tokens,
        "score_batch_size": args.score_batch_size,
        "reference_device_map": args.reference_device_map,
        "allow_fineweb_fallback": bool(args.allow_fineweb_fallback),
        "sample_scores_sha256": sha256_file(output_dir / "scores" / "sample_scores.csv"),
        "sample_outcomes_sha256": sha256_file(output_dir / "sample_outcomes.csv"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_pass(row: dict[str, str], rule: dict[str, Any]) -> bool:
    preset = dict(UNCONDITIONAL_PRESET)
    preset["max_bad"] = float(rule["max_bad_char_count"])
    checks = [
        ("whitespace_token_count", ">=", preset["min_tokens"]),
        ("repeat_trigram_ratio", "<=", preset["max_repeat_3gram"]),
        ("repeat_4gram_ratio", "<=", preset["max_repeat_4gram"]),
        ("unique_token_ratio", ">=", preset["min_unique"]),
        ("compression_ratio", ">=", preset["min_compression"]),
        ("compression_ratio", "<=", preset["max_compression"]),
        ("max_token_run", "<=", preset["max_token_run"]),
        ("eot_count", "<=", preset["max_eot"]),
        ("mask_token_count", "<=", preset["max_mask"]),
        ("bad_char_count", "<=", preset["max_bad"]),
    ]
    for metric, op, threshold in checks:
        value = as_float(row.get(metric))
        if not finite(value):
            return False
        if op == ">=" and value < float(threshold):
            return False
        if op == "<=" and value > float(threshold):
            return False
    return True


def sample_pass(row: dict[str, str], rule: dict[str, Any]) -> bool:
    if row.get("protocol") != "unconditional":
        return False
    if not deterministic_pass(row, rule):
        return False

    bpb = as_float(row.get("ref_bpb"))
    if not finite(bpb) or bpb > float(rule["bpb_threshold"]):
        return False

    pair_count = as_float(row.get("order_pair_count"))
    order_acc = as_float(row.get("order_contrast_acc"))
    if not finite(pair_count) or pair_count < float(rule["order_min_pairs"]):
        return False
    if not finite(order_acc) or order_acc < float(rule["order_contrast_acc_threshold"]):
        return False

    doc_count = as_float(row.get("doc_coherence_corruption_count"))
    doc_margin = as_float(row.get("doc_coherence_margin_bpb_median"))
    if not finite(doc_count) or doc_count < float(rule["doc_coherence_min_corruptions"]):
        return False
    if not finite(doc_margin) or doc_margin < float(rule["doc_coherence_margin_threshold"]):
        return False
    return True


def timing(stage: str, seconds: float, **fields: Any) -> None:
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"TIMING score_generation_quality.{stage} seconds={seconds:.3f}{suffix}", flush=True)


def run_command(command: list[str], cwd: Path, stage: str) -> float:
    print("+ " + " ".join(command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    elapsed = time.perf_counter() - start
    timing(stage, elapsed)
    return elapsed


def generate_sample_manifest(
    checkpoint_path: str,
    checkpoint_sha256: str,
    evaluation_seeds: dict[str, int],
    output_dir: Path,
    tokens: int,
    samples: int,
    temperature: float,
    top_k: int,
    dlm_sampler: str,
    checkpoint_variant: str,
    amp_dtype: str,
    num_sampling_steps: int,
    sampling_eps: float,
    sampling_batch_size: int,
) -> Path:
    if not checkpoint_path:
        raise ValueError("Pass the checkpoint whose contents determine evaluation randomness.")
    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_label = checkpoint.name or checkpoint_path
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_json = samples_dir / "unconditional_samples.json"
    command = [
        sys.executable,
        "-m",
        "speedrun_dlm.sample_text",
        str(checkpoint),
        "--max_new_tokens",
        str(tokens),
        "--samples_per_prompt",
        str(samples),
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--seed",
        str(evaluation_seeds["sampling"]),
        "--dlm_sampler",
        dlm_sampler,
        "--checkpoint_variant",
        checkpoint_variant,
        "--amp_dtype",
        amp_dtype,
        "--num_sampling_steps",
        str(num_sampling_steps),
        "--sampling_eps",
        str(sampling_eps),
        "--sampling_batch_size",
        str(sampling_batch_size),
        "--output_json",
        str(sample_json),
        "--output_md",
        str(output_dir / "samples.md"),
    ]
    sample_seconds = run_command(command, REPO_ROOT, "sample_text")
    sample_payload = json.loads(sample_json.read_text())
    training_seed = sample_payload.get("training_seed")
    if training_seed is None:
        raise ValueError(f"Checkpoint {checkpoint} does not record its training seed")
    manifest_path = output_dir / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "description": "Generated samples and checkpoint evaluation metadata.",
                "evaluation": {
                    "seed_protocol": SEED_PROTOCOL,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_seed": training_seed,
                    "sampling_batch_size": sampling_batch_size,
                    "seeds": evaluation_seeds,
                },
                "sample_sets": [
                    {
                        "name": "checkpoint_unconditional",
                        "family": "candidate",
                        "protocol": "unconditional",
                        "sample_path": str(sample_json.relative_to(output_dir)),
                        "sample_sha256": sha256_file(sample_json),
                        "source_path": checkpoint_label,
                        "subjective_tier": None,
                        "gate_label": "candidate_pass",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    timing("generate_samples_total", sample_seconds)
    return manifest_path


def score_generated_samples(
    manifest_path: Path,
    output_dir: Path,
    reference_model: str,
    pinned_reference_model: str,
    dtype: str,
    max_eval_tokens: int,
    score_batch_size: int,
    reference_device_map: str,
    allow_fineweb_fallback: bool,
) -> Path:
    command = [
        sys.executable,
        "-m",
        "speedrun_dlm._quality_metrics",
        str(manifest_path),
        "--output_dir",
        str(output_dir / "scores"),
        "--reference_models",
        reference_model,
        *(["--pinned_reference_model", pinned_reference_model] if pinned_reference_model else []),
        "--dtype",
        dtype,
        "--max_eval_tokens",
        str(max_eval_tokens),
        "--score_batch_size",
        str(score_batch_size),
        *(["--device_map", reference_device_map] if reference_device_map else []),
        "--order_contrast",
        "--order_contrast_pairs_per_sample",
        str(ORDER_CONTRAST["pairs_per_sample"]),
        "--order_contrast_distractors",
        str(ORDER_CONTRAST["distractors"]),
        "--order_contrast_context_sentences",
        str(ORDER_CONTRAST["context_sentences"]),
        "--order_contrast_min_sentence_tokens",
        str(ORDER_CONTRAST["min_sentence_tokens"]),
        "--order_contrast_max_sentence_tokens",
        str(ORDER_CONTRAST["max_sentence_tokens"]),
        "--order_contrast_distractor_source",
        "fineweb_controls",
        "--order_contrast_distractor_fineweb_count",
        str(ORDER_CONTRAST["distractor_fineweb_count"]),
        "--document_coherence",
        "--doc_coherence_min_sentences",
        str(DOCUMENT_COHERENCE["min_sentences"]),
        "--doc_coherence_max_sentences",
        str(DOCUMENT_COHERENCE["max_sentences"]),
        "--doc_coherence_max_sentence_tokens",
        str(DOCUMENT_COHERENCE["max_sentence_tokens"]),
        "--doc_coherence_shuffles",
        str(DOCUMENT_COHERENCE["shuffles"]),
        "--doc_coherence_swaps",
        str(DOCUMENT_COHERENCE["swaps"]),
        "--doc_coherence_replacements",
        str(DOCUMENT_COHERENCE["replacements"]),
        "--doc_coherence_block_replacements",
        str(DOCUMENT_COHERENCE["block_replacements"]),
        "--doc_coherence_tail_splices",
        str(DOCUMENT_COHERENCE["tail_splices"]),
        "--doc_coherence_replacement_sentences",
        str(DOCUMENT_COHERENCE["replacement_sentences"]),
        "--doc_coherence_block_sentences",
        str(DOCUMENT_COHERENCE["block_sentences"]),
    ]
    if allow_fineweb_fallback:
        command.append("--allow_fineweb_fallback")
    run_command(command, REPO_ROOT, "quality_metrics")
    return output_dir / "scores" / "sample_scores.csv"


def write_sample_outcomes(
    sample_scores: Path,
    output_dir: Path,
    rule: dict[str, Any],
    reference_model: str,
    temperature: float,
    top_k: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    rows = [
        row
        for row in read_csv(sample_scores)
        if row.get("reference_model") == reference_model and row.get("protocol") == "unconditional"
    ]
    if not rows:
        raise ValueError(f"No unconditional rows found for reference model {reference_model!r}.")

    summaries: list[dict[str, Any]] = []
    passed = 0
    for row in rows:
        ok = sample_pass(row, rule)
        passed += int(ok)
        summaries.append(
            {
                "sample_id": row.get("sample_id", row.get("sample_index", "")),
                "protocol": row.get("protocol", ""),
                "reference_model": reference_model,
                "pass": int(ok),
                "ref_bpb": row.get("ref_bpb", ""),
                "order_contrast_acc": row.get("order_contrast_acc", ""),
                "doc_coherence_margin_bpb_median": row.get("doc_coherence_margin_bpb_median", ""),
                "repeat_trigram_ratio": row.get("repeat_trigram_ratio", ""),
                "bad_char_count": row.get("bad_char_count", ""),
            }
        )

    total = len(rows)
    pass_rate = passed / total
    write_csv(output_dir / "sample_outcomes.csv", summaries)

    lines = [
        "# Checkpoint Sample Outcomes",
        "",
        f"- reference model: `{reference_model}`",
        f"- evaluation randomness: `checkpoint-derived ({SEED_PROTOCOL})`",
        f"- sampling temperature: `{temperature}`",
        f"- sampling top_k: `{top_k}`",
        f"- samples: `{total}`",
        f"- passing samples: `{passed}`",
        f"- pass rate: `{pass_rate:.4f}`",
        "",
        "This is a descriptive result for one checkpoint. The official quality-gate result is computed by "
        "`records/make_record.py` after pooling the sample outcomes from all 50 training seeds.",
        "",
        "## Rule",
        "",
        f"- surface filters: `whitespace_token_count >= {UNCONDITIONAL_PRESET['min_tokens']}`, "
        f"`repeat_trigram_ratio <= {UNCONDITIONAL_PRESET['max_repeat_3gram']}`, "
        f"`repeat_4gram_ratio <= {UNCONDITIONAL_PRESET['max_repeat_4gram']}`, "
        f"`unique_token_ratio >= {UNCONDITIONAL_PRESET['min_unique']}`, "
        f"`compression_ratio` in `[{UNCONDITIONAL_PRESET['min_compression']}, {UNCONDITIONAL_PRESET['max_compression']}]`, "
        f"`max_token_run <= {UNCONDITIONAL_PRESET['max_token_run']}`, "
        f"`eot_count <= {UNCONDITIONAL_PRESET['max_eot']}`, "
        f"`mask_token_count <= {UNCONDITIONAL_PRESET['max_mask']}`",
        f"- BPB: `ref_bpb <= {rule['bpb_threshold']}`",
        f"- order contrast: `order_pair_count >= {rule['order_min_pairs']}` "
        f"and `order_contrast_acc >= {rule['order_contrast_acc_threshold']}`",
        f"- document coherence: `doc_coherence_corruption_count >= {rule['doc_coherence_min_corruptions']}` "
        f"and `doc_coherence_margin_bpb_median >= {rule['doc_coherence_margin_threshold']}`",
        f"- bad characters: `bad_char_count <= {rule['max_bad_char_count']}`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    return total, passed, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unconditional samples from one checkpoint and record their quality outcomes."
    )
    parser.add_argument("checkpoint", help="Checkpoint whose contents determine all evaluation randomness.")
    parser.add_argument(
        "--reuse_samples_from",
        default="",
        metavar="OUTPUT_DIR",
        help="Reuse generated samples from a previous scorer output directory for this exact checkpoint.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rule_json", default=str(REPO_ROOT / "generation_quality_rule.json"))
    parser.add_argument("--reference_model", default="")
    parser.add_argument("--pinned_reference_model", default="")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--dlm_sampler",
        default="auto",
        help="DLM sampler name, e.g. subs_mask_ancestral, d3pm_mask_ancestral, sedd_analytic, duo_psi_rescale.",
    )
    parser.add_argument("--checkpoint_variant", choices=("auto", "model", "ema_model"), default="model")
    parser.add_argument("--amp_dtype", choices=("auto", "none", "bfloat16"), default="auto")
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--sampling_eps", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max_eval_tokens", type=int, default=4096)
    parser.add_argument("--score_batch_size", type=int, default=4)
    parser.add_argument(
        "--reference_device_map",
        default="",
        help="Optional Hugging Face device_map for sharded reference-model scoring.",
    )
    parser.add_argument("--allow_fineweb_fallback", action="store_true")
    args = parser.parse_args()

    total_start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_path = Path(args.rule_json).resolve()
    rule = json.loads(rule_path.read_text())
    official_sampling_batch_size = int(rule["official_sampling_batch_size"])
    if official_sampling_batch_size < 1:
        raise ValueError("official_sampling_batch_size must be positive")
    reference_model = args.reference_model or rule["candidate_reference_model"]
    pinned_reference_model = args.pinned_reference_model or rule.get("pinned_reference_model", "")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    evaluation_seeds = checkpoint_evaluation_seeds(checkpoint_sha256)

    if args.reuse_samples_from:
        source_manifest = Path(args.reuse_samples_from).resolve() / "sample_manifest.json"
        load_verified_sample_manifest(
            source_manifest,
            checkpoint_sha256,
            official_sampling_batch_size,
        )
        manifest_path = materialize_reused_samples(source_manifest, output_dir)
        _, sample_payload = load_verified_sample_manifest(
            manifest_path,
            checkpoint_sha256,
            official_sampling_batch_size,
        )
    else:
        manifest_path = generate_sample_manifest(
            args.checkpoint,
            checkpoint_sha256,
            evaluation_seeds,
            output_dir,
            args.tokens,
            args.samples,
            args.temperature,
            args.top_k,
            args.dlm_sampler,
            args.checkpoint_variant,
            args.amp_dtype,
            args.num_sampling_steps,
            args.sampling_eps,
            official_sampling_batch_size,
        )
        _, sample_payload = load_verified_sample_manifest(
            manifest_path,
            checkpoint_sha256,
            official_sampling_batch_size,
        )

    sample_rows = sample_payload["samples"]
    temperatures = {float(row["temperature"]) for row in sample_rows}
    top_ks = {int(row["top_k"]) for row in sample_rows}
    if len(temperatures) != 1 or len(top_ks) != 1:
        raise ValueError(f"Samples in {manifest_path} do not use one sampling configuration")
    sample_scores = score_generated_samples(
        manifest_path,
        output_dir,
        reference_model,
        pinned_reference_model,
        args.dtype,
        args.max_eval_tokens,
        args.score_batch_size,
        args.reference_device_map,
        args.allow_fineweb_fallback,
    )
    summarize_start = time.perf_counter()
    total_samples, passing_samples, _ = write_sample_outcomes(
        sample_scores,
        output_dir,
        rule,
        reference_model,
        temperature=temperatures.pop(),
        top_k=top_ks.pop(),
    )
    record_scoring_provenance(
        manifest_path,
        rule_path,
        rule,
        reference_model,
        pinned_reference_model,
        args,
    )
    timing("write_sample_outcomes", time.perf_counter() - summarize_start)
    print(
        f"QUALITY_SAMPLE_RESULT samples={total_samples} passing_samples={passing_samples} "
        f"pass_rate={passing_samples / total_samples:.6f} report={output_dir / 'README.md'}",
        flush=True,
    )
    timing("total", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
