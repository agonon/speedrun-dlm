import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Release gate settings passed through to _quality_metrics.
ORDER_CONTRAST = {
    "pairs_per_sample": 8,
    "distractors": 4,
    "context_sentences": 3,
    "min_sentence_tokens": 5,
    "max_sentence_tokens": 80,
    "seed": 1337,
    "distractor_fineweb_count": 128,
    "distractor_seed": 4242,
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
    "seed": 2027,
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


def binom_tail_ge(n: int, x: int, p0: float) -> float:
    total = 0.0
    for k in range(x, n + 1):
        total += math.comb(n, k) * (p0**k) * ((1.0 - p0) ** (n - k))
    return min(1.0, max(0.0, total))


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


def generate_panel(
    checkpoint_path: str,
    output_dir: Path,
    tokens: int,
    samples: int,
    temperature: float,
    top_k: int,
    seed: int,
    dlm_sampler: str,
    checkpoint_variant: str,
    amp_dtype: str,
    num_sampling_steps: int,
    sampling_eps: float,
    sampling_batch_size: int,
) -> Path:
    if not checkpoint_path:
        raise ValueError("Pass a checkpoint path, or provide --panel_json to score existing samples.")
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
        str(seed),
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
    panel_json = output_dir / "panel.json"
    panel_json.write_text(
        json.dumps(
            {
                "version": 1,
                "description": "Generation-quality gate panel generated from a checkpoint.",
                "panels": [
                    {
                        "name": "checkpoint_unconditional",
                        "family": "candidate",
                        "protocol": "unconditional",
                        "sample_path": str(sample_json.relative_to(output_dir)),
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
    timing("generate_panel_total", sample_seconds)
    return panel_json


def score_panel(
    panel_json: Path,
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
        str(panel_json),
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
        "--order_contrast_seed",
        str(ORDER_CONTRAST["seed"]),
        "--order_contrast_distractor_source",
        "fineweb_controls",
        "--order_contrast_distractor_fineweb_count",
        str(ORDER_CONTRAST["distractor_fineweb_count"]),
        "--order_contrast_distractor_seed",
        str(ORDER_CONTRAST["distractor_seed"]),
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
        "--doc_coherence_seed",
        str(DOCUMENT_COHERENCE["seed"]),
    ]
    if allow_fineweb_fallback:
        command.append("--allow_fineweb_fallback")
    run_command(command, REPO_ROOT, "quality_metrics")
    return output_dir / "scores" / "sample_scores.csv"


def summarize_scores(
    sample_scores: Path,
    output_dir: Path,
    rule: dict[str, Any],
    reference_model: str,
    require_significance: bool,
    sample_seed: int | None,
    temperature: float | None,
    top_k: int | None,
) -> tuple[bool, list[dict[str, Any]]]:
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
    threshold = float(rule["pass_rate_threshold"])
    p_value = binom_tail_ge(total, passed, threshold)
    significance_pass = not require_significance or p_value < float(rule["significance_p_value"])
    overall_pass = pass_rate >= threshold and significance_pass
    write_csv(output_dir / "sample_passes.csv", summaries)

    lines = [
        "# Quality Gate Result",
        "",
        f"- reference model: `{reference_model}`",
        f"- sampling seed: `{sample_seed}`"
        if sample_seed is not None
        else "- sampling seed: `not applicable (precomputed panel)`",
        f"- sampling temperature: `{temperature}`"
        if temperature is not None
        else "- sampling temperature: `not applicable (precomputed panel)`",
        f"- sampling top_k: `{top_k}`"
        if top_k is not None
        else "- sampling top_k: `not applicable (precomputed panel)`",
        f"- samples: `{total}`",
        f"- passing samples: `{passed}`",
        f"- pass rate: `{pass_rate:.4f}`",
        f"- required pass rate: `{threshold:.4f}`",
        f"- exact binomial p-value vs threshold: `{p_value:.6g}`",
        f"- significance required: `{require_significance}`",
        f"- overall pass: `{overall_pass}`",
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
    return overall_pass, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unconditional samples from a checkpoint and apply the v0 quality gate.")
    parser.add_argument("checkpoint", nargs="?", default="", help="Checkpoint to sample. Omit when using --panel_json.")
    parser.add_argument("--panel_json", default="", help="Existing panel JSON to score instead of sampling a checkpoint.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rule_json", default=str(REPO_ROOT / "generation_quality_rule.json"))
    parser.add_argument("--reference_model", default="")
    parser.add_argument("--pinned_reference_model", default="")
    parser.add_argument("--require_significance", action="store_true")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dlm_sampler",
        default="auto",
        help="DLM sampler name, e.g. subs_mask_ancestral, d3pm_mask_ancestral, sedd_analytic, duo_psi_rescale.",
    )
    parser.add_argument("--checkpoint_variant", choices=("auto", "model", "ema_model"), default="model")
    parser.add_argument("--amp_dtype", choices=("auto", "none", "bfloat16"), default="auto")
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--sampling_eps", type=float, default=1e-5)
    parser.add_argument(
        "--sampling_batch_size",
        type=int,
        default=1,
        help="Number of unconditional DLM samples to generate per batch; AR checkpoints ignore this.",
    )
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
    output_dir.mkdir(parents=True, exist_ok=True)
    rule = json.loads(Path(args.rule_json).read_text())
    reference_model = args.reference_model or rule["candidate_reference_model"]
    pinned_reference_model = args.pinned_reference_model or rule.get("pinned_reference_model", "")

    panel_json = (
        Path(args.panel_json).resolve()
        if args.panel_json
        else generate_panel(
            args.checkpoint,
            output_dir,
            args.tokens,
            args.samples,
            args.temperature,
            args.top_k,
            args.seed,
            args.dlm_sampler,
            args.checkpoint_variant,
            args.amp_dtype,
            args.num_sampling_steps,
            args.sampling_eps,
            args.sampling_batch_size,
        )
    )
    sample_scores = score_panel(
        panel_json,
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
    passed, _ = summarize_scores(
        sample_scores,
        output_dir,
        rule,
        reference_model,
        require_significance=args.require_significance,
        sample_seed=None if args.panel_json else args.seed,
        temperature=None if args.panel_json else args.temperature,
        top_k=None if args.panel_json else args.top_k,
    )
    timing("summarize_scores", time.perf_counter() - summarize_start)
    print(f"QUALITY_GATE_RESULT pass={str(passed).lower()} report={output_dir / 'README.md'}", flush=True)
    timing("total", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
