#!/usr/bin/env python3
"""Zero-shot HellaSwag evaluation for the benchmark DLM objectives.

The prompt remains clean and only continuation tokens are corrupted.  Every
answer choice for an example receives the same sampled time and the same
underlying random stream.  Scores are Monte Carlo estimates of the
objective-native conditional variational quantity; only within-model option
orders are compared, never the raw values across objectives.

SUBS uses the conditional masked-token estimator also used by LLaDA.  D3PM,
SEDD and DUO use the corresponding per-token terms implemented by the pinned
training code, restricted to answer positions.  This is direct zero-shot
evaluation: no checkpoint weight or task head is fitted.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from speedrun_dlm.checkpoint_utils import load_snapshot
from speedrun_dlm.train_dlm import (
    TimeBatch,
    d3pm_reverse_posterior_probs,
    duo_input_inverse_temperature,
    duo_keep_prob_and_rate_from_table,
    duo_uniform_nll_per_token,
    keep_prob_from_time,
    load_duo_keep_prob_table,
    model_forward_noisy,
    sample_duo_forward_relaxation,
    total_noise_rate_from_time,
)
from speedrun_dlm._hellaswag import (
    Candidate,
    REFERENCE_MC_BATCH_SIZE,
    build_candidates,
    load_rows,
    reference_mask_counts,
    sha256_file,
    stable_uint64,
    summarize_multiple_choice,
    write_jsonl,
)


# This identifier also seeds the deterministic corruption stream. Keep it
# fixed so released evaluations reproduce the measured benchmark protocol.
VERSION = "dlm-zero-shot-objective-mc-v3-protocol-audit"
MASK_OBJECTIVES = {"subs_mask", "d3pm_mask", "sedd_mask"}
UNIFORM_OBJECTIVES = {"d3pm_uniform", "sedd_uniform", "duo_uniform"}


def coupled_uniform(example_id: str, draw: int, seed: int, purpose: str) -> float:
    """One open-interval uniform shared by all choices for an example."""
    integer = stable_uint64(VERSION, seed, example_id, draw, purpose)
    return (integer + 0.5) / float(2**64)


def radical_inverse_base2(index: int) -> float:
    """Base-two radical inverse, used to make power-of-two prefixes balanced."""
    if index < 0:
        raise ValueError("index must be non-negative")
    value = 0.0
    scale = 0.5
    while index:
        value += scale * (index & 1)
        index >>= 1
        scale *= 0.5
    return value


def sampling_uniform(
    example_id: str,
    draw: int,
    seed: int,
    purpose: str,
    scheme: str,
) -> float:
    """Return either the reference iid draw or a randomized lattice draw.

    A random rotation keeps the lattice estimator unbiased.  The radical-
    inverse order makes every power-of-two prefix cover the unit interval,
    which lets the convergence pilot compare 1, 2, ..., 128 draws fairly.
    """
    if scheme == "iid":
        return coupled_uniform(example_id, draw, seed, purpose)
    if scheme != "randomized_lattice":
        raise ValueError(f"unknown sampling scheme: {scheme}")
    shift = coupled_uniform(example_id, 0, seed, purpose + "-lattice-shift")
    return (radical_inverse_base2(draw) + shift) % 1.0


def d3pm_per_token_vb(
    raw_logits: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    time_batch: TimeBatch,
    noise_type: str,
    mask_token_id: int,
) -> torch.Tensor:
    predicted = d3pm_reverse_posterior_probs(
        raw_logits,
        noisy,
        time_batch,
        noise_type=noise_type,
        vocab_size=raw_logits.shape[-1],
        mask_token_id=mask_token_id,
        x0_is_logits=True,
    )
    truth = d3pm_reverse_posterior_probs(
        clean,
        noisy,
        time_batch,
        noise_type=noise_type,
        vocab_size=raw_logits.shape[-1],
        mask_token_id=mask_token_id,
        x0_is_logits=False,
    )
    tiny = torch.finfo(predicted.dtype).tiny
    return (
        truth
        * (truth.clamp_min(tiny).log() - predicted.clamp_min(tiny).log())
    ).sum(dim=-1)


def sedd_mask_per_token(
    raw_logits: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    time_batch: TimeBatch,
    mask_token_id: int,
) -> torch.Tensor:
    tiny = torch.finfo(time_batch.keep_prob.dtype).tiny
    keep = time_batch.keep_prob.clamp_min(tiny)
    noise = (1.0 - keep).clamp_min(tiny)
    log_scores = raw_logits.float() - (noise.log() - keep.log())[:, None, None]
    log_scores = log_scores - math.log(raw_logits.shape[-1])
    mask_column = torch.zeros(
        *log_scores.shape[:2], 1, device=log_scores.device, dtype=log_scores.dtype
    )
    log_scores = torch.cat([log_scores, mask_column], dim=-1)
    log_scores = torch.scatter(
        log_scores, -1, noisy.unsqueeze(-1), torch.zeros_like(log_scores[..., :1])
    )
    selected = noisy.eq(mask_token_id)
    entropy = torch.zeros_like(noisy, dtype=log_scores.dtype)
    if selected.any():
        ratio = (keep / noise).unsqueeze(1).expand_as(noisy)[selected]
        scores = log_scores[selected]
        targets = clean[selected]
        negative = ratio * scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        positive = scores[:, :-1].exp().sum(dim=-1)
        constant = ratio * (ratio.log() - 1.0)
        entropy[selected] = positive - negative + constant
    return time_batch.loss_weight.unsqueeze(1) * entropy


def sedd_uniform_per_token(
    raw_logits: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    time_batch: TimeBatch,
) -> torch.Tensor:
    vocab_size = raw_logits.shape[-1]
    log_scores = torch.scatter(
        raw_logits.float(),
        -1,
        noisy.unsqueeze(-1),
        torch.zeros_like(raw_logits[..., :1], dtype=torch.float32),
    )
    tiny = torch.finfo(time_batch.keep_prob.dtype).tiny
    keep = time_batch.keep_prob.clamp_min(tiny).unsqueeze(1)
    noise = (1.0 - time_batch.keep_prob).clamp_min(tiny).unsqueeze(1)
    noise_to_keep = noise / keep
    ratio = noise_to_keep / (noise_to_keep + vocab_size)
    clean_score = log_scores.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
    negative = log_scores.mean(dim=-1)
    same = noisy.eq(clean)
    negative = torch.where(same, ratio * negative, clean_score * keep / noise + negative)
    ratio_log = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
    constant_same = (vocab_size - 1) / vocab_size * ratio * (ratio_log.log() - 1.0)
    constant_different = ((-ratio_log.log() - 1.0) / ratio_log - (vocab_size - 2)) / vocab_size
    constant = torch.where(same, constant_same, constant_different)
    positive = log_scores.exp().mean(dim=-1) - (1.0 / vocab_size)
    return time_batch.loss_weight.unsqueeze(1) * (positive - negative + constant)


def objective_per_token_terms(
    objective: str,
    raw_logits: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    time_batch: TimeBatch,
    mask_token_id: int,
) -> torch.Tensor:
    if objective == "subs_mask":
        nll = -F.log_softmax(raw_logits.float(), dim=-1).gather(
            -1, clean.unsqueeze(-1)
        ).squeeze(-1)
        weights = noisy.eq(mask_token_id) * time_batch.loss_weight.unsqueeze(1)
        return nll * weights
    if objective in {"d3pm_mask", "d3pm_uniform"}:
        return d3pm_per_token_vb(
            raw_logits,
            clean,
            noisy,
            time_batch,
            noise_type="mask" if objective.endswith("mask") else "uniform",
            mask_token_id=mask_token_id,
        )
    if objective == "sedd_mask":
        return sedd_mask_per_token(raw_logits, clean, noisy, time_batch, mask_token_id)
    if objective == "sedd_uniform":
        return sedd_uniform_per_token(raw_logits, clean, noisy, time_batch)
    if objective == "duo_uniform":
        if time_batch.duo_discrete_noisy_tokens is None:
            raise RuntimeError("DUO forward did not record its discrete noisy state")
        return duo_uniform_nll_per_token(
            raw_logits,
            time_batch.duo_discrete_noisy_tokens,
            clean,
            time_batch.keep_prob.unsqueeze(-1),
            time_batch.loss_weight.unsqueeze(-1),
            raw_logits.shape[-1],
        )
    raise ValueError(f"unsupported objective: {objective}")


def continuous_time_batch(
    objective: str,
    units: torch.Tensor,
    args: dict[str, Any],
    duo_table: dict[str, torch.Tensor | float | int] | None,
    vocab_size: int,
) -> TimeBatch:
    eps = float(args.get("continuous_time_eps", 1e-3))
    times = eps + (1.0 - eps) * units
    if objective == "duo_uniform":
        if duo_table is None:
            raise RuntimeError("DUO evaluation requires its pinned keep-probability table")
        lower = float(args.get("duo_log_noise_ratio_min", -3.55))
        upper = float(args.get("duo_log_noise_ratio_max", -1.85))
        log_ratio = lower + times * (upper - lower)
        keep, rate = duo_keep_prob_and_rate_from_table(
            log_ratio,
            duo_table,
            vocab_size=vocab_size,
            log_noise_ratio_min=lower,
            log_noise_ratio_max=upper,
        )
        return TimeBatch(
            model_times=-torch.log(keep.clamp_min(1e-12)),
            keep_prob=keep,
            loss_weight=rate,
            duo_log_noise_ratio=log_ratio,
        )
    schedule = str(args.get("noise_schedule", "loglinear"))
    geometric_min = float(args.get("geometric_noise_level_min", 1e-5))
    geometric_max = float(args.get("geometric_noise_level_max", 20.0))
    keep = keep_prob_from_time(times, schedule, eps, geometric_min, geometric_max)
    rate = total_noise_rate_from_time(times, schedule, eps, geometric_min, geometric_max)
    if objective == "subs_mask":
        weight = rate * keep / (1.0 - keep)
    else:
        weight = rate
    return TimeBatch(model_times=-torch.log(keep), keep_prob=keep, loss_weight=weight)


def d3pm_time_batch(
    units: torch.Tensor,
    args: dict[str, Any],
) -> tuple[TimeBatch, int]:
    steps_count = int(args.get("num_diffusion_steps", 100))
    steps = torch.clamp((units * steps_count).floor().long() + 1, 1, steps_count)
    eps = float(args.get("continuous_time_eps", 1e-3))
    schedule = str(args.get("noise_schedule", "linear"))
    grid_times = torch.linspace(0.0, 1.0, steps_count + 1, device=units.device)
    grid = keep_prob_from_time(
        grid_times,
        schedule,
        eps,
        float(args.get("geometric_noise_level_min", 1e-5)),
        float(args.get("geometric_noise_level_max", 20.0)),
    )
    return (
        TimeBatch(
            model_times=steps.float() / steps_count,
            keep_prob=grid[steps],
            prev_keep_prob=grid[steps - 1],
        ),
        steps_count,
    )


def sample_discrete_corruption(
    clean: torch.Tensor,
    target: torch.Tensor,
    time_batch: TimeBatch,
    objective: str,
    candidates: list[Candidate],
    draw: int,
    seed: int,
    mask_token_id: int,
    vocab_size: int,
    sampling_scheme: str,
) -> torch.Tensor:
    noisy = clean.clone()
    for row, candidate in enumerate(candidates):
        count = candidate.target_tokens
        keep = float(time_batch.keep_prob[row].item())
        if sampling_scheme == "iid":
            rng = np.random.default_rng(
                stable_uint64(VERSION, seed, candidate.example_id, draw, "corruption")
            )
            uniforms = rng.random(count)
            replacement_uniforms = rng.random(count)
        elif sampling_scheme == "randomized_lattice":
            base = radical_inverse_base2(draw)
            change_rng = np.random.default_rng(
                stable_uint64(VERSION, seed, candidate.example_id, "corruption-change-shifts")
            )
            replacement_rng = np.random.default_rng(
                stable_uint64(VERSION, seed, candidate.example_id, "corruption-token-shifts")
            )
            uniforms = (change_rng.random(count) + base) % 1.0
            replacement_uniforms = (replacement_rng.random(count) + base) % 1.0
        else:
            raise ValueError(f"unknown sampling scheme: {sampling_scheme}")
        changed = uniforms >= keep
        positions = np.arange(candidate.target_start, candidate.target_end)[changed]
        if objective in MASK_OBJECTIVES:
            if len(positions):
                noisy[row, torch.as_tensor(positions, device=noisy.device)] = mask_token_id
        else:
            replacements = np.floor(
                replacement_uniforms[changed] * vocab_size
            ).astype(np.int64)
            if len(positions):
                noisy[row, torch.as_tensor(positions, device=noisy.device)] = torch.as_tensor(
                    replacements, dtype=torch.long, device=noisy.device
                )
    return noisy


def sample_subs_stratified(
    clean: torch.Tensor,
    candidates: list[Candidate],
    draw: int,
    seed: int,
    mask_token_id: int,
    continuous_time_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, TimeBatch]:
    """LLaDA's stratified answer-mask estimator with choice-coupled draws."""
    noisy = clean.clone()
    selected = torch.zeros_like(clean, dtype=torch.bool)
    probabilities = []
    block, within_block = divmod(draw, REFERENCE_MC_BATCH_SIZE)
    for row, candidate in enumerate(candidates):
        count = candidate.target_tokens
        count_rng = np.random.default_rng(
            stable_uint64(VERSION, seed, candidate.example_id, block, "subs-count-start")
        )
        count_start = int(count_rng.integers(1, count + 1))
        masked_count = int(
            reference_mask_counts(count, REFERENCE_MC_BATCH_SIZE, count_start)[within_block]
        )
        position_rng = np.random.default_rng(
            stable_uint64(VERSION, seed, candidate.example_id, draw, "subs-positions")
        )
        relative = np.sort(position_rng.choice(count, size=masked_count, replace=False))
        positions = torch.as_tensor(
            relative + candidate.target_start, dtype=torch.long, device=clean.device
        )
        selected[row, positions] = True
        noisy[row, positions] = mask_token_id
        probabilities.append(masked_count / count)
    probability = torch.tensor(probabilities, dtype=torch.float32, device=clean.device)
    keep = (1.0 - probability).clamp_min(continuous_time_eps)
    return noisy, selected, TimeBatch(
        model_times=-torch.log(keep), keep_prob=keep, loss_weight=None
    )


def prepare_duo_inputs(
    clean: torch.Tensor,
    target: torch.Tensor,
    time_batch: TimeBatch,
    candidates: list[Candidate],
    draw: int,
    seed: int,
    args: dict[str, Any],
    vocab_size: int,
    checkpoint_step: int,
) -> torch.Tensor:
    mode = str(args.get("duo_curriculum_mode", "dense_softmax"))
    top_k = int(args.get("duo_top_k", 32))
    inverse_temperature = duo_input_inverse_temperature(
        checkpoint_step,
        float(args.get("duo_softmax_temperature_log10_start", -3.0)),
        float(args.get("duo_softmax_temperature_log10_end", -3.0)),
        int(args.get("duo_curriculum_start", 0)),
        int(args.get("duo_curriculum_end", 0)),
    )
    noisy_rows = []
    discrete_rows = []
    weight_rows = []
    for row, candidate in enumerate(candidates):
        torch.manual_seed(
            stable_uint64(VERSION, seed, candidate.example_id, draw, "duo") % (2**63 - 1)
        )
        row_time = TimeBatch(
            model_times=time_batch.model_times[row : row + 1],
            keep_prob=time_batch.keep_prob[row : row + 1],
            loss_weight=time_batch.loss_weight[row : row + 1],
            duo_log_noise_ratio=time_batch.duo_log_noise_ratio[row : row + 1],
        )
        row_noisy = sample_duo_forward_relaxation(
            clean[row : row + 1],
            row_time,
            curriculum_mode=mode,
            top_k=top_k,
            inverse_temperature=inverse_temperature,
            vocab_size=vocab_size,
        )
        prompt = ~target[row : row + 1]
        if row_time.duo_sparse_input_weights is None:
            row_noisy[prompt] = float("-inf")
            prompt_rows, prompt_positions = torch.where(prompt)
            row_noisy[prompt_rows, prompt_positions, clean[row : row + 1][prompt]] = 0.0
        else:
            weights = row_time.duo_sparse_input_weights
            row_noisy[prompt, 0] = clean[row : row + 1][prompt]
            weights[prompt, 0] = 1.0
            if weights.shape[-1] > 1:
                weights[prompt, 1:] = 0.0
            weight_rows.append(weights)
        row_time.duo_discrete_noisy_tokens[prompt] = clean[row : row + 1][prompt]
        noisy_rows.append(row_noisy)
        discrete_rows.append(row_time.duo_discrete_noisy_tokens)
    noisy = torch.cat(noisy_rows, dim=0)
    time_batch.duo_discrete_noisy_tokens = torch.cat(discrete_rows, dim=0)
    time_batch.duo_sparse_input_weights = (
        torch.cat(weight_rows, dim=0) if weight_rows else None
    )
    return noisy


@torch.inference_mode()
def evaluate_losses(
    model,
    args: dict[str, Any],
    candidates: list[Candidate],
    mc_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    amp,
    duo_table: dict[str, torch.Tensor | float | int] | None,
    checkpoint_step: int,
    sampling_scheme: str,
    duo_batch_size: int,
) -> np.ndarray:
    objective = str(args.get("objective"))
    if objective not in MASK_OBJECTIVES | UNIFORM_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    # Dense DUO stores one vocabulary-sized distribution per position.  Its
    # batch is controlled separately, after a memory smoke, rather than making
    # every other objective use this smaller value.
    if objective == "duo_uniform" and str(args.get("duo_curriculum_mode", "dense_softmax")) == "dense_softmax":
        batch_size = min(batch_size, duo_batch_size)
    vocab_size = int(model.config.vocab_size)
    multiplier = 1.0 - float(args.get("continuous_time_eps", 1e-3))
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_length[len(candidate.ids)].append(index)
    losses = np.empty((len(candidates), mc_samples), dtype=np.float64)
    model.eval()
    for draw in range(mc_samples):
        for length in sorted(by_length):
            indexes = by_length[length]
            for start in range(0, len(indexes), batch_size):
                selected_indexes = indexes[start : start + batch_size]
                selected = [candidates[index] for index in selected_indexes]
                clean = torch.tensor(
                    [candidate.ids for candidate in selected], dtype=torch.long, device=device
                )
                target = torch.zeros_like(clean, dtype=torch.bool)
                for row, candidate in enumerate(selected):
                    target[row, candidate.target_start : candidate.target_end] = True
                units = torch.tensor(
                    [
                        sampling_uniform(
                            c.example_id, draw, seed, "time", sampling_scheme
                        )
                        for c in selected
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                subs_selected = None
                if objective == "subs_mask":
                    noisy, subs_selected, time_batch = sample_subs_stratified(
                        clean,
                        selected,
                        draw,
                        seed,
                        int(model.mask_token_id),
                        float(args.get("continuous_time_eps", 1e-3)),
                    )
                    cell_multiplier = 1.0
                elif objective.startswith("d3pm_"):
                    time_batch, multiplier_int = d3pm_time_batch(units, args)
                    cell_multiplier = float(multiplier_int)
                else:
                    time_batch = continuous_time_batch(
                        objective, units, args, duo_table, vocab_size
                    )
                    cell_multiplier = multiplier
                if objective == "subs_mask":
                    pass
                elif objective == "duo_uniform":
                    noisy = prepare_duo_inputs(
                        clean,
                        target,
                        time_batch,
                        selected,
                        draw,
                        seed,
                        args,
                        vocab_size,
                        checkpoint_step,
                    )
                else:
                    noisy = sample_discrete_corruption(
                        clean,
                        target,
                        time_batch,
                        objective,
                        selected,
                        draw,
                        seed,
                        int(model.mask_token_id),
                        vocab_size,
                        sampling_scheme,
                    )
                with amp():
                    raw_logits = model_forward_noisy(model, noisy, time_batch, objective)
                if objective == "subs_mask":
                    assert subs_selected is not None
                    nll = -F.log_softmax(raw_logits.float(), dim=-1).gather(
                        -1, clean.unsqueeze(-1)
                    ).squeeze(-1)
                    masked_counts = subs_selected.sum(dim=-1).clamp_min(1)
                    target_counts = target.sum(dim=-1).clamp_min(1)
                    probability = masked_counts / target_counts
                    terms = nll * subs_selected / probability.unsqueeze(1)
                else:
                    terms = objective_per_token_terms(
                        objective,
                        raw_logits,
                        clean,
                        noisy,
                        time_batch,
                        int(model.mask_token_id),
                    )
                estimates = (terms * target).sum(dim=-1) * cell_multiplier
                losses[selected_indexes, draw] = estimates.float().cpu().numpy()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--prefixes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--duo-keep-prob-table", default="")
    parser.add_argument(
        "--sampling-scheme",
        choices=("iid", "randomized_lattice"),
        default="iid",
    )
    parser.add_argument("--duo-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.duo_batch_size < 1:
        raise ValueError("duo-batch-size must be positive")
    prefixes = sorted({int(value) for value in args.prefixes.split(",") if value})
    if not prefixes or prefixes[-1] > args.mc_samples:
        raise ValueError("prefixes must be non-empty and no larger than mc-samples")
    rows = load_rows(args.data, args.start, args.max_examples)
    candidates, groups, labels = build_candidates(rows, args.max_length)
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type == "cuda" and dtype != torch.float32:
        amp = lambda: torch.autocast(device_type="cuda", dtype=dtype)
    else:
        from contextlib import nullcontext
        amp = nullcontext
    snapshot = load_snapshot(str(args.checkpoint), device=args.device, checkpoint_variant="model")
    if snapshot.trainer != "dlm":
        raise ValueError("checkpoint is not a DLM checkpoint")
    objective = str(snapshot.args.get("objective"))
    checkpoint_payload = torch.load(args.checkpoint, map_location="meta", weights_only=False)
    checkpoint_step = int(checkpoint_payload.get("checkpoint_step", 0))
    duo_table = None
    if objective == "duo_uniform":
        table_path = args.duo_keep_prob_table or str(
            snapshot.args.get("duo_keep_prob_table_path", "")
        )
        duo_table = load_duo_keep_prob_table(table_path, args.device)
    started = time.perf_counter()
    losses = evaluate_losses(
        snapshot.model,
        snapshot.args,
        candidates,
        args.mc_samples,
        args.batch_size,
        args.seed,
        device,
        amp,
        duo_table,
        checkpoint_step,
        args.sampling_scheme,
        args.duo_batch_size,
    )
    example_results, summaries = summarize_multiple_choice(
        rows, candidates, groups, labels, losses, prefixes
    )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "example_results.jsonl.gz", example_results)
    write_jsonl(args.output / "summary.jsonl.gz", summaries)
    manifest = {
        "schema": VERSION,
        "interpretation": (
            "Direct zero-shot option accuracy from objective-native conditional "
            "variational estimates; raw bound values are not compared across objectives."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "total_tokens_processed": checkpoint_payload.get("total_tokens_processed"),
        "objective": objective,
        "task": "hellaswag",
        "data": str(args.data),
        "examples": len(rows),
        "mc_samples": args.mc_samples,
        "prefixes": prefixes,
        "seed": args.seed,
        "sampling_scheme": args.sampling_scheme,
        "duo_batch_size": args.duo_batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summaries[-1],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
