import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint_utils import (
    DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
    autocast_context,
    model_time_from_keep_prob,
    keep_prob_from_time,
    load_snapshot,
    make_d3pm_sampling_keep_prob_grid,
    masked_x0_logprobs,
    total_noise_rate_from_time,
    resolve_amp_dtype,
    tokenizer,
)
from .train_dlm import DLM_OBJECTIVES


DLM_SAMPLER_CHOICES = (
    "auto",
    "subs_mask_ancestral",
    "d3pm_mask_ancestral",
    "d3pm_uniform_ancestral",
    "sedd_euler",
    "sedd_analytic",
    "duo_ancestral",
    "duo_greedy_tail",
    "duo_psi_rescale",
    "duo_psi_capped",
    "duo_psi_loop",
)

UNIFORM_PSI_SAMPLERS = {"duo_psi_rescale", "duo_psi_capped", "duo_psi_loop"}

DUO_PSI_CONFIGS = {
    # DUO Chapter II Psi/ReMDM modes. The formulas are official; eta values here
    # are the explicit benchmark settings used for the release curves.
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/configs/config.yaml#L37-L50
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/scripts/psi_samplers/owt/duo_max_capped_remdm.sh#L1-L30
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/scripts/psi_samplers/owt/duo_max_rescale_eta.sh#L1-L30
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/scripts/psi_samplers/owt/duo_loop_remdm.sh#L1-L30
    "duo_psi_rescale": {
        "time_profile": "linear",
        "high_mode": "max-rescale-0.05",
        "middle_mode": "max-rescale-0.05",
        "low_mode": "max-rescale-0.05",
        "high_frac": 0.0,
        "middle_frac": 0.0,
        "top_p": 0.9,
    },
    "duo_psi_capped": {
        # Same official max-capped mode; this benchmark uses eta=0.05.
        "time_profile": "linear",
        "high_mode": "max-capped-0.05",
        "middle_mode": "max-capped-0.05",
        "low_mode": "max-capped-0.05",
        "high_frac": 0.0,
        "middle_frac": 0.0,
        "top_p": 0.9,
    },
    "duo_psi_loop": {
        "time_profile": "linear-constant-linear-0.9-inv",
        "high_mode": "pure-posterior",
        "middle_mode": "constant-remdm-0.01",
        "low_mode": "pure-posterior",
        "high_frac": 0.45,
        "middle_frac": 0.5,
        "top_p": 0.95,
    },
}


def checkpoint_objective_and_noise(snapshot_args: dict) -> tuple[str, str]:
    objective = snapshot_args.get("objective")
    if objective is None:
        raise NotImplementedError("DLM checkpoint metadata must include an objective.")
    if objective not in DLM_OBJECTIVES:
        raise NotImplementedError(f"Unsupported DLM objective: {objective!r}")
    required_noise_type = DLM_OBJECTIVES[objective].noise_type
    noise_type = snapshot_args.get("noise_type") or required_noise_type
    if noise_type not in {"mask", "uniform"}:
        raise NotImplementedError(f"Unsupported DLM noise_type: {noise_type!r}")
    if noise_type != required_noise_type:
        raise NotImplementedError(
            f"objective={objective} requires noise_type={required_noise_type}, got {noise_type!r}."
        )
    return objective, noise_type


def resolve_dlm_sampler(objective: str, dlm_sampler: str) -> str:
    if dlm_sampler != "auto":
        return dlm_sampler
    if objective == "subs_mask":
        return "subs_mask_ancestral"
    if objective == "d3pm_mask":
        return "d3pm_mask_ancestral"
    if objective == "d3pm_uniform":
        return "d3pm_uniform_ancestral"
    if objective in {"sedd_mask", "sedd_uniform"}:
        return "sedd_analytic"
    if objective == "duo_uniform":
        return "duo_ancestral"
    raise NotImplementedError(f"Unsupported DLM objective: {objective!r}")


# =============================================================================
# SHARED SAMPLING UTILITIES
# =============================================================================


def checkpoint_label(path: str) -> str:
    return Path(path).name or path


def timing(stage: str, seconds: float, **fields) -> None:
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"TIMING sample_text.{stage} seconds={seconds:.3f}{suffix}", flush=True)


def sample_from_probs(
    probs: torch.Tensor,  # [..., vocab] categorical probabilities or nonnegative weights
    generator: torch.Generator,
    greedy: bool,
) -> torch.Tensor:
    if greedy:
        return probs.argmax(dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    draws = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator)
    return draws.view(probs.shape[:-1])


def sample_unnormalized_probs(
    probs: torch.Tensor,  # [..., vocab] unnormalized nonnegative categorical mass
    generator: torch.Generator,
) -> torch.Tensor:
    # Categorical draw from unnormalized nonnegative mass. This is equivalent
    # to Gumbel-max over log-mass and keeps mask/token posterior code direct.
    probs = probs.float()
    uniforms = torch.rand(probs.shape, device=probs.device, dtype=probs.dtype, generator=generator)
    exp_noise = 1e-10 - (uniforms + 1e-10).log()
    return (probs / exp_noise).argmax(dim=-1)


def filter_logits(
    logits: torch.Tensor,  # [..., vocab]
    top_k: int,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        filtered = logits
    else:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        filtered = logits.masked_fill(logits < kth, float("-inf"))
    if top_p >= 1.0:
        return filtered
    if top_p <= 0.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    sorted_remove = sorted_probs.cumsum(dim=-1) > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
    return filtered.masked_fill(remove, float("-inf"))


def next_token_probs(
    logits: torch.Tensor,  # [batch, vocab]
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
    logits = filter_logits(logits.float() / temperature, top_k)
    return torch.softmax(logits, dim=-1)


def predict_x0_probs(
    model,
    current: torch.Tensor,  # [batch, seq]
    model_times: torch.Tensor,  # [batch]
    device: str,
    amp_dtype: str,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
    preserve_unmasked: bool = False,
) -> torch.Tensor:
    with torch.inference_mode():
        with autocast_context(device, amp_dtype):
            raw_logits = model(current, model_times)
    logits = raw_logits.float()
    if temperature > 0.0:
        logits = logits / temperature
    logits = filter_logits(logits, top_k, top_p=top_p)
    if preserve_unmasked:
        return masked_x0_logprobs(logits, current, model.mask_token_id, preserve_unmasked=True).exp()
    return torch.softmax(logits, dim=-1)


# =============================================================================
# DIFFUSION SCHEDULES
# =============================================================================


def require_sampling_steps(num_sampling_steps: int) -> int:
    if num_sampling_steps < 1:
        raise ValueError(f"num_sampling_steps must be at least 1, got {num_sampling_steps}.")
    return num_sampling_steps


def sample_d3pm_keep_prob_grid(
    snapshot_args: dict,
    num_sampling_steps: int,
    device: str,
    terminal_zero: bool = True,
) -> torch.Tensor:
    # D3PM: t_i=i/S, then keep_prob_i=the configured schedule at t_i;
    # the first/last points are forced to fully clean/noisy by default.
    # If S is omitted, use the T-step grid on which D3PM was trained.
    steps = num_sampling_steps or int(snapshot_args.get("num_diffusion_steps", 1000))
    return make_d3pm_sampling_keep_prob_grid(
        steps,
        snapshot_args.get("noise_schedule", "loglinear"),
        float(snapshot_args.get("continuous_time_eps", 1e-3)),
        device=device,
        geometric_noise_level_min=float(snapshot_args.get("geometric_noise_level_min", DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN)),
        geometric_noise_level_max=float(snapshot_args.get("geometric_noise_level_max", DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX)),
        terminal_zero=terminal_zero,
    )


def sample_sedd_noise_grids(snapshot_args: dict, num_sampling_steps: int, sampling_eps: float, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Total noise and instantaneous rate on the sampling time grid.

    The analytic predictor uses total-noise differences; Euler uses dt * rate.
    """
    steps = require_sampling_steps(num_sampling_steps)
    # SEDD: t_i=sampling_eps+i*(1-sampling_eps)/S, independent of training T.
    times = torch.linspace(float(sampling_eps), 1.0, steps + 1, device=device, dtype=torch.float32)
    schedule = snapshot_args.get("noise_schedule", "loglinear")
    eps = float(snapshot_args.get("continuous_time_eps", 1e-3))
    level_min = float(snapshot_args.get("geometric_noise_level_min", DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN))
    level_max = float(snapshot_args.get("geometric_noise_level_max", DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX))
    keep_prob = keep_prob_from_time(times, schedule, eps, level_min, level_max)
    total_noise = -torch.log(keep_prob)
    total_noise_rate = total_noise_rate_from_time(times, schedule, eps, level_min, level_max)
    return total_noise, total_noise_rate


def sample_duo_official_keep_grid(
    snapshot_args: dict,
    num_sampling_steps: int,
    sampling_eps: float,
    device: str,
    ancestral_noise_removal: bool,
) -> torch.Tensor:
    # DUO sampling grid/noise-removal path:
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/trainer_base.py#L748-L797
    steps = require_sampling_steps(num_sampling_steps)
    eps = float(sampling_eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"DUO sampling eps must be in (0, 1), got {eps}.")

    noise_schedule = str(snapshot_args.get("duo_sampling_noise_schedule", "loglinear"))
    if noise_schedule == "log-linear":
        noise_schedule = "loglinear"
    # DUO ancestral: t_i=eps+i*(1-eps)/S, then map t_i to keep probability.
    times = torch.linspace(eps, 1.0, steps + 1, device=device, dtype=torch.float32)
    scheduled_keep = keep_prob_from_time(
        times,
        noise_schedule,
        float(snapshot_args.get("continuous_time_eps", 1e-3)),
        float(snapshot_args.get("geometric_noise_level_min", DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN)),
        float(snapshot_args.get("geometric_noise_level_max", DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX)),
    )
    if not ancestral_noise_removal:
        return scheduled_keep.clamp_min(torch.finfo(scheduled_keep.dtype).tiny)

    keep_grid = torch.empty(steps + 2, device=device, dtype=torch.float32)
    keep_grid[0] = 1.0
    keep_grid[1:] = scheduled_keep
    return keep_grid.clamp_min(torch.finfo(keep_grid.dtype).tiny)


def _duo_keep_from_sampling_times(
    snapshot_args: dict,
    times: torch.Tensor,  # [S+1] DUO sampler time grid
) -> torch.Tensor:
    noise_schedule = str(snapshot_args.get("duo_sampling_noise_schedule", "loglinear"))
    if noise_schedule == "log-linear":
        noise_schedule = "loglinear"
    return keep_prob_from_time(
        times,
        noise_schedule,
        float(snapshot_args.get("continuous_time_eps", 1e-3)),
        float(snapshot_args.get("geometric_noise_level_min", DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN)),
        float(snapshot_args.get("geometric_noise_level_max", DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX)),
    )


def _duo_sampling_time_from_keep_prob(snapshot_args: dict, keep_prob: float) -> float:
    # Needed only for the official linear-constant-linear-0.9-inv Psi/ReMDM profile.
    noise_schedule = str(snapshot_args.get("duo_sampling_noise_schedule", "loglinear"))
    if noise_schedule == "log-linear":
        noise_schedule = "loglinear"
    eps = float(snapshot_args.get("continuous_time_eps", 1e-3))
    keep_prob = min(max(float(keep_prob), eps), 1.0)
    if noise_schedule == "loglinear":
        return (1.0 - keep_prob) / (1.0 - eps)
    if noise_schedule == "linear":
        return 1.0 - keep_prob
    raise ValueError(f"DUO Psi inverse profile supports linear/loglinear schedules, got {noise_schedule!r}.")


def sample_duo_psi_keep_grid(
    snapshot_args: dict,
    num_sampling_steps: int,
    sampling_eps: float,
    device: str,
    config: dict,
) -> torch.Tensor:
    steps = require_sampling_steps(num_sampling_steps)
    eps = float(sampling_eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"DUO sampling eps must be in (0, 1), got {eps}.")

    profile = str(config["time_profile"])
    if profile == "linear":
        # DUO Psi linear profile: the same S+1 uniform time points as ancestral.
        times = torch.linspace(eps, 1.0, steps + 1, device=device, dtype=torch.float32)
        keep_prob = _duo_keep_from_sampling_times(snapshot_args, times)
        return keep_prob.clamp_min(torch.finfo(keep_prob.dtype).tiny)

    if not profile.startswith("linear-constant-linear"):
        raise ValueError(f"Unsupported DUO Psi time_profile={profile!r}.")
    parts = profile.split("-")
    constant_keep = float(parts[3])
    constant_time = _duo_sampling_time_from_keep_prob(snapshot_args, constant_keep) if "inv" in parts[4:] else constant_keep
    n_hi = round(float(config["high_frac"]) * (steps + 1))
    n_mid = round(float(config["middle_frac"]) * (steps + 1))
    n_hi = min(max(n_hi, 1), steps + 1)
    n_mid = min(max(n_mid, 0), steps + 1 - n_hi)
    n_low = steps + 1 - n_hi - n_mid
    noisy_to_clean = [
        torch.linspace(1.0, constant_time, n_hi, device=device, dtype=torch.float32),
    ]
    if n_mid:
        noisy_to_clean.append(torch.full((n_mid,), constant_time, device=device, dtype=torch.float32))
    if n_low:
        noisy_to_clean.append(torch.linspace(constant_time, eps, n_low, device=device, dtype=torch.float32))
    times = torch.cat(noisy_to_clean)
    keep_prob = _duo_keep_from_sampling_times(snapshot_args, torch.flip(times, dims=[0]))
    return keep_prob.clamp_min(torch.finfo(keep_prob.dtype).tiny)


def duo_psi_posterior_mix_weights(
    keep_grid: torch.Tensor,  # [S+1] clean-to-noisy keep probabilities
    config: dict,
) -> torch.Tensor:
    n = keep_grid.numel()
    noisy_to_clean = torch.flip(keep_grid, dims=[0])
    n_hi = round(float(config["high_frac"]) * n)
    n_mid = round(float(config["middle_frac"]) * n)
    n_hi = min(max(n_hi, 0), n)
    n_mid = min(max(n_mid, 0), n - n_hi)
    modes = (
        (str(config["high_mode"]), noisy_to_clean[:n_hi]),
        (str(config["middle_mode"]), noisy_to_clean[n_hi : n_hi + n_mid]),
        (str(config["low_mode"]), noisy_to_clean[n_hi + n_mid :]),
    )
    chunks = []
    for mode, segment in modes:
        if segment.numel() == 0:
            continue
        chunks.append(_duo_psi_mode_to_mix_weights(mode, segment))
    weights_noisy_to_clean = torch.cat(chunks) if chunks else torch.ones(n, device=keep_grid.device)
    return torch.flip(weights_noisy_to_clean, dims=[0])


def _duo_psi_mode_to_mix_weights(
    mode: str,
    keep_segment: torch.Tensor,  # [segment_steps] noisy-to-clean keep probabilities
) -> torch.Tensor:
    n = keep_segment.numel()
    if mode == "pure-posterior":
        return torch.ones(n, device=keep_segment.device)
    if mode == "pure-pc":
        return torch.zeros(n, device=keep_segment.device)
    eta = float(mode.split("-")[-1])
    if mode.startswith("constant-") and not mode.startswith("constant-remdm"):
        return torch.full((n,), eta, device=keep_segment.device)
    if n < 2:
        return torch.ones(n, device=keep_segment.device)
    keep_t = keep_segment[:-1]
    keep_s = keep_segment[1:]
    eta_t = torch.full_like(keep_t, eta)
    tiny = torch.finfo(keep_segment.dtype).tiny
    if mode.startswith("max-capped-"):
        removal_mass = torch.minimum(eta_t, (1.0 - keep_s) / keep_t.clamp_min(tiny))
        removal_mass = torch.where(keep_t == 0, eta_t, removal_mass)
    elif mode.startswith("max-rescale-"):
        max_removal_mass = torch.minimum(eta_t, (1.0 - keep_s) / keep_t.clamp_min(tiny))
        removal_mass = torch.where(keep_t > 0, max_removal_mass, torch.ones_like(keep_t)) * eta
    elif mode.startswith("constant-remdm"):
        removal_mass = eta_t
    else:
        raise ValueError(f"Unsupported DUO Psi mode={mode!r}.")
    mix_weights = torch.clip(1.0 - removal_mass / (1.0 - keep_s).clamp_min(tiny), 0.0, 1.0)
    return torch.cat([mix_weights, torch.ones(1, device=keep_segment.device, dtype=keep_segment.dtype)])


# =============================================================================
# MASK-STATE ANCESTRAL KERNEL
# =============================================================================


def mask_loglinear_model_time(snapshot_args: dict, mask_prob: float, batch_size: int, device: str) -> torch.Tensor:
    if snapshot_args.get("noise_schedule", "loglinear") != "loglinear":
        raise ValueError("The release sampler expects a loglinear mask-corruption checkpoint.")
    eps = float(snapshot_args.get("continuous_time_eps", 1e-3))
    mask_prob_tensor = torch.tensor(mask_prob, dtype=torch.float32, device=device)
    total_noise = -torch.log1p(-(1.0 - eps) * mask_prob_tensor)
    return total_noise.expand(batch_size)


def mask_ancestral_posterior_mass(
    x0_probs: torch.Tensor,  # [batch, seq, vocab] x0 probabilities
    mask_token_id: int,
    prev_keep: float,
    current_keep: float,
) -> torch.Tensor:
    # Absorbing posterior mass q(x_s | x_t, x0). SUBS uses a continuous mask-
    # probability grid; D3PM-mask uses the same expression on a discrete grid.
    token_mass_scale = max(float(prev_keep) - float(current_keep), 0.0)
    mask_mass_value = max(1.0 - float(prev_keep), 0.0)
    token_mass = x0_probs * max(token_mass_scale, 0.0)
    vocab_size = token_mass.size(-1)
    if mask_token_id == vocab_size:
        mask_mass = torch.full(
            (*token_mass.shape[:-1], 1),
            max(mask_mass_value, 0.0),
            device=x0_probs.device,
            dtype=token_mass.dtype,
        )
        return torch.cat([token_mass, mask_mass], dim=-1)
    if 0 <= mask_token_id < vocab_size:
        mass = token_mass.clone()
        mass[..., mask_token_id] = max(mask_mass_value, 0.0)
        return mass
    raise ValueError(f"mask_token_id={mask_token_id} is incompatible with vocab size {vocab_size}.")


def sample_mask_ancestral_state(
    model,
    current: torch.Tensor,  # [batch, seq]
    posterior_keep_grid: torch.Tensor,  # [S+1] clean-to-noisy keep probabilities
    model_times_for_step,
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    preserve_unmasked_x0: bool,
    cache_x0_probs: bool,
    time_conditioning: bool,
    final_denoise: str,
) -> torch.Tensor:
    if final_denoise not in {"argmax_all", "sample_masked"}:
        raise ValueError(f"Unsupported mask final_denoise={final_denoise!r}.")
    greedy = temperature <= 0.0
    x0_probs_cache = None

    for step in range(posterior_keep_grid.numel() - 1, 0, -1):
        if x0_probs_cache is None:
            x0_probs_cache = predict_x0_probs(
                model,
                current,
                model_times_for_step(step, current.size(0)),
                device=device,
                amp_dtype=amp_dtype,
                temperature=temperature,
                top_k=top_k,
                preserve_unmasked=preserve_unmasked_x0,
            )

        posterior_mass = mask_ancestral_posterior_mass(
            x0_probs_cache,
            model.mask_token_id,
            prev_keep=float(posterior_keep_grid[step - 1].item()),
            current_keep=float(posterior_keep_grid[step].item()),
        )
        proposal = posterior_mass.argmax(dim=-1) if greedy else sample_unnormalized_probs(posterior_mass, generator)
        next_state = torch.where(current.eq(model.mask_token_id), proposal, current)
        if (not cache_x0_probs) or (not torch.equal(next_state, current)) or time_conditioning:
            x0_probs_cache = None
        current = next_state

    final_mask = current.eq(model.mask_token_id)
    if final_denoise == "sample_masked" and not final_mask.any():
        return current
    final_probs = predict_x0_probs(
        model,
        current,
        model_times_for_step(0, current.size(0)),
        device=device,
        amp_dtype=amp_dtype,
        temperature=temperature,
        top_k=top_k,
        preserve_unmasked=preserve_unmasked_x0,
    )
    if final_denoise == "argmax_all":
        return final_probs.argmax(dim=-1)
    fill = sample_from_probs(final_probs, generator, greedy=greedy)
    if final_denoise == "sample_masked":
        return torch.where(final_mask, fill, current)
    raise AssertionError("unreachable mask final_denoise branch")


# =============================================================================
# AR SAMPLER
# =============================================================================


def sample_ar(
    model,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
) -> list[int]:
    tokens = list(prompt_tokens) if prompt_tokens else [getattr(tokenizer, "eot_token", 50256)]
    greedy = temperature <= 0.0
    for _ in range(max_new_tokens):
        # no kv cache here each step reruns the current prefix
        idx = torch.tensor(tokens[-model.config.block_size :], dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            with autocast_context(device, amp_dtype):
                logits, _ = model(idx, targets=None, return_logits=True)
        probs = next_token_probs(logits[:, -1, :], temperature, top_k)
        tokens.append(int(sample_from_probs(probs, generator, greedy=greedy)[0].item()))
    return tokens


# =============================================================================
# MASK-SAMPLER METHOD WRAPPERS
# =============================================================================


def sample_subs_mask_ancestral_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    num_sampling_steps: int,
    sampling_eps: float,
) -> torch.Tensor:
    if not (0.0 < sampling_eps < 1.0):
        raise ValueError(f"sampling_eps must be in (0, 1), got {sampling_eps}.")

    num_steps = require_sampling_steps(num_sampling_steps)
    # SUBS: p_mask_i=sampling_eps+i*(1-sampling_eps)/S, independent of training T.
    mask_prob_grid = torch.linspace(float(sampling_eps), 1.0, num_steps + 1, device=current.device, dtype=torch.float32)
    posterior_keep_grid = 1.0 - mask_prob_grid

    def model_times_for_step(step: int, batch_size: int) -> torch.Tensor:
        return mask_loglinear_model_time(snapshot_args, float(mask_prob_grid[step].item()), batch_size, device)

    # SUBS mask ancestral sampler:
    # https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L612-L637
    return sample_mask_ancestral_state(
        model,
        current,
        posterior_keep_grid=posterior_keep_grid,
        model_times_for_step=model_times_for_step,
        device=device,
        amp_dtype=amp_dtype,
        generator=generator,
        temperature=temperature,
        top_k=top_k,
        preserve_unmasked_x0=True,
        cache_x0_probs=True,
        time_conditioning=bool(snapshot_args.get("time_conditioning", False)),
        final_denoise="argmax_all",
    )


# =============================================================================
# UNIFORM-STATE ANCESTRAL KERNEL
# =============================================================================


def uniform_forward_process_probs(
    x0_probs: torch.Tensor,  # [batch, seq, vocab] x0 probabilities
    keep_prob: float,
) -> torch.Tensor:
    vocab_size = x0_probs.size(-1)
    keep = x0_probs.new_tensor(float(keep_prob))
    return keep * x0_probs + (1.0 - keep) / vocab_size


def uniform_ancestral_posterior_probs(
    x0_probs: torch.Tensor,  # [batch, seq, vocab] x0 probabilities
    current: torch.Tensor,  # [batch, seq]
    prev_keep: float,
    current_keep: float,
) -> torch.Tensor:
    # Uniform-state posterior q(x_s | x_t, x0), used by D3PM, DUO, and DUO Psi:
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/algo.py#L307-L335
    vocab_size = x0_probs.size(-1)
    dtype = x0_probs.dtype
    tiny = torch.finfo(dtype).tiny
    keep_s = x0_probs.new_tensor(float(prev_keep)).clamp_min(tiny)
    keep_t = x0_probs.new_tensor(float(current_keep)).clamp_min(0.0)
    keep_t_given_s = keep_t / keep_s
    keep_delta = keep_s - keep_t
    xt_one_hot = F.one_hot(current, num_classes=vocab_size).to(dtype)
    x0_at_xt = x0_probs.gather(-1, current.unsqueeze(-1))
    numerator = (
        keep_t * vocab_size * x0_probs * xt_one_hot
        + (keep_t_given_s - keep_t) * xt_one_hot
        + keep_delta * x0_probs
        + (1.0 - keep_t_given_s) * (1.0 - keep_s) / vocab_size
    )
    denom = keep_t * vocab_size * x0_at_xt + (1.0 - keep_t)
    posterior = numerator / denom.clamp_min(tiny)
    return posterior.clamp_min(0.0) / posterior.sum(dim=-1, keepdim=True).clamp_min(tiny)


def sample_uniform_ancestral_chain(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    keep_grid: torch.Tensor,  # [S+1] clean-to-noisy keep probabilities
    final_step_argmax: bool = False,
) -> torch.Tensor:
    greedy = temperature <= 0.0

    for step in range(keep_grid.numel() - 1, 0, -1):
        model_times = model_time_from_keep_prob(snapshot_args, keep_grid, step, batch_size=current.size(0), device=device)
        x0_probs = predict_x0_probs(
            model,
            current,
            model_times,
            device=device,
            amp_dtype=amp_dtype,
            temperature=temperature,
            top_k=top_k,
        )

        prev_keep = float(keep_grid[step - 1].item())
        current_keep = float(keep_grid[step].item())
        posterior = uniform_ancestral_posterior_probs(
            x0_probs,
            current,
            prev_keep=prev_keep,
            current_keep=current_keep,
        )
        if greedy or (final_step_argmax and step == 1):
            current = posterior.argmax(dim=-1)
        else:
            current = sample_from_probs(posterior, generator, greedy=False)

    return current


def sample_uniform_psi_chain(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    temperature: float,
    top_k: int,
    keep_grid: torch.Tensor,  # [S+1] clean-to-noisy keep probabilities
    psi_config: dict,
) -> torch.Tensor:
    greedy = temperature <= 0.0
    top_p = float(psi_config.get("top_p", 1.0))
    posterior_mix_weights = duo_psi_posterior_mix_weights(keep_grid, psi_config).to(device=current.device, dtype=torch.float32)

    for step in range(keep_grid.numel() - 1, 0, -1):
        model_times = model_time_from_keep_prob(snapshot_args, keep_grid, step, batch_size=current.size(0), device=device)
        x0_probs = predict_x0_probs(
            model,
            current,
            model_times,
            device=device,
            amp_dtype=amp_dtype,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        prev_keep = float(keep_grid[step - 1].item())
        current_keep = float(keep_grid[step].item())
        q_xs = uniform_ancestral_posterior_probs(
            x0_probs,
            current,
            prev_keep=prev_keep,
            current_keep=current_keep,
        )
        q_x0 = uniform_ancestral_posterior_probs(
            x0_probs,
            current,
            prev_keep=1.0,
            current_keep=current_keep,
        )
        pc_q_xs = uniform_forward_process_probs(q_x0, prev_keep)
        posterior_mix = posterior_mix_weights[step].to(dtype=q_xs.dtype)
        posterior = posterior_mix * q_xs + (1.0 - posterior_mix) * pc_q_xs
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(posterior.dtype).tiny)
        current = posterior.argmax(dim=-1) if greedy else sample_from_probs(posterior, generator, greedy=False)

    return current


# =============================================================================
# D3PM METHOD WRAPPERS
# =============================================================================


def sample_d3pm_uniform_ancestral_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    num_sampling_steps: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    keep_grid = sample_d3pm_keep_prob_grid(snapshot_args, num_sampling_steps, device)
    return sample_uniform_ancestral_chain(
        model,
        snapshot_args,
        current,
        device=device,
        amp_dtype=amp_dtype,
        generator=generator,
        temperature=temperature,
        top_k=top_k,
        keep_grid=keep_grid,
        final_step_argmax=True,
    )


def sample_d3pm_uniform_psi_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    num_sampling_steps: int,
    temperature: float,
    top_k: int,
    sampler: str,
) -> torch.Tensor:
    keep_grid = sample_d3pm_keep_prob_grid(snapshot_args, num_sampling_steps, device)
    return sample_uniform_psi_chain(
        model,
        snapshot_args,
        current,
        device=device,
        amp_dtype=amp_dtype,
        generator=generator,
        temperature=temperature,
        top_k=top_k,
        keep_grid=keep_grid,
        psi_config=DUO_PSI_CONFIGS[sampler],
    )


def sample_d3pm_mask_ancestral_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    num_sampling_steps: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    keep_grid = sample_d3pm_keep_prob_grid(snapshot_args, num_sampling_steps, device)

    def model_times_for_step(step: int, batch_size: int) -> torch.Tensor:
        return model_time_from_keep_prob(snapshot_args, keep_grid, step, batch_size=batch_size, device=device)

    # Absorbing-state D3PM ancestral posterior specialization:
    # https://github.com/google-research/google-research/blob/1fa17414f56c3703d5adb3818338b6e35e0fd550/d3pm/text/diffusion.py#L303-L428
    return sample_mask_ancestral_state(
        model,
        current,
        posterior_keep_grid=keep_grid,
        model_times_for_step=model_times_for_step,
        device=device,
        amp_dtype=amp_dtype,
        generator=generator,
        temperature=temperature,
        top_k=top_k,
        preserve_unmasked_x0=False,
        cache_x0_probs=False,
        time_conditioning=bool(snapshot_args.get("time_conditioning", False)),
        final_denoise="sample_masked",
    )


# =============================================================================
# SEDD REVERSE-RATE SAMPLERS
# =============================================================================

# SEDD reverse rates, staggered scores, and denoiser:
# https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L77-L189
# https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L192-L266
# https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L59-L105


def sedd_log_score(
    model,
    current: torch.Tensor,  # [batch, seq]
    total_noise: torch.Tensor,  # [batch]
    objective: str,
    device: str,
    amp_dtype: str,
) -> torch.Tensor:
    with torch.inference_mode():
        with autocast_context(device, amp_dtype):
            raw_logits = model(current, total_noise.expand(current.size(0)))
    if objective == "sedd_uniform":
        log_score = raw_logits.float()
        log_score.scatter_(-1, current.unsqueeze(-1), torch.zeros_like(log_score[..., :1]))
        return log_score
    if objective == "sedd_mask":
        vocab_size = raw_logits.size(-1)
        expm1_noise = torch.expm1(total_noise)
        esigm1_log = torch.log(expm1_noise.clamp_min(torch.finfo(expm1_noise.dtype).tiny)).view(-1, 1, 1)
        log_score = raw_logits.float() - esigm1_log - math.log(vocab_size)
        mask_col = torch.zeros(*log_score.shape[:-1], 1, device=log_score.device, dtype=log_score.dtype)
        log_score = torch.cat([log_score, mask_col], dim=-1)
        log_score.scatter_(-1, current.unsqueeze(-1), torch.zeros_like(log_score[..., :1]))
        return log_score
    raise NotImplementedError(f"Unsupported SEDD objective: {objective!r}")


def sample_sedd_euler_step(
    current: torch.Tensor,  # [batch, seq]
    score: torch.Tensor,  # [batch, seq, vocab]
    noise_type: str,
    noise_step: torch.Tensor,  # [batch]
    generator: torch.Generator,
) -> torch.Tensor:
    probs = sedd_euler_probs(current, score, noise_type, noise_step)
    return sample_from_probs(probs, generator, greedy=False)


def sedd_euler_probs(
    current: torch.Tensor,  # [batch, seq]
    score: torch.Tensor,  # [batch, seq, vocab]
    noise_type: str,
    noise_step: torch.Tensor,  # [batch]
) -> torch.Tensor:
    dim = score.size(-1)
    one_hot = F.one_hot(current, num_classes=dim).to(score.dtype)
    if noise_type == "uniform":
        rate = torch.ones_like(score) / dim
        rate.scatter_(-1, current.unsqueeze(-1), -torch.full_like(score[..., :1], (dim - 1) / dim))
    elif noise_type == "mask":
        mask_id = dim - 1
        rate = -one_hot
        mask_rows = current.eq(mask_id)
        if mask_rows.any():
            rate[mask_rows] += 1
    else:
        raise NotImplementedError(f"Unsupported SEDD noise_type: {noise_type!r}")
    reverse_rate = rate * score
    reverse_rate.scatter_(-1, current.unsqueeze(-1), torch.zeros_like(reverse_rate[..., :1]))
    reverse_rate.scatter_(-1, current.unsqueeze(-1), -reverse_rate.sum(dim=-1, keepdim=True))
    probs = one_hot + noise_step.view(-1, 1, 1).to(score.dtype) * reverse_rate
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)


def sample_sedd_analytic_step(
    current: torch.Tensor,  # [batch, seq]
    score: torch.Tensor,  # [batch, seq, vocab]
    noise_type: str,
    noise_step: torch.Tensor,  # [batch]
    generator: torch.Generator,
    denoise: bool = False,
) -> torch.Tensor:
    probs = sedd_analytic_probs(current, score, noise_type, noise_step, denoise=denoise)
    return sample_from_probs(probs, generator, greedy=False)


def sample_sedd_denoise_step(
    current: torch.Tensor,  # [batch, seq]
    score: torch.Tensor,  # [batch, seq, vocab]
    noise_type: str,
    total_noise: torch.Tensor,  # [batch]
    generator: torch.Generator,
) -> torch.Tensor:
    return sample_sedd_analytic_step(
        current,
        score,
        noise_type,
        total_noise,
        generator,
        denoise=(noise_type == "mask"),
    )


def sedd_analytic_probs(
    current: torch.Tensor,  # [batch, seq]
    score: torch.Tensor,  # [batch, seq, vocab]
    noise_type: str,
    noise_step: torch.Tensor,  # [batch]
    denoise: bool = False,
) -> torch.Tensor:
    dim = score.size(-1)
    noise_step_view = noise_step.view(-1, 1, 1).to(score.dtype)
    if noise_type == "uniform":
        epow = torch.exp(-noise_step_view)
        staggered = ((epow - 1.0) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow
        transition = torch.full_like(score, 0.0)
        off_diag = (1.0 - epow) / dim
        transition += off_diag
        transition.scatter_(
            -1,
            current.unsqueeze(-1),
            (1.0 - off_diag.squeeze(-1).squeeze(-1) * (dim - 1)).view(-1, 1, 1).expand_as(current.unsqueeze(-1)).to(score.dtype),
        )
        probs = staggered * transition
    elif noise_type == "mask":
        exp_noise_step = torch.exp(noise_step_view)
        staggered = score.clone()
        extra_const = (1.0 - exp_noise_step.squeeze(-1)) * score.sum(dim=-1)
        staggered = staggered * exp_noise_step
        staggered[..., -1] += extra_const
        exp_neg = torch.exp(-noise_step_view)
        transition = exp_neg * F.one_hot(current, num_classes=dim).to(score.dtype)
        transition = transition + torch.where(
            current.eq(dim - 1).unsqueeze(-1),
            # Keep [batch, 1, 1] so batched sampling broadcasts over sequence/vocab axes.
            1.0 - exp_neg,
            torch.zeros_like(score),
        )
        probs = staggered * transition
        if denoise:
            probs = probs[..., :-1]
    else:
        raise NotImplementedError(f"Unsupported SEDD noise_type: {noise_type!r}")
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)


def sample_sedd_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    objective: str,
    noise_type: str,
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    num_sampling_steps: int,
    sampling_eps: float,
    predictor: str,
    denoise: bool = True,
) -> torch.Tensor:
    total_noise_grid, total_noise_rate_grid = sample_sedd_noise_grids(snapshot_args, num_sampling_steps, sampling_eps, device)
    dt = (1.0 - float(sampling_eps)) / (total_noise_grid.numel() - 1)  # official get_pc_sampler step_size
    for step in range(total_noise_grid.numel() - 1, 0, -1):
        current_noise = total_noise_grid[step].view(1)
        prev_noise = total_noise_grid[step - 1].view(1)
        score = sedd_log_score(
            model,
            current,
            current_noise.expand(current.size(0)),
            objective=objective,
            device=device,
            amp_dtype=amp_dtype,
        ).exp()
        if predictor == "sedd_euler":
            # Official EulerPredictor: reverse update uses dt * rate_noise(t) (sampling.py L61-67).
            noise_step = (dt * total_noise_rate_grid[step]).clamp_min(0.0).view(1).expand(current.size(0))
            current = sample_sedd_euler_step(current, score, noise_type, noise_step, generator)
        elif predictor == "sedd_analytic":
            # Official AnalyticPredictor: reverse update uses the total-noise drop (sampling.py L77-86).
            noise_step = (current_noise - prev_noise).clamp_min(0.0).expand(current.size(0))
            current = sample_sedd_analytic_step(current, score, noise_type, noise_step, generator)
        else:
            raise NotImplementedError(f"Unsupported SEDD predictor: {predictor!r}")

    if denoise:
        final_noise = total_noise_grid[0].view(1).expand(current.size(0))
        score = sedd_log_score(
            model,
            current,
            final_noise,
            objective=objective,
            device=device,
            amp_dtype=amp_dtype,
        ).exp()
        current = sample_sedd_denoise_step(
            current,
            score,
            noise_type,
            final_noise,
            generator,
        )
    return current


# =============================================================================
# DUO UNIFORM-STATE SAMPLERS
# =============================================================================

# DUO official sampling reuses the same uniform-state ancestral posterior as
# D3PM-uniform, with DUO's keep grid and final noise-removal convention.
# https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/algo.py#L307-L335
# https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/trainer_base.py#L633-L672
# https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/trainer_base.py#L748-L797


def sample_duo_uniform_state(
    model,
    snapshot_args: dict,
    current: torch.Tensor,  # [batch, seq]
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    num_sampling_steps: int,
    sampling_eps: float,
    noise_removal: str = "ancestral",
    sampler: str = "duo_ancestral",
) -> torch.Tensor:
    if noise_removal not in {"ancestral", "greedy"}:
        raise ValueError(f"Unsupported DUO noise_removal={noise_removal!r}.")
    if sampler in UNIFORM_PSI_SAMPLERS:
        keep_grid = sample_duo_psi_keep_grid(
            snapshot_args,
            num_sampling_steps,
            sampling_eps,
            device,
            config=DUO_PSI_CONFIGS[sampler],
        )
        current = sample_uniform_psi_chain(
            model,
            snapshot_args,
            current,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            temperature=1.0,
            top_k=0,
            keep_grid=keep_grid,
            psi_config=DUO_PSI_CONFIGS[sampler],
        )
        if noise_removal == "ancestral":
            final_grid = torch.stack([torch.ones_like(keep_grid[0]), keep_grid[0]]).to(device=current.device)
            current = sample_uniform_ancestral_chain(
                model,
                snapshot_args,
                current,
                device=device,
                amp_dtype=amp_dtype,
                generator=generator,
                temperature=1.0,
                top_k=0,
                keep_grid=final_grid,
            )
    else:
        keep_grid = sample_duo_official_keep_grid(
            snapshot_args,
            num_sampling_steps,
            sampling_eps,
            device,
            ancestral_noise_removal=(noise_removal == "ancestral"),
        )
        current = sample_uniform_ancestral_chain(
            model,
            snapshot_args,
            current,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            temperature=1.0,
            top_k=0,
            keep_grid=keep_grid,
        )
    if noise_removal == "greedy":
        model_times = model_time_from_keep_prob(snapshot_args, keep_grid, 0, batch_size=current.size(0), device=device)
        current = predict_x0_probs(
            model,
            current,
            model_times,
            device=device,
            amp_dtype=amp_dtype,
            temperature=1.0,
            top_k=0,
        ).argmax(dim=-1)
    return current


def sample_dlm(
    model,
    snapshot_args: dict,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    amp_dtype: str,
    generator: torch.Generator,
    dlm_sampler: str,
    num_sampling_steps: int,
    sampling_eps: float,
    batch_size: int = 1,
) -> list[list[int]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    objective, noise_type = checkpoint_objective_and_noise(snapshot_args)
    dlm_sampler = resolve_dlm_sampler(objective, dlm_sampler)
    if objective == "subs_mask":
        if noise_type != "mask":
            raise NotImplementedError(f"subs_mask checkpoints require mask noise, got noise_type={noise_type!r}.")
        current = torch.full((batch_size, max_new_tokens), model.mask_token_id, dtype=torch.long, device=device)
        if dlm_sampler != "subs_mask_ancestral":
            raise NotImplementedError(f"Unsupported SUBS mask sampler: {dlm_sampler!r}")
        current = sample_subs_mask_ancestral_state(
            model,
            snapshot_args,
            current,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
            num_sampling_steps=num_sampling_steps,
            sampling_eps=sampling_eps,
        )
    elif objective in {"d3pm_mask", "d3pm_uniform"}:
        expected_sampler = f"{objective}_ancestral"
        if dlm_sampler != expected_sampler and not (objective == "d3pm_uniform" and dlm_sampler in UNIFORM_PSI_SAMPLERS):
            raise NotImplementedError(
                f"objective={objective} requires sampler={expected_sampler}"
                f"{' or a DUO Psi uniform sampler' if objective == 'd3pm_uniform' else ''}, got {dlm_sampler!r}."
            )
        if noise_type == "uniform":
            current = torch.randint(model.config.vocab_size, (batch_size, max_new_tokens), device=device, generator=generator)
            if dlm_sampler in UNIFORM_PSI_SAMPLERS:
                current = sample_d3pm_uniform_psi_state(
                    model,
                    snapshot_args,
                    current,
                    device=device,
                    amp_dtype=amp_dtype,
                    generator=generator,
                    num_sampling_steps=num_sampling_steps,
                    temperature=temperature,
                    top_k=top_k,
                    sampler=dlm_sampler,
                )
            else:
                current = sample_d3pm_uniform_ancestral_state(
                    model,
                    snapshot_args,
                    current,
                    device=device,
                    amp_dtype=amp_dtype,
                    generator=generator,
                    num_sampling_steps=num_sampling_steps,
                    temperature=temperature,
                    top_k=top_k,
                )
        elif noise_type == "mask":
            current = torch.full((batch_size, max_new_tokens), model.mask_token_id, dtype=torch.long, device=device)
            current = sample_d3pm_mask_ancestral_state(
                model,
                snapshot_args,
                current,
                device=device,
                amp_dtype=amp_dtype,
                generator=generator,
                num_sampling_steps=num_sampling_steps,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            raise NotImplementedError(f"Unsupported D3PM noise_type: {noise_type!r}")
    elif objective == "sedd_mask":
        if dlm_sampler not in {"sedd_euler", "sedd_analytic"}:
            raise NotImplementedError(f"Unsupported SEDD mask sampler: {dlm_sampler!r}")
        current = torch.full((batch_size, max_new_tokens), model.mask_token_id, dtype=torch.long, device=device)
        current = sample_sedd_state(
            model,
            snapshot_args,
            current,
            objective=objective,
            noise_type=noise_type,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            num_sampling_steps=num_sampling_steps,
            sampling_eps=sampling_eps,
            predictor=dlm_sampler,
            denoise=True,
        )
    elif objective == "sedd_uniform":
        if dlm_sampler not in {"sedd_euler", "sedd_analytic"}:
            raise NotImplementedError(f"Unsupported SEDD uniform sampler: {dlm_sampler!r}")
        current = torch.randint(model.config.vocab_size, (batch_size, max_new_tokens), device=device, generator=generator)
        current = sample_sedd_state(
            model,
            snapshot_args,
            current,
            objective=objective,
            noise_type=noise_type,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            num_sampling_steps=num_sampling_steps,
            sampling_eps=sampling_eps,
            predictor=dlm_sampler,
            denoise=True,
        )
    elif objective == "duo_uniform":
        if dlm_sampler not in {"duo_ancestral", "duo_greedy_tail", *UNIFORM_PSI_SAMPLERS}:
            raise NotImplementedError(f"Unsupported DUO sampler: {dlm_sampler!r}")
        current = torch.randint(model.config.vocab_size, (batch_size, max_new_tokens), device=device, generator=generator)
        current = sample_duo_uniform_state(
            model,
            snapshot_args,
            current,
            device=device,
            amp_dtype=amp_dtype,
            generator=generator,
            num_sampling_steps=num_sampling_steps,
            sampling_eps=sampling_eps,
            noise_removal="greedy" if dlm_sampler == "duo_greedy_tail" else "ancestral",
            sampler=dlm_sampler,
        )
    else:
        raise NotImplementedError(f"Unsupported DLM path: noise_type={noise_type!r}, objective={objective!r}")
    return current.tolist()


# =============================================================================
# IO AND CLI
# =============================================================================


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Samples", ""]
    for row in rows:
        lines.extend(
            [
                f"## sample {row['sample_index']}",
                "",
                row["target_text"].strip(),
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample unconditional text from an AR or release DLM checkpoint.")
    parser.add_argument("snapshot_path")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_variant", choices=("auto", "model", "ema_model"), default="model")
    parser.add_argument("--amp_dtype", choices=("auto", "none", "bfloat16"), default="auto")
    parser.add_argument(
        "--dlm_sampler",
        choices=DLM_SAMPLER_CHOICES,
        default="auto",
    )
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--sampling_eps", type=float, default=1e-5)
    parser.add_argument(
        "--sampling_batch_size",
        type=int,
        default=1,
        help="Number of unconditional DLM samples to generate per model pass batch. AR sampling remains unbatched.",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    device = args.device
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    timing("resolve_amp_dtype", time.perf_counter() - setup_start, device=device, amp_dtype=amp_dtype)
    load_start = time.perf_counter()
    snapshot = load_snapshot(args.snapshot_path, device=device, checkpoint_variant=args.checkpoint_variant)
    timing("load_snapshot", time.perf_counter() - load_start, trainer=snapshot.trainer)
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    reported_dlm_sampler = ""
    if snapshot.trainer == "dlm":
        objective, _ = checkpoint_objective_and_noise(snapshot.args)
        reported_dlm_sampler = resolve_dlm_sampler(objective, args.dlm_sampler)

    rows = []
    generation_start = time.perf_counter()
    sample_index = 0
    while sample_index < args.samples_per_prompt:
        chunk_start = time.perf_counter()
        if snapshot.trainer == "ar":
            chunk_size = 1
            local_seed = args.seed + sample_index
            generator.manual_seed(local_seed)
            tokens = sample_ar(
                snapshot.model,
                [getattr(tokenizer, "eot_token", 50256)],
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                device,
                amp_dtype,
                generator,
            )[1:]
            token_batches = [tokens]
        elif snapshot.trainer == "dlm":
            chunk_size = min(max(1, args.sampling_batch_size), args.samples_per_prompt - sample_index)
            generator.manual_seed(args.seed + sample_index)
            token_batches = sample_dlm(
                snapshot.model,
                snapshot.args,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                device,
                amp_dtype,
                generator,
                args.dlm_sampler,
                args.num_sampling_steps,
                args.sampling_eps,
                batch_size=chunk_size,
            )
        else:
            raise NotImplementedError(f"Unsupported trainer for sampling: {snapshot.trainer!r}")

        for offset, tokens in enumerate(token_batches):
            current_index = sample_index + offset
            local_seed = args.seed + current_index
            text = tokenizer.decode(tokens)
            rows.append(
                {
                    "prompt_index": 0,
                    "sample_index": current_index,
                    "seed": local_seed,
                    "trainer": snapshot.trainer,
                    "protocol": "unconditional",
                    "sample_mode": "unconditional",
                    "checkpoint": checkpoint_label(args.snapshot_path),
                    "checkpoint_variant": snapshot.checkpoint_variant,
                    "prompt": "",
                    "generated_text": text,
                    "target_text": text,
                    "full_text": text,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "dlm_sampler": reported_dlm_sampler,
                    "num_sampling_steps": args.num_sampling_steps if snapshot.trainer == "dlm" else 0,
                    "sampling_eps": args.sampling_eps if snapshot.trainer == "dlm" else 0,
                    "sampling_batch_size": chunk_size if snapshot.trainer == "dlm" else 1,
                }
            )
        sample_index += chunk_size
        elapsed = time.perf_counter() - generation_start
        timing(
            "sample_progress",
            elapsed,
            samples=sample_index,
            total=args.samples_per_prompt,
            chunk_size=chunk_size,
            last_chunk_seconds=f"{time.perf_counter() - chunk_start:.3f}",
            avg_sample_seconds=f"{elapsed / sample_index:.3f}",
        )
    timing("sample_generation_total", time.perf_counter() - generation_start, samples=args.samples_per_prompt)

    payload = {
        "snapshot": checkpoint_label(args.snapshot_path),
        "trainer": snapshot.trainer,
        "checkpoint_variant": snapshot.checkpoint_variant,
        "sample_mode": "unconditional",
        "samples": rows,
    }
    if args.output_json:
        write_start = time.perf_counter()
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {output_json}")
        timing("write_json", time.perf_counter() - write_start)
    else:
        print(json.dumps(payload, indent=2))
    if args.output_md:
        write_start = time.perf_counter()
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, output_md)
        print(f"Wrote {output_md}")
        timing("write_markdown", time.perf_counter() - write_start)
    timing("total", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
