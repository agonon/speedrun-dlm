from contextlib import nullcontext
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F

from .train_ar import GPT, get_model_config as get_ar_model_config
from .train_dlm import (
    DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
    DiffusionTransformer,
    get_model_config as get_dlm_model_config,
    keep_prob_from_time,
    total_noise_rate_from_time,
)


tokenizer = tiktoken.get_encoding("gpt2")


@dataclass
class LoadedSnapshot:
    trainer: str
    model: torch.nn.Module
    args: dict
    checkpoint_variant: str


def resolve_amp_dtype(device: str, amp_dtype: str) -> str:
    if amp_dtype == "auto":
        return "bfloat16" if device.startswith("cuda") else "none"
    return amp_dtype


def autocast_context(device: str, amp_dtype: str):
    if device.startswith("cuda") and amp_dtype == "bfloat16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_snapshot(snapshot_path: str, device: str, checkpoint_variant: str = "auto") -> LoadedSnapshot:
    checkpoint = torch.load(snapshot_path, map_location="cpu")
    trainer = checkpoint.get("trainer")
    args = checkpoint["args"]
    if trainer == "ar":
        model = GPT(get_ar_model_config(args["model"]))
    elif trainer == "dlm":
        model = DiffusionTransformer(
            get_dlm_model_config(
                args["model"],
                cond_dim=int(args.get("cond_dim", 128)),
                dropout=float(args.get("dropout", 0.1)),
                time_conditioning=bool(args.get("time_conditioning", False)),
            )
        )
    else:
        raise ValueError(f"Unsupported trainer in snapshot: {trainer}")

    if checkpoint_variant == "auto":
        state_dict_key = "ema_model" if checkpoint.get("ema_model") is not None else "model"
    elif checkpoint_variant in {"model", "ema_model"}:
        if checkpoint.get(checkpoint_variant) is None:
            raise ValueError(f"Checkpoint variant {checkpoint_variant!r} not found in snapshot: {snapshot_path}")
        state_dict_key = checkpoint_variant
    else:
        raise ValueError(f"Unsupported checkpoint variant: {checkpoint_variant}")

    state_dict = checkpoint[state_dict_key]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return LoadedSnapshot(trainer=trainer, model=model, args=args, checkpoint_variant=state_dict_key)


def make_d3pm_sampling_keep_prob_grid(
    num_sampling_steps: int,
    noise_schedule: str,
    eps: float,
    device: str,
    geometric_noise_level_min: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    geometric_noise_level_max: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
    terminal_zero: bool = True,
) -> torch.Tensor:
    # D3PM sampling: t_i=i/S, mapped through the training noise schedule.
    # The endpoints are forced to fully clean and, when requested, fully noisy.
    times = torch.linspace(0.0, 1.0, num_sampling_steps + 1, device=device, dtype=torch.float32)
    keep_grid = keep_prob_from_time(
        times,
        noise_schedule,
        eps,
        geometric_noise_level_min,
        geometric_noise_level_max,
    )
    keep_grid[0] = 1.0
    if terminal_zero:
        keep_grid[-1] = 0.0
    return keep_grid


def model_time_from_keep_prob(
    args: dict,
    keep_grid: torch.Tensor,
    step: int,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if args.get("objective") in {"d3pm_mask", "d3pm_uniform"}:
        # D3PM training conditions on the discrete time coordinate, not on
        # total_noise. Keep this tied to the sampled grid when using fewer
        # reverse steps than the training grid.
        value = float(step) / max(float(keep_grid.numel() - 1), 1.0)
        return torch.full((batch_size,), value, dtype=torch.float32, device=device)
    eps = float(args.get("continuous_time_eps", 1e-3))
    value = float((-torch.log(keep_grid[step].clamp_min(eps))).item())
    return torch.full((batch_size,), value, dtype=torch.float32, device=device)


def masked_x0_logprobs(
    raw_logits: torch.Tensor,
    noisy_tokens: torch.Tensor,
    mask_token_id: int,
    preserve_unmasked: bool,
) -> torch.Tensor:
    logprobs = F.log_softmax(raw_logits.float(), dim=-1)
    if not preserve_unmasked:
        return logprobs

    unmasked = noisy_tokens.ne(mask_token_id)
    if not unmasked.any():
        return logprobs

    forced_tokens = noisy_tokens.masked_fill(~unmasked, 0)
    forced_token_values = logprobs.gather(-1, forced_tokens.unsqueeze(-1)).squeeze(-1)
    forced_token_values = torch.where(unmasked, torch.zeros_like(forced_token_values), forced_token_values)
    logprobs = logprobs.masked_fill(unmasked.unsqueeze(-1), float("-inf"))
    logprobs.scatter_(-1, forced_tokens.unsqueeze(-1), forced_token_values.unsqueeze(-1))
    return logprobs
