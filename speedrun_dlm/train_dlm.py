import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import tempfile
import time
import uuid
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

with open(__file__) as f:
    code = f.read()


DATA_HEADER_INTS = 256
DATA_FILE_MAGIC = 20240520
DATA_FILE_VERSION = 1
DUO_KEEP_PROB_EPS = 1e-6  # DUO keep probs are clamped away from zero
DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN = 1e-4
DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX = 20.0
DUO_CURRICULUM_MODES = ("dense_softmax", "topk_softmax")
DUO_KEEP_PROB_TABLE_COMMIT = "492505208b361fa330f4703b705abc54cf7ead20"
DUO_KEEP_PROB_TABLE_SHA256 = "19e3ff05d86f033a99df864087c40b5e5c55fd9dc5f1a863e712b0f376b85b30"
DUO_KEEP_PROB_TABLE_URL = (
    "https://raw.githubusercontent.com/s-sahoo/duo/"
    f"{DUO_KEEP_PROB_TABLE_COMMIT}/integral/gpt2.pkl"
)


@dataclass(frozen=True)
class ObjectiveSpec:
    noise_type: str
    default_noise_schedule: str
    default_time_conditioning: bool
    loss_label: str
    d3pm_default_coeffs: tuple[float, float] | None = None  # (vb, ce) to weight the two D3PM loss terms as vb * L_vb + ce * L_ce


DLM_OBJECTIVES = {
    "subs_mask": ObjectiveSpec("mask", "loglinear", False, "continuous-time mask x0 weighted cross entropy"),
    # D3PM-mask is CE-dominant here. Official text D3PM used (vb=1, ce=0.01).
    "d3pm_mask": ObjectiveSpec("mask", "loglinear", True, "discrete-time mask posterior matching", (0.001, 1.0)),
    "d3pm_uniform": ObjectiveSpec("uniform", "linear", True, "discrete-time uniform posterior matching", (1.0, 0.0)),
    "sedd_mask": ObjectiveSpec("mask", "loglinear", True, "continuous-time mask score entropy"),
    "sedd_uniform": ObjectiveSpec("uniform", "geometric", True, "continuous-time uniform score entropy"),
    "duo_uniform": ObjectiveSpec("uniform", "linear", True, "DUO uniform diffusion loss"),
}


def is_d3pm_objective(objective: str) -> bool:
    return DLM_OBJECTIVES[objective].d3pm_default_coeffs is not None


def check_duo_curriculum_mode(mode: str) -> str:
    if mode not in DUO_CURRICULUM_MODES:
        choices = ", ".join(DUO_CURRICULUM_MODES)
        raise ValueError(f"Unsupported DUO curriculum mode: {mode!r}. Choose one of: {choices}.")
    return mode


# =============================================================================
# MODEL
# =============================================================================


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x0.float(), (self.dim,))
        return (x * self.weight[None, None, :]).type_as(x0)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half, 1)
    freqs = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(timesteps, self.frequency_embedding_size)
        return self.mlp(t_freq)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even per-head dimension.")
    seq_len = x.size(-2)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    angles = torch.cat((freqs, freqs), dim=-1)
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class DDiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDiTFinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_final = LayerNorm(config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


@dataclass
class DLMConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    cond_dim: int = 128
    dropout: float = 0.1
    time_conditioning: bool = False


class DiffusionTransformer(nn.Module):
    # shared non-causal DDiT backbone for every DLM objective
    def __init__(self, config: DLMConfig):
        super().__init__()
        self.config = config
        self.mask_token_id = config.vocab_size
        self.transformer = nn.ModuleDict(
            dict(
                # One extra input-only row encodes the absorbing mask token. The
                # output head remains data-vocab-sized and never predicts mask.
                # Uniform objectives do not index the extra row. Keeping it here
                # preserves a single shared checkpoint shape across objectives.
                wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
                h=nn.ModuleList([DDiTBlock(config) for _ in range(config.n_layer)]),
            )
        )
        self.time_map = TimestepEmbedder(config.cond_dim)
        self.output_layer = DDiTFinalLayer(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def forward(
        self,
        idx: torch.Tensor,
        timesteps: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len = idx.shape[:2]
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        )
        if weights is not None:
            if idx.ndim != 3 or weights.shape != idx.shape:
                raise ValueError("Weighted embedding inputs require idx and weights with shape (batch, seq, k).")
            flat_idx = idx.reshape(-1, idx.shape[-1])
            flat_weights = weights.reshape(-1, weights.shape[-1]).float()
            tok_emb = F.embedding_bag(
                flat_idx,
                self.transformer.wte.weight.float(),
                per_sample_weights=flat_weights,
                mode="sum",
            ).view(bsz, seq_len, -1)
        elif idx.ndim == 3:
            if idx.shape[-1] > self.transformer.wte.weight.shape[0]:
                raise ValueError(
                    f"Dense distribution input has vocab dimension {idx.shape[-1]}, "
                    f"but embedding table has {self.transformer.wte.weight.shape[0]} rows."
                )
            token_weight = self.transformer.wte.weight[: idx.shape[-1]].float()
            tok_emb = torch.einsum("blv,ve->ble", F.softmax(idx, dim=-1).float(), token_weight).to(idx.dtype)
        else:
            tok_emb = self.transformer.wte(idx)
        x = tok_emb
        model_times = timesteps.float() if self.config.time_conditioning else torch.zeros_like(timesteps.float())
        cond = F.silu(self.time_map(model_times)).type_as(tok_emb)
        for block in self.transformer.h:
            x = block(x, cond)
        return self.output_layer(x, cond)

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps=1e-8):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)


def model_forward_noisy(model, noisy_tokens: torch.Tensor, time_batch: "TimeBatch", objective: str) -> torch.Tensor:
    if objective == "duo_uniform":
        if time_batch.duo_sparse_input_weights is None:
            if noisy_tokens.ndim == 3:
                return model(noisy_tokens, time_batch.model_times)
            raise RuntimeError("DUO forward requires sample_duo_forward_relaxation to populate DUO noisy inputs.")
        return model(noisy_tokens, time_batch.model_times, weights=time_batch.duo_sparse_input_weights)
    return model(noisy_tokens, time_batch.model_times)


# =============================================================================
# DATA
# =============================================================================


def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(DATA_HEADER_INTS * 4), dtype=np.int32)
    if header[0] != DATA_FILE_MAGIC:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --train_files?")
        raise SystemExit(1)
    assert header[1] == DATA_FILE_VERSION, "unsupported version"
    return int(header[2])


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(DATA_HEADER_INTS * 4), dtype=np.int32)
        assert header[0] == DATA_FILE_MAGIC, "magic number mismatch in the data .bin file"
        assert header[1] == DATA_FILE_VERSION, "unsupported version"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header"
    return tokens


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_len: int,
        process_rank: int,
        num_processes: int,
        name: str,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.files = sorted(glob.glob(filename_pattern))
        assert self.files, f"did not find any files matching {filename_pattern}"
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * batch_size * seq_len
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"{name} data: {ntok_total:,} tokens across {len(self.files)} files")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.batch_size * self.seq_len
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size * self.seq_len
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> torch.Tensor:
        batch_size = self.batch_size
        seq_len = self.seq_len
        buf = self.tokens[self.current_position : self.current_position + batch_size * seq_len]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf.view(batch_size, seq_len)
        self.current_position += batch_size * seq_len * self.num_processes
        if self.current_position + (batch_size * seq_len * self.num_processes) > len(self.tokens):
            self.advance()
        return x.cuda()


# =============================================================================
# TIME GRIDS AND NOISE SCHEDULES
# =============================================================================


@dataclass
class TimeBatch:
    model_times: torch.Tensor  # [batch] scalar sent to the time-conditioning MLP
    keep_prob: torch.Tensor  # [batch] probability the original token survived to sampled time
    loss_weight: torch.Tensor | None = None  # [batch] objective-specific rate or CE multiplier
    prev_keep_prob: torch.Tensor | None = None  # [batch] D3PM keep_prob at step-1, cached instead of passing step+grid into the loss
    duo_log_noise_ratio: torch.Tensor | None = None  # [batch] DUO continuous relaxation time coordinate
    duo_discrete_noisy_tokens: torch.Tensor | None = None  # [batch, seq] DUO argmax tokens used by the uniform loss
    duo_sparse_input_weights: torch.Tensor | None = None  # [batch, seq, top_k] weights for DUO topk-softmax input


def keep_prob_from_time(
    times: torch.Tensor,  # any shape, repo uses [batch] for sampled continuous training times, [T+1] D3PM grids, [S+1] sampler grids
    noise_schedule: str = "loglinear",
    eps: float = 1e-6,
    geometric_noise_level_min: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    geometric_noise_level_max: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
) -> torch.Tensor:
    # returns keep_prob[...] = chance the original token is still unnoised at times[...]
    if noise_schedule == "linear":
        keep_prob = 1.0 - times
    elif noise_schedule == "loglinear":
        keep_prob = 1.0 - (1.0 - eps) * times
    elif noise_schedule == "cosine":
        offset = 0.008
        base = math.cos(offset / (1.0 + offset) * math.pi / 2) ** 2
        keep_prob = torch.cos((times + offset) / (1.0 + offset) * torch.pi / 2).pow(2) / base
    elif noise_schedule == "geometric":
        log_total_noise_min = math.log(geometric_noise_level_min)
        log_total_noise_max = math.log(geometric_noise_level_max)
        total_noise = torch.exp((1.0 - times) * log_total_noise_min + times * log_total_noise_max)
        return torch.exp(-total_noise)
    else:
        raise ValueError(f"Unsupported noise_schedule={noise_schedule!r}")
    return keep_prob.clamp_min(eps)


def total_noise_rate_from_time(
    times: torch.Tensor,  # any shape, repo uses [batch] sampled times and [S+1] SEDD sampler grids
    noise_schedule: str,
    eps: float = 1e-6,
    geometric_noise_level_min: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    geometric_noise_level_max: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
) -> torch.Tensor:
    # returns total_noise_rate[...] = d[-log keep_prob(t)]/dt at times[...]
    if noise_schedule == "linear":
        return 1.0 / (1.0 - times).clamp_min(eps)
    if noise_schedule == "loglinear":
        return (1.0 - eps) / (1.0 - (1.0 - eps) * times)
    if noise_schedule == "cosine":
        offset = 0.008
        base = math.cos(offset / (1.0 + offset) * math.pi / 2) ** 2
        angle = (times + offset) / (1.0 + offset) * math.pi
        keep_prime = -(torch.sin(angle) * math.pi / (2.0 * (1.0 + offset) * base))
        keep_prob = keep_prob_from_time(times, noise_schedule, eps)
        return -keep_prime / keep_prob.clamp_min(eps)
    if noise_schedule == "geometric":
        log_total_noise_min = math.log(geometric_noise_level_min)
        log_total_noise_max = math.log(geometric_noise_level_max)
        total_noise = torch.exp((1.0 - times) * log_total_noise_min + times * log_total_noise_max)
        return total_noise * (math.log(geometric_noise_level_max) - math.log(geometric_noise_level_min))
    raise ValueError(f"Unsupported noise_schedule={noise_schedule!r}")


def sample_time(
    batch_size: int,
    objective: str,
    noise_schedule: str,
    continuous_time_eps: float,
    antithetic_sampling: bool,
    device: torch.device,
    keep_prob_grid: torch.Tensor | None = None,  # [T+1] only for D3PM
    num_diffusion_steps: int = 0,
    geometric_noise_level_min: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN,
    geometric_noise_level_max: float = DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX,
    duo_log_noise_ratio_min: float = -3.55,
    duo_log_noise_ratio_max: float = -1.85,
    duo_keep_prob_table: dict[str, torch.Tensor | float | int] | None = None,
    duo_vocab_size: int | None = None,
) -> TimeBatch:
    # Each training example samples one time for the chosen objective's loss.
    # The training T is only used by D3PM: it samples an integer step in {1..T}.
    # Continuous-time objectives sample one scalar t in [eps, 1]. Text samplers
    # build their S-step grids in sample_text.py, not from training T.
    if objective in {"d3pm_mask", "d3pm_uniform"}:
        if keep_prob_grid is None:
            raise RuntimeError("D3PM training requires its T-step keep-probability grid.")
        steps = torch.randint(1, num_diffusion_steps + 1, (batch_size,), device=device)
        return TimeBatch(
            model_times=steps.float() / num_diffusion_steps,
            keep_prob=keep_prob_grid[steps],
            prev_keep_prob=keep_prob_grid[steps - 1],
        )

    if antithetic_sampling:
        jitter = torch.rand(batch_size, device=device)
        offsets = torch.arange(batch_size, device=device, dtype=torch.float32) / batch_size
        unit_times = torch.remainder(jitter / batch_size + offsets, 1.0)
    else:
        unit_times = torch.rand(batch_size, device=device)
    times = continuous_time_eps + (1.0 - continuous_time_eps) * unit_times

    if objective == "duo_uniform":
        if duo_keep_prob_table is None or duo_vocab_size is None:
            raise RuntimeError("DUO time sampling requires the keep-probability table and vocab size.")
        log_noise_ratio = duo_log_noise_ratio_min + times * (duo_log_noise_ratio_max - duo_log_noise_ratio_min)
        duo_keep_prob, duo_keep_prob_rate = duo_keep_prob_and_rate_from_table(
            log_noise_ratio,
            duo_keep_prob_table,
            vocab_size=duo_vocab_size,
            log_noise_ratio_min=duo_log_noise_ratio_min,
            log_noise_ratio_max=duo_log_noise_ratio_max,
        )
        return TimeBatch(
            model_times=-torch.log(duo_keep_prob.clamp_min(DUO_KEEP_PROB_EPS)),
            keep_prob=duo_keep_prob,
            loss_weight=duo_keep_prob_rate,
            duo_log_noise_ratio=log_noise_ratio,
        )

    keep_prob = keep_prob_from_time(
        times,
        noise_schedule,
        continuous_time_eps,
        geometric_noise_level_min,
        geometric_noise_level_max,
    )
    total_noise_rate = total_noise_rate_from_time(
        times,
        noise_schedule,
        continuous_time_eps,
        geometric_noise_level_min,
        geometric_noise_level_max,
    )
    model_times = -torch.log(keep_prob)
    if objective == "subs_mask":
        return TimeBatch(
            model_times=model_times,
            keep_prob=keep_prob,
            loss_weight=total_noise_rate * keep_prob / (1.0 - keep_prob),
        )
    if objective in {"sedd_mask", "sedd_uniform"}:
        return TimeBatch(
            model_times=model_times,
            keep_prob=keep_prob,
            loss_weight=total_noise_rate,
        )
    raise ValueError(f"Unsupported objective={objective!r}")


# =============================================================================
# FORWARD CORRUPTION
# =============================================================================


def sample_mask_forward_marginal(
    clean_tokens: torch.Tensor,  # [batch, seq]
    keep_prob: torch.Tensor,  # [batch]
    mask_token_id: int,
) -> torch.Tensor:
    keep_mask = torch.rand_like(clean_tokens, dtype=torch.float32) < keep_prob.unsqueeze(1)
    noisy_tokens = clean_tokens.clone()
    noisy_tokens[~keep_mask] = mask_token_id
    return noisy_tokens


def sample_uniform_forward_marginal(
    clean_tokens: torch.Tensor,  # [batch, seq]
    keep_prob: torch.Tensor,  # [batch]
    vocab_size: int,
) -> torch.Tensor:
    keep_mask = torch.rand_like(clean_tokens, dtype=torch.float32) < keep_prob.unsqueeze(1)
    random_tokens = torch.randint(vocab_size, clean_tokens.shape, device=clean_tokens.device)
    return torch.where(keep_mask, clean_tokens, random_tokens)


def default_duo_keep_prob_table_path() -> str:
    cache_root = os.environ.get("SPEEDRUN_DLM_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "speedrun-dlm"))
    return os.path.join(cache_root, "duo", f"gpt2-{DUO_KEEP_PROB_TABLE_COMMIT[:12]}.pkl")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_duo_keep_prob_table(path: str) -> str:
    if not path:
        path = default_duo_keep_prob_table_path()
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".duo-table-", suffix=".pkl", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            with urlopen(DUO_KEEP_PROB_TABLE_URL, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        actual_sha256 = file_sha256(tmp_path)
        if actual_sha256 != DUO_KEEP_PROB_TABLE_SHA256:
            raise ValueError(
                "downloaded DUO keep-probability table checksum mismatch: "
                f"expected {DUO_KEEP_PROB_TABLE_SHA256}, got {actual_sha256}"
            )
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_duo_keep_prob_table(path: str, device: str) -> dict[str, torch.Tensor | float | int]:
    path = ensure_duo_keep_prob_table(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    # DUO samples a gaussian latent at log-noise ratio gamma 
    # the DUO loss depends on the keep_prob of the uniform marginals obtained by argmaxing that gaussian latent 
    # this function loads a precomputed table, pt, used to get keep_prob as a function of gamma:
    # pt stores P(argmax gaussian latent is still the clean token) as a function of gamma, precomputed once since it is a gaussian max probability over vocab_size competitors
    # it is converted to keep_prob via P(clean) = pt[gamma] in Gaussian formulation = keep_prob + (1 - keep_prob) / vocab_size in uniform formulation 
    # so keep_prob = (vocab_size * pt[gamma] - 1) / (vocab_size - 1), see duo_keep_prob_from_table() for the conversion
    required = {"vocab_size", "gamma_min", "gamma_max", "num_points", "pt"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"DUO keep-probability table is missing keys: {sorted(missing)}")
    return {
        "vocab_size": int(data["vocab_size"]),
        "log_noise_ratio_min": float(data["gamma_min"]),
        "log_noise_ratio_max": float(data["gamma_max"]),
        "num_points": int(data["num_points"]),
        "clean_argmax_prob": torch.as_tensor(data["pt"], device=device),
    }


def duo_keep_prob_from_table(
    log_noise_ratio: torch.Tensor,  # [batch] DUO continuous relaxation time coordinate
    table: dict[str, torch.Tensor | float | int],
    vocab_size: int,
) -> torch.Tensor:
    # DUO table gives clean argmax prob, this function converts to uniform keep prob, see comments in load_duo_keep_prob_table()
    log_noise_ratio_min = float(table["log_noise_ratio_min"])
    log_noise_ratio_max = float(table["log_noise_ratio_max"])
    num_points = int(table["num_points"])
    log_noise_ratio = torch.clip(log_noise_ratio, log_noise_ratio_min, log_noise_ratio_max)
    indices = torch.round(
        (num_points - 1) * (log_noise_ratio - log_noise_ratio_min) / (log_noise_ratio_max - log_noise_ratio_min)
    ).long()
    clean_argmax_prob = table["clean_argmax_prob"][indices]
    return (vocab_size * clean_argmax_prob - 1) / (vocab_size - 1)


def duo_keep_prob_and_rate_from_table(
    log_noise_ratio: torch.Tensor,  # [batch] DUO continuous relaxation time coordinate
    table: dict[str, torch.Tensor | float | int],
    vocab_size: int,
    log_noise_ratio_min: float,
    log_noise_ratio_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_prob = duo_keep_prob_from_table(log_noise_ratio, table, vocab_size)
    log_noise_ratio_span = log_noise_ratio_max - log_noise_ratio_min
    finite_difference_steps = 1000
    keep_prob_rate = log_noise_ratio_span * finite_difference_steps * (
        duo_keep_prob_from_table(log_noise_ratio + 1 / finite_difference_steps, table, vocab_size)
        - keep_prob
    )
    return keep_prob, keep_prob_rate


def duo_input_inverse_temperature(
    step: int,
    temperature_log10_start: float,
    temperature_log10_end: float,
    curriculum_start: int,
    curriculum_end: int,
) -> float:
    # DUO argmax of gaussian latents has uniform state marginals
    # training feeds softmax(latents / temp) and lowers temp toward argmax
    if step < curriculum_start:
        temperature_log10 = temperature_log10_start
    elif step < curriculum_end:
        frac = (step - curriculum_start) / max(curriculum_end - curriculum_start, 1)
        temperature_log10 = temperature_log10_start + frac * (temperature_log10_end - temperature_log10_start)
    else:
        temperature_log10 = -10.0  # after the curriculum, DUO uses a near-argmax softmax
    return 10 ** (-temperature_log10)


def _sample_k_int(bs: int, length: int, k: int, max_value: int, device: torch.device) -> torch.Tensor:
    # vectorized partial shuffle: k distinct ids from range(max_value) for each batch/position
    out = torch.empty(size=(bs, length, k), dtype=torch.int64, device=device)
    for t, i in enumerate(range(max_value - k, max_value)):
        j = torch.randint(0, i + 1, size=(bs, length), device=device)
        if t > 0:
            duplicate = (out[..., :t] == j[..., None]).any(dim=-1)
            out[..., t] = torch.where(duplicate, i, j)
        else:
            out[..., 0] = j
    return out


def _sample_topk_gaussian(
    N: int,
    noise_scale: torch.Tensor,  # [batch]
    length: int,
    k: int,
) -> torch.Tensor:
    # top-k order statistics for N iid zero-mean Gaussian competitors
    batch = noise_scale.shape[0]
    device = noise_scale.device
    dtype = noise_scale.dtype
    log_u = torch.log(torch.rand(batch, length, k, device=device, dtype=dtype))
    divisors = torch.arange(N, N - k, -1, device=device, dtype=dtype)
    log_rj = log_u / divisors
    uniforms = torch.exp(torch.cumsum(log_rj, dim=-1))
    return torch.special.ndtri(uniforms) * noise_scale[:, None, None]


def _sample_topk_and_extra(
    N: int,
    clean_scale: torch.Tensor,
    noise_scale: torch.Tensor,
    length: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_k_others = _sample_topk_gaussian(N - 1, noise_scale, length, k)
    extra = clean_scale[:, None] + torch.randn(size=(clean_scale.shape[0], length), device=clean_scale.device) * noise_scale[:, None]
    min_values = top_k_others[:, :, -1]
    is_extra_in_topk = extra > min_values
    top_k_others[:, :, -1][is_extra_in_topk] = extra[is_extra_in_topk]
    return extra, top_k_others, is_extra_in_topk


def _log_mean_exp_trunc_normal(cutoff: torch.Tensor, normal_std: torch.Tensor) -> torch.Tensor:
    log_num = torch.special.log_ndtr((cutoff - normal_std**2) / normal_std)
    log_den = torch.special.log_ndtr(cutoff / normal_std)
    return normal_std**2 / 2.0 + log_num - log_den


def sample_duo_tempered_softmax_topk(
    extra_index: torch.Tensor,  # [batch, seq] clean token ids
    clean_scale: torch.Tensor,  # [batch] clean-token Gaussian mean scale
    noise_scale: torch.Tensor,  # [batch] Gaussian noise std
    length: int,
    k: int,
    vocab_size: int,
    inverse_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # target: softmax(inverse_temperature * noisy_vocab_vector) without materializing all vocab columns
    # sample the top-k non-clean competitors, add the clean token candidate, and integrate the remaining tail
    # DUO topk-softmax gaussian relaxation
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/utils.py#L529-L655
    if k < 2 or k >= vocab_size:
        raise ValueError(f"DUO topk-softmax requires 2 <= k < vocab_size, got k={k}, vocab_size={vocab_size}.")
    clean_scale = clean_scale.to(torch.float64)
    noise_scale = noise_scale.to(torch.float64)
    extra, top_k, is_extra_in_topk = _sample_topk_and_extra(vocab_size, clean_scale, noise_scale, length, k)
    min_rv = torch.where(is_extra_in_topk, top_k[:, :, -2], top_k[:, :, -1])
    scaled_noise_std = noise_scale[:, None] * inverse_temperature
    scaled_c = min_rv * inverse_temperature
    log_mu = _log_mean_exp_trunc_normal(scaled_c, scaled_noise_std)
    log_topk = top_k * inverse_temperature
    count = torch.where(is_extra_in_topk, vocab_size - k, vocab_size - k - 1).to(log_mu.dtype)
    log_tail = torch.log(count) + log_mu
    log_extra = extra * inverse_temperature
    extra_not_in_topk = ~is_extra_in_topk
    log_extra_masked = torch.full_like(log_tail, float("-inf"))
    log_extra_masked[extra_not_in_topk] = log_extra[extra_not_in_topk]
    log_contribs = torch.cat([log_topk, log_tail[..., None], log_extra_masked[..., None]], dim=-1)
    log_denom = torch.logsumexp(log_contribs, dim=-1, keepdim=True)
    softmax_approx = torch.exp(log_topk - log_denom)
    normalizer = softmax_approx.sum(dim=-1, keepdim=True)
    zero_sum = normalizer == 0.0
    softmax_approx = torch.where(zero_sum, 0.0, softmax_approx)
    softmax_approx[..., 0][zero_sum[..., 0]] = 1.0
    indices = _sample_k_int(clean_scale.shape[0], length, k, vocab_size - 1, device=clean_scale.device)
    indices[indices >= extra_index[..., None]] += 1
    indices[..., -1][is_extra_in_topk] = extra_index[is_extra_in_topk]
    xt_usdm = torch.where(is_extra_in_topk, extra_index, indices[..., 0])
    return softmax_approx, indices, xt_usdm


def sample_duo_forward_relaxation(
    clean_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    curriculum_mode: str,
    top_k: int,
    inverse_temperature: float,
    vocab_size: int,
) -> torch.Tensor:
    # DUO dense-softmax and topk-softmax curricula
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/algo.py#L582-L636
    curriculum_mode = check_duo_curriculum_mode(curriculum_mode)
    clean_scale = torch.sigmoid(-time_batch.duo_log_noise_ratio).sqrt()
    noise_scale = torch.sigmoid(time_batch.duo_log_noise_ratio).sqrt()
    if curriculum_mode == "dense_softmax":
        # DUO replaces exact argmax input with a full vocab softmax
        # DUO dense-softmax builds dense batch seq vocab tensors so memory gets huge
        x0_one_hot = F.one_hot(clean_tokens, vocab_size).to(torch.float32)
        xt = clean_scale[:, None, None] * x0_one_hot + noise_scale[:, None, None] * torch.randn_like(x0_one_hot)
        xt = xt * inverse_temperature
        time_batch.duo_discrete_noisy_tokens = xt.argmax(-1)
        time_batch.duo_sparse_input_weights = None
        return xt
    if curriculum_mode != "topk_softmax":
        raise NotImplementedError(f"Unsupported DUO curriculum mode: {curriculum_mode!r}.")
    # DUO topk-softmax keeps sparse softmax weights instead of all vocab columns
    weights, indices, xt_usdm = sample_duo_tempered_softmax_topk(
        extra_index=clean_tokens,
        clean_scale=clean_scale,
        noise_scale=noise_scale,
        length=clean_tokens.shape[1],
        k=top_k,
        vocab_size=vocab_size,
        inverse_temperature=inverse_temperature,
    )
    time_batch.duo_discrete_noisy_tokens = xt_usdm
    time_batch.duo_sparse_input_weights = weights
    return indices


def sample_forward_marginal(
    clean_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    objective: str,
    noise_type: str,
    mask_token_id: int,
    vocab_size: int,
    duo_curriculum_mode: str = "dense_softmax",
    duo_top_k: int = 32,
    duo_input_inverse_temperature: float = 1.0,
) -> torch.Tensor:
    if objective == "duo_uniform":
        return sample_duo_forward_relaxation(
            clean_tokens,
            time_batch,
            curriculum_mode=duo_curriculum_mode,
            top_k=duo_top_k,
            inverse_temperature=duo_input_inverse_temperature,
            vocab_size=vocab_size,
        )
    if noise_type == "mask":
        return sample_mask_forward_marginal(clean_tokens, time_batch.keep_prob, mask_token_id)
    if noise_type == "uniform":
        return sample_uniform_forward_marginal(clean_tokens, time_batch.keep_prob, vocab_size)
    raise ValueError(f"Unsupported noise_type={noise_type!r}")


# =============================================================================
# SUBS MASK LOSS
# =============================================================================

# SUBS mask and absorbing-state D3PM share the same mask-corruption family.
# SUBS uses a continuous-time x0 CE weight. Absorbing D3PM's posterior KL
# collapses to the same masked x0 CE with a discrete posterior weight.


def x0_weighted_ce(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    token_weights: torch.Tensor | None = None,  # [batch] or [batch, seq]
    mask: torch.Tensor | None = None,  # [batch, seq]
    normalization: str = "token_mean",
) -> torch.Tensor:
    """Weighted x0 CE: weights set importance, mask excludes padding (all tokens by default)."""
    logprobs = F.log_softmax(raw_logits.float(), dim=-1)
    nll = -logprobs.gather(-1, clean_tokens.unsqueeze(-1)).squeeze(-1)
    if token_weights is None:
        token_weights = torch.ones_like(nll)
    else:
        token_weights = token_weights.to(device=nll.device, dtype=nll.dtype)
        if token_weights.ndim == 1 and token_weights.shape[0] == nll.shape[0]:
            token_weights = token_weights.unsqueeze(1)
    if mask is None:
        mask = torch.ones_like(nll)
    else:
        mask = mask.to(device=nll.device, dtype=nll.dtype)
    weighted_mask = token_weights * mask
    numerator = (nll * weighted_mask).sum()
    if normalization == "token_mean":
        return numerator / mask.sum().clamp_min(1.0)
    if normalization == "masked_weight_mean":
        return numerator / weighted_mask.sum().clamp_min(1.0)
    raise ValueError(f"Unsupported x0 CE normalization: {normalization!r}")


def subs_mask_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    mask_token_id: int,
    normalization: str,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    # SUBS mask-only weighted CE:
    # https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/diffusion.py#L847-L894
    token_weights = noisy_tokens.eq(mask_token_id) * time_batch.loss_weight.unsqueeze(1)
    loss = x0_weighted_ce(
        raw_logits,
        clean_tokens,
        token_weights,
        mask=attention_mask,
        normalization=normalization,
    )
    return loss, {"ce_loss": float(loss.detach().item())}


# =============================================================================
# D3PM POSTERIOR-MATCHING LOSSES
# =============================================================================


def d3pm_reverse_posterior_probs(
    x0: torch.Tensor,  # [batch, seq] tokens or [batch, seq, vocab] logits
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    noise_type: str,
    vocab_size: int,
    mask_token_id: int,
    x0_is_logits: bool,
) -> torch.Tensor:
    # Dense D3PM posterior q(x_{t-1} | x_t, x0):
    # https://github.com/google-research/google-research/blob/1fa17414f56c3703d5adb3818338b6e35e0fd550/d3pm/text/diffusion.py#L303-L428
    # https://github.com/google-research/google-research/blob/1fa17414f56c3703d5adb3818338b6e35e0fd550/d3pm/text/diffusion.py#L2431-L2514
    if time_batch.prev_keep_prob is None:
        raise RuntimeError("D3PM posterior requires the previous grid keep_prob.")
    posterior_dtype = x0.dtype if x0_is_logits and x0.is_floating_point() else time_batch.keep_prob.dtype
    if time_batch.keep_prob.device.type == "cuda" and posterior_dtype == torch.float32:
        posterior_dtype = torch.bfloat16
    tiny = torch.finfo(posterior_dtype).tiny
    prev_keep_prob = time_batch.prev_keep_prob.view(-1, 1, 1).to(dtype=posterior_dtype)
    keep_prob = time_batch.keep_prob.view(-1, 1, 1).to(dtype=posterior_dtype)
    one = keep_prob.new_tensor(1.0)
    beta_t = (one - keep_prob / prev_keep_prob.clamp_min(tiny)).clamp(1e-6, 0.999)

    if noise_type == "uniform":
        inv_support = keep_prob.new_tensor(1.0 / vocab_size)
        if x0_is_logits:
            x0_probs = torch.softmax(x0.to(dtype=posterior_dtype), dim=-1)
        else:
            x0_probs = F.one_hot(x0, num_classes=vocab_size).to(dtype=posterior_dtype)
        prev_probs = prev_keep_prob * x0_probs + (one - prev_keep_prob) * inv_support
        base = beta_t * inv_support
        xt_prev_probs = prev_probs.gather(-1, noisy_tokens.unsqueeze(-1))
        posterior = prev_probs * base
        posterior.scatter_add_(-1, noisy_tokens.unsqueeze(-1), (one - beta_t) * xt_prev_probs)
        denom = (base + (one - beta_t) * xt_prev_probs).clamp_min(tiny)
        return posterior / denom

    if noise_type == "mask":
        support_size = vocab_size + 1
        if x0_is_logits:
            x0_probs = torch.softmax(x0.to(dtype=posterior_dtype), dim=-1)
            x0_probs = torch.cat(
                [x0_probs, torch.zeros(*x0_probs.shape[:2], 1, device=x0_probs.device, dtype=x0_probs.dtype)],
                dim=-1,
            )
        else:
            x0_probs = F.one_hot(x0, num_classes=support_size).to(dtype=posterior_dtype)
        mask_probs = F.one_hot(torch.full_like(noisy_tokens, mask_token_id), num_classes=support_size).to(dtype=posterior_dtype)
        prev_probs = prev_keep_prob * x0_probs + (one - prev_keep_prob) * mask_probs
        xt_onehot = F.one_hot(noisy_tokens, num_classes=support_size).to(dtype=posterior_dtype)
        xt_is_mask = noisy_tokens.eq(mask_token_id).unsqueeze(-1).to(dtype=posterior_dtype)
        step_probs = (one - beta_t) * xt_onehot + beta_t * xt_is_mask
        unnorm = prev_probs * step_probs
        return unnorm / unnorm.sum(dim=-1, keepdim=True).clamp_min(tiny)

    raise ValueError(f"Unsupported noise_type={noise_type!r}")


def d3pm_vb_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    noise_type: str,
    mask_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_posterior = d3pm_reverse_posterior_probs(
        raw_logits,
        noisy_tokens,
        time_batch,
        noise_type=noise_type,
        vocab_size=raw_logits.size(-1),
        mask_token_id=mask_token_id,
        x0_is_logits=True,
    )
    tiny = torch.finfo(pred_posterior.dtype).tiny
    pred_log = pred_posterior.clamp_min(tiny).log()
    true_posterior = d3pm_reverse_posterior_probs(
        clean_tokens,
        noisy_tokens,
        time_batch,
        noise_type=noise_type,
        vocab_size=raw_logits.size(-1),
        mask_token_id=mask_token_id,
        x0_is_logits=False,
    )
    per_token_kl = (true_posterior * (true_posterior.clamp_min(tiny).log() - pred_log)).sum(dim=-1)
    if attention_mask is None:
        return per_token_kl.mean()
    attention_mask = attention_mask.to(device=per_token_kl.device, dtype=per_token_kl.dtype)
    return (per_token_kl * attention_mask).sum() / attention_mask.sum().clamp_min(1.0)


def d3pm_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    noise_type: str,
    vb_coeff: float,
    ce_coeff: float,
    mask_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if noise_type == "mask":  # For mask diffusion, the VB term simplifies to weighted CE, so we call x0_weighted_ce to highlight this shared backbone with SUBS mask
        if time_batch.prev_keep_prob is None:
            raise RuntimeError("D3PM mask VB requires time_batch.prev_keep_prob.")
        posterior_weight = (time_batch.prev_keep_prob - time_batch.keep_prob) / (1.0 - time_batch.keep_prob)
        token_weights = noisy_tokens.eq(mask_token_id) * posterior_weight.clamp_min(0.0).unsqueeze(1)
        vb_loss = x0_weighted_ce(
            raw_logits,
            clean_tokens,
            token_weights,
            mask=attention_mask,
        )
    else:  # For non-mask D3PM (currently uniform), compute the full KL
        vb_loss = d3pm_vb_loss(
            raw_logits,
            clean_tokens,
            noisy_tokens,
            time_batch,
            noise_type=noise_type,
            mask_token_id=mask_token_id,
            attention_mask=attention_mask,
        )
    ce_loss = x0_weighted_ce(raw_logits, clean_tokens, mask=attention_mask)
    loss = vb_coeff * vb_loss + ce_coeff * ce_loss
    return loss, {"ce_loss": float(ce_loss.detach().item()), "vb_loss": float(vb_loss.detach().item())}


# =============================================================================
# SEDD SCORE-ENTROPY LOSSES
# =============================================================================


def sedd_mask_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    mask_token_id: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    tiny = torch.finfo(time_batch.keep_prob.dtype).tiny
    keep_prob = time_batch.keep_prob.clamp_min(tiny)
    noise_prob = (1.0 - keep_prob).clamp_min(tiny)
    log_noise_to_keep = noise_prob.log() - keep_prob.log()
    log_scores = raw_logits.float() - log_noise_to_keep[:, None, None] - math.log(raw_logits.shape[-1])
    mask_column = torch.zeros(*log_scores.shape[:2], 1, device=log_scores.device, dtype=log_scores.dtype)
    log_scores = torch.cat([log_scores, mask_column], dim=-1)
    log_scores = torch.scatter(log_scores, -1, noisy_tokens.unsqueeze(-1), torch.zeros_like(log_scores[..., :1]))
    masked_positions = noisy_tokens.eq(mask_token_id)
    entropy = torch.zeros_like(noisy_tokens, dtype=log_scores.dtype)
    if masked_positions.any():
        keep_to_noise = (keep_prob / noise_prob).unsqueeze(1).expand_as(noisy_tokens)
        ratio = keep_to_noise[masked_positions]
        masked_scores = log_scores[masked_positions]
        target_tokens = clean_tokens[masked_positions]
        neg_term = ratio * masked_scores.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        pos_term = masked_scores[:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1.0)
        entropy[masked_positions] = pos_term - neg_term + const
    loss = (time_batch.loss_weight.unsqueeze(1) * entropy).sum(dim=-1).mean()
    return loss, {"score_entropy": float(loss.detach().item())}


def sedd_uniform_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    time_batch: TimeBatch,
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    # https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/graph_lib.py#L116-L189
    log_scores = raw_logits.float()
    log_scores = torch.scatter(log_scores, -1, noisy_tokens.unsqueeze(-1), torch.zeros_like(log_scores[..., :1]))
    tiny = torch.finfo(time_batch.keep_prob.dtype).tiny
    keep_prob = time_batch.keep_prob.clamp_min(tiny).unsqueeze(1)
    noise_prob = (1.0 - time_batch.keep_prob).clamp_min(tiny).unsqueeze(1)
    noise_to_keep = noise_prob / keep_prob
    ratio = noise_to_keep / (noise_to_keep + vocab_size)
    clean_score = log_scores.gather(-1, clean_tokens.unsqueeze(-1)).squeeze(-1)
    neg_term = log_scores.mean(dim=-1)
    same = noisy_tokens.eq(clean_tokens)
    neg_term = torch.where(same, ratio * neg_term, clean_score * keep_prob / noise_prob + neg_term)
    ratio_for_log = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
    const_same = (vocab_size - 1) / vocab_size * ratio * (ratio_for_log.log() - 1.0)
    const_diff = ((-ratio_for_log.log() - 1.0) / ratio_for_log - (vocab_size - 2)) / vocab_size
    const = torch.where(same, const_same, const_diff)
    pos_term = log_scores.exp().mean(dim=-1) - (1.0 / vocab_size)
    entropy = pos_term - neg_term + const
    loss = (time_batch.loss_weight.unsqueeze(1) * entropy).sum(dim=-1).mean()
    return loss, {"score_entropy": float(loss.detach().item())}


# =============================================================================
# DUO UNIFORM LOSSES
# =============================================================================


def duo_uniform_nll_per_token(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    noisy_tokens: torch.Tensor,  # [batch, seq]
    clean_tokens: torch.Tensor,  # [batch, seq]
    keep_prob: torch.Tensor,  # [batch, 1]
    keep_prob_rate: torch.Tensor,  # [batch, 1]
    vocab_size: int,
) -> torch.Tensor:
    # this is the continuous time uniform state nll using keep prob and its rate
    # https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/algo.py#L337-L372
    pred_clean_probs = F.softmax(raw_logits.float(), dim=-1)
    scaled_pred_noisy_probs = (
        vocab_size * keep_prob[:, :, None] * pred_clean_probs
        + 1
        - keep_prob[:, :, None]
    )
    same_token = (clean_tokens == noisy_tokens).float()
    different_token = 1 - same_token

    scaled_true_noisy_prob = (1 - keep_prob) + vocab_size * keep_prob * same_token
    pred_noisy_at_noisy = torch.gather(scaled_pred_noisy_probs, -1, noisy_tokens.unsqueeze(-1)).squeeze(-1)
    pred_noisy_at_clean = torch.gather(scaled_pred_noisy_probs, -1, clean_tokens.unsqueeze(-1)).squeeze(-1)
    first_term = vocab_size * (1 / scaled_true_noisy_prob - 1 / pred_noisy_at_noisy)

    uniform_floor = (1 - keep_prob) / (vocab_size * keep_prob + 1 - keep_prob)
    second_term_weight = same_token * uniform_floor + different_token
    second_term_offset = (
        (vocab_size - 1) * uniform_floor * same_token
        - (1 / uniform_floor) * different_token
    ) * uniform_floor.log()
    second_term_model = -second_term_weight * (
        scaled_pred_noisy_probs.log().sum(-1) - vocab_size * pred_noisy_at_noisy.log()
    )
    second_term_model = (
        second_term_model
        - vocab_size
        * keep_prob
        / (1 - keep_prob)
        * (pred_noisy_at_clean.log() - pred_noisy_at_noisy.log())
        * different_token
    )
    second_term = second_term_model + second_term_offset
    loss_scale = keep_prob_rate / (vocab_size * keep_prob)
    return loss_scale * (first_term - second_term)


def duo_uniform_loss(
    raw_logits: torch.Tensor,
    clean_tokens: torch.Tensor,
    time_batch: TimeBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    if time_batch.duo_discrete_noisy_tokens is None:
        raise RuntimeError("DUO loss requires sample_duo_forward_relaxation to populate discrete noisy tokens.")
    keep_prob = time_batch.keep_prob.unsqueeze(-1)
    keep_prob_rate = time_batch.loss_weight.unsqueeze(-1)
    loss_matrix = duo_uniform_nll_per_token(
        raw_logits=raw_logits,
        noisy_tokens=time_batch.duo_discrete_noisy_tokens,
        clean_tokens=clean_tokens,
        keep_prob=keep_prob,
        keep_prob_rate=keep_prob_rate,
        vocab_size=raw_logits.size(-1),
    )
    loss = loss_matrix.mean()
    return loss, {"duo_usdm_loss": float(loss.detach().item())}


# =============================================================================
# OBJECTIVE LOSS ROUTING
# =============================================================================


def compute_dlm_loss(
    raw_logits: torch.Tensor,  # [batch, seq, vocab]
    clean_tokens: torch.Tensor,  # [batch, seq]
    noisy_tokens: torch.Tensor,  # [batch, seq] or DUO sparse ids
    time_batch: TimeBatch,
    objective: str,
    noise_type: str,
    mask_token_id: int,
    subs_mask_normalization: str = "token_mean",
    d3pm_vb_coeff: float = 1.0,
    d3pm_ce_coeff: float = 0.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "subs_mask":
        return subs_mask_loss(
            raw_logits,
            clean_tokens,
            noisy_tokens,
            time_batch,
            mask_token_id,
            subs_mask_normalization,
            attention_mask=attention_mask,
        )
    if objective in {"d3pm_mask", "d3pm_uniform"}:
        return d3pm_loss(
            raw_logits,
            clean_tokens,
            noisy_tokens,
            time_batch,
            noise_type=noise_type,
            vb_coeff=d3pm_vb_coeff,
            ce_coeff=d3pm_ce_coeff,
            mask_token_id=mask_token_id,
            attention_mask=attention_mask,
        )
    if objective == "sedd_mask":
        return sedd_mask_loss(raw_logits, clean_tokens, noisy_tokens, time_batch, mask_token_id)
    if objective == "sedd_uniform":
        return sedd_uniform_loss(raw_logits, clean_tokens, noisy_tokens, time_batch, raw_logits.size(-1))
    if objective == "duo_uniform":
        return duo_uniform_loss(raw_logits, clean_tokens, time_batch)
    raise ValueError(f"Unsupported objective={objective!r}")


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def objective_summary_lines(args) -> list[str]:
    spec = DLM_OBJECTIVES[args.objective]
    lines = [
        f"objective: {args.objective} ({args.noise_type} corruption)",
        f"loss: {spec.loss_label}",
    ]
    if is_d3pm_objective(args.objective):
        lines.append(f"time grid: discrete {args.num_diffusion_steps} steps")
        lines.append(f"d3pm_coeffs: vb={args.d3pm_vb_coeff:g}, ce={args.d3pm_ce_coeff:g}")
    else:
        lines.append(f"time grid: continuous eps={args.continuous_time_eps:g}")

    if args.objective == "subs_mask":
        lines.append(f"subs_mask_normalization: {args.subs_mask_normalization}")
    if args.objective.startswith("sedd_"):
        state_space = "mask" if args.objective == "sedd_mask" else "uniform"
        lines.append(f"sedd_state_space: {state_space}")
    if args.objective == "duo_uniform":
        lines.append(
            "duo_curriculum: "
            f"{args.duo_curriculum_mode}, log_noise_ratio=[{args.duo_log_noise_ratio_min:g}, {args.duo_log_noise_ratio_max:g}], "
            f"softmax_temperature_log10=[{args.duo_softmax_temperature_log10_start:g}, {args.duo_softmax_temperature_log10_end:g}], "
            f"top_k={args.duo_top_k}"
        )
    if args.noise_schedule == "geometric":
        lines.append(f"noise_schedule: geometric total_noise=[{args.geometric_noise_level_min:g}, {args.geometric_noise_level_max:g}]")
    else:
        lines.append(f"noise_schedule: {args.noise_schedule}")
    return lines


def objective_summary_fields(args) -> dict:
    # structured TRAIN_RESULT fields; objective_summary_lines is only for humans
    fields = {
        "noise_type": args.noise_type,
        "objective": args.objective,
        "noise_schedule": args.noise_schedule,
        "recipe": args.objective,
    }
    if is_d3pm_objective(args.objective):
        fields["num_diffusion_steps"] = args.num_diffusion_steps
    else:
        fields["continuous_time_eps"] = args.continuous_time_eps
    if args.objective == "subs_mask":
        fields["subs_mask_normalization"] = args.subs_mask_normalization
    if is_d3pm_objective(args.objective):
        fields["d3pm_vb_coeff"] = args.d3pm_vb_coeff
        fields["d3pm_ce_coeff"] = args.d3pm_ce_coeff
    if args.noise_schedule == "geometric":
        fields["geometric_noise_level_min"] = args.geometric_noise_level_min
        fields["geometric_noise_level_max"] = args.geometric_noise_level_max
    if args.objective == "duo_uniform":
        fields.update(
            duo_curriculum_mode=args.duo_curriculum_mode,
            duo_top_k=args.duo_top_k,
            duo_log_noise_ratio_min=args.duo_log_noise_ratio_min,
            duo_log_noise_ratio_max=args.duo_log_noise_ratio_max,
            duo_softmax_temperature_log10_start=args.duo_softmax_temperature_log10_start,
            duo_softmax_temperature_log10_end=args.duo_softmax_temperature_log10_end,
            duo_curriculum_start=args.duo_curriculum_start,
            duo_curriculum_end=args.duo_curriculum_end,
            duo_keep_prob_table_path=args.duo_keep_prob_table_path,
        )
    return fields


# =============================================================================
# CONFIGURATION
# =============================================================================


def apply_dlm_objective_defaults(args) -> None:
    spec = DLM_OBJECTIVES[args.objective]
    if args.noise_schedule is None:
        args.noise_schedule = spec.default_noise_schedule
    if args.time_conditioning is None:
        args.time_conditioning = spec.default_time_conditioning


def get_model_config(
    model_name: str,
    cond_dim: int = 128,
    dropout: float = 0.1,
    time_conditioning: bool = False,
) -> DLMConfig:
    configs = {
        "d12": DLMConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": DLMConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": DLMConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": DLMConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }
    config = configs[model_name]
    config.cond_dim = cond_dim
    config.dropout = dropout
    config.time_conditioning = time_conditioning
    return config


# =============================================================================
# TRAINING ENTRYPOINT
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val_files", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model", choices=("d12", "d24", "d36", "d48"), default="d12")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--total_batch_size", type=int, default=50 * 1024 * 8)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--lr_schedule", choices=("constant", "cosine", "inverse_sqrt"), default="constant")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_loss_every", type=int, default=0)
    parser.add_argument("--val_max_steps", type=int, default=20)
    parser.add_argument("--num_checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint_steps", type=str, default="")
    parser.add_argument("--skip_checkpoint_save", action="store_true")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000, help="D3PM training steps T; unused by continuous-time objectives")
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--continuous_time_eps", type=float, default=1e-3)
    parser.add_argument("--objective", choices=tuple(DLM_OBJECTIVES), default="subs_mask")
    parser.add_argument("--noise_schedule", choices=("linear", "loglinear", "cosine", "geometric"), default=None)
    parser.add_argument("--geometric_noise_level_min", type=float, default=DEFAULT_GEOMETRIC_NOISE_LEVEL_MIN)
    parser.add_argument("--geometric_noise_level_max", type=float, default=DEFAULT_GEOMETRIC_NOISE_LEVEL_MAX)
    parser.add_argument(
        "--subs_mask_normalization",
        dest="subs_mask_normalization",
        choices=("token_mean", "masked_weight_mean"),
        default="token_mean",
    )
    parser.add_argument("--d3pm_vb_coeff", type=float, default=None)
    parser.add_argument("--d3pm_ce_coeff", type=float, default=None)
    parser.add_argument(
        "--duo_curriculum_mode",
        choices=DUO_CURRICULUM_MODES,
        default="dense_softmax",
        help="DUO input relaxation: dense_softmax materializes [B,L,V]; topk_softmax uses sparse top-k weights.",
    )
    parser.add_argument(
        "--duo_keep_prob_table_path",
        type=str,
        default="",
        help="Pickle table mapping DUO gaussian log noise ratio to uniform keep probability; downloaded to cache if omitted.",
    )
    parser.add_argument("--duo_top_k", type=int, default=32)
    parser.add_argument("--duo_log_noise_ratio_min", type=float, default=-3.55)
    parser.add_argument("--duo_log_noise_ratio_max", type=float, default=-1.85)
    parser.add_argument(
        "--duo_softmax_temperature_log10_start",
        type=float,
        default=-3.0,
    )
    parser.add_argument(
        "--duo_softmax_temperature_log10_end",
        type=float,
        default=-3.0,
    )
    parser.add_argument("--duo_curriculum_start", type=int, default=0)
    parser.add_argument("--duo_curriculum_end", type=int, default=500_000)
    parser.add_argument("--time_conditioning", dest="time_conditioning", action="store_true")
    parser.add_argument("--no_time_conditioning", dest="time_conditioning", action="store_false")
    parser.set_defaults(time_conditioning=None)
    parser.add_argument("--antithetic_sampling", dest="antithetic_sampling", action="store_true")
    parser.add_argument("--no_antithetic_sampling", dest="antithetic_sampling", action="store_false")
    parser.set_defaults(antithetic_sampling=True)
    args = parser.parse_args()
    args.duo_curriculum_mode = check_duo_curriculum_mode(args.duo_curriculum_mode)

    batch_size, seq_len = args.batch_size, args.sequence_length
    if not (1 <= seq_len <= 1024):
        raise ValueError("--sequence_length must be between 1 and 1024.")
    if args.num_checkpoints < 1:
        raise ValueError("--num_checkpoints must be at least 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trainer.")
    apply_dlm_objective_defaults(args)
    objective_spec = DLM_OBJECTIVES[args.objective]
    args.noise_type = objective_spec.noise_type
    if objective_spec.d3pm_default_coeffs is not None:
        default_vb, default_ce = objective_spec.d3pm_default_coeffs
        if args.d3pm_vb_coeff is None:
            args.d3pm_vb_coeff = default_vb
        if args.d3pm_ce_coeff is None:
            args.d3pm_ce_coeff = default_ce
        if args.d3pm_vb_coeff < 0 or args.d3pm_ce_coeff < 0:
            raise ValueError("D3PM coefficients must be non-negative.")
        if args.d3pm_vb_coeff == 0 and args.d3pm_ce_coeff == 0:
            raise ValueError("At least one D3PM coefficient must be positive.")
    if args.objective == "duo_uniform":
        if args.duo_curriculum_mode == "topk_softmax" and args.duo_top_k < 2:
            raise ValueError("--duo_top_k must be >=2 for DUO topk_softmax curriculum.")
        if not args.time_conditioning:
            print0("objective=duo_uniform follows official DUO UniformState and requires time_conditioning; enabling it.")
            args.time_conditioning = True
    if args.noise_schedule == "geometric":
        if args.geometric_noise_level_min <= 0 or args.geometric_noise_level_max <= args.geometric_noise_level_min:
            raise ValueError("--geometric_noise_level_min must be > 0 and < --geometric_noise_level_max.")
        if args.objective != "sedd_uniform":
            print0(
                "noise_schedule=geometric is the official SEDD-uniform schedule; "
                f"continuing with objective={args.objective} for diagnostics."
            )

    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    rank_seed = args.seed + ddp_rank
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    print0(f"PyTorch: {torch.version.__version__}")
    print0(f"Device: {device}")
    print0(f"seed: base {args.seed} | rank0 local seed {args.seed}")

    tokens_per_step = batch_size * seq_len * ddp_world_size
    assert args.total_batch_size == tokens_per_step

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model = DiffusionTransformer(
        get_model_config(
            args.model,
            cond_dim=args.cond_dim,
            dropout=args.dropout,
            time_conditioning=args.time_conditioning,
        )
    ).train().cuda()
    num_parameters = sum(p.numel() for p in model.parameters())
    non_embedding_parameters = sum(p.numel() for name, p in model.named_parameters() if "transformer.wte" not in name)
    if args.objective == "duo_uniform":
        args._duo_vocab_size = model.config.vocab_size
        args._duo_keep_prob_table = load_duo_keep_prob_table(args.duo_keep_prob_table_path, device)
        table_vocab = int(args._duo_keep_prob_table["vocab_size"])
        if table_vocab != model.config.vocab_size:
            raise ValueError(
                f"DUO keep-probability table vocab_size={table_vocab} does not match model vocab_size={model.config.vocab_size}."
            )

    if hasattr(inductor_config, "coordinate_descent_tuning"):
        inductor_config.coordinate_descent_tuning = True
    print0("Compiling model...")
    model = torch.compile(model)

    train_loader = DistributedDataLoader(args.train_files, batch_size, seq_len, ddp_rank, ddp_world_size, name="train")
    val_loader = (
        DistributedDataLoader(args.val_files, batch_size, seq_len, ddp_rank, ddp_world_size, name="val")
        if args.val_files
        else None
    )
    x = train_loader.next_batch()

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    checkpoint_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    keep_prob_grid = None
    if is_d3pm_objective(args.objective):
        # D3PM training grid: t_i=i/T, then keep_prob_i=schedule(t_i).
        grid_times = torch.linspace(0.0, 1.0, args.num_diffusion_steps + 1, device=device, dtype=torch.float32)
        keep_prob_grid = keep_prob_from_time(
            grid_times,
            args.noise_schedule,
            args.continuous_time_eps,
            args.geometric_noise_level_min,
            args.geometric_noise_level_max,
        )
        keep_prob_grid[0] = 1.0

    def sample_training_time(batch_count: int, batch_device: torch.device) -> TimeBatch:
        if is_d3pm_objective(args.objective):
            return sample_time(
                batch_count,
                args.objective,
                args.noise_schedule,
                args.continuous_time_eps,
                args.antithetic_sampling,
                batch_device,
                keep_prob_grid=keep_prob_grid,
                num_diffusion_steps=args.num_diffusion_steps,
                geometric_noise_level_min=args.geometric_noise_level_min,
                geometric_noise_level_max=args.geometric_noise_level_max,
            )
        if args.objective == "duo_uniform":
            return sample_time(
                batch_count,
                args.objective,
                args.noise_schedule,
                args.continuous_time_eps,
                args.antithetic_sampling,
                batch_device,
                geometric_noise_level_min=args.geometric_noise_level_min,
                geometric_noise_level_max=args.geometric_noise_level_max,
                duo_log_noise_ratio_min=args.duo_log_noise_ratio_min,
                duo_log_noise_ratio_max=args.duo_log_noise_ratio_max,
                duo_keep_prob_table=getattr(args, "_duo_keep_prob_table", None),
                duo_vocab_size=getattr(args, "_duo_vocab_size", None),
            )
        return sample_time(
            batch_count,
            args.objective,
            args.noise_schedule,
                args.continuous_time_eps,
                args.antithetic_sampling,
                batch_device,
                geometric_noise_level_min=args.geometric_noise_level_min,
                geometric_noise_level_max=args.geometric_noise_level_max,
            )

    def current_duo_inverse_temperature() -> float:
        return duo_input_inverse_temperature(
            getattr(args, "_current_step", 0),
            args.duo_softmax_temperature_log10_start,
            args.duo_softmax_temperature_log10_end,
            curriculum_start=args.duo_curriculum_start,
            curriculum_end=args.duo_curriculum_end,
        )

    def corrupt_batch(clean_tokens: torch.Tensor, time_batch: TimeBatch) -> torch.Tensor:
        if args.objective == "duo_uniform":
            return sample_forward_marginal(
                clean_tokens,
                time_batch,
                args.objective,
                args.noise_type,
                raw_model.mask_token_id,
                raw_model.config.vocab_size,
                args.duo_curriculum_mode,
                args.duo_top_k,
                current_duo_inverse_temperature(),
            )
        return sample_forward_marginal(
            clean_tokens,
            time_batch,
            args.objective,
            args.noise_type,
            raw_model.mask_token_id,
            raw_model.config.vocab_size,
        )

    def forward_noisy(noisy_tokens: torch.Tensor, time_batch: TimeBatch) -> torch.Tensor:
        return model_forward_noisy(model, noisy_tokens, time_batch, args.objective)

    def get_loss(
        logits: torch.Tensor,
        clean_tokens: torch.Tensor,
        noisy_tokens: torch.Tensor,
        time_batch: TimeBatch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_dlm_loss(
            logits,
            clean_tokens,
            noisy_tokens,
            time_batch,
            args.objective,
            args.noise_type,
            raw_model.mask_token_id,
            args.subs_mask_normalization,
            args.d3pm_vb_coeff or 0.0,
            args.d3pm_ce_coeff or 0.0,
        )

    print0("DLM trainer configuration:")
    for line in objective_summary_lines(args):
        print0(f"  {line}")
    print0(f"  model: {args.model} DDiT, sequence_length={args.sequence_length}, time_conditioning={args.time_conditioning}")
    print0(f"  optimizer: AdamW, lr={args.learning_rate:g}, warmup_iters={args.warmup_iters}, lr_schedule={args.lr_schedule}")
    print0(f"  dropout: {args.dropout:g}, cond_dim={args.cond_dim}, antithetic_sampling={args.antithetic_sampling}")
    print0(f"Parameters: total {num_parameters:,} | non-input-embedding {non_embedding_parameters:,}")

    print0("Warming up compiled training step...")
    model.train()
    args._current_step = 0
    warmup_time_batch = sample_training_time(x.size(0), x.device)
    warmup_noisy_x = corrupt_batch(x, warmup_time_batch)
    with ctx:
        warmup_logits = forward_noisy(warmup_noisy_x, warmup_time_batch)
        warmup_loss, _ = get_loss(warmup_logits, x, warmup_noisy_x, warmup_time_batch)
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    overall_t0 = time.perf_counter()

    def get_lr(it: int) -> float:
        assert it <= args.num_iterations
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        if args.lr_schedule == "constant":
            return args.learning_rate
        decay_step = max(1, it + 1)
        if args.lr_schedule == "inverse_sqrt":
            warmup = max(1, args.warmup_iters)
            return args.learning_rate * math.sqrt(warmup) / math.sqrt(decay_step)
        if args.lr_schedule == "cosine":
            decay_steps = max(1, args.num_iterations - args.warmup_iters)
            decay_progress = min(1.0, max(0, it - args.warmup_iters) / decay_steps)
            return 0.5 * args.learning_rate * (1.0 + math.cos(math.pi * decay_progress))
        raise ValueError(f"Unsupported lr_schedule={args.lr_schedule!r}")

    run_id = str(uuid.uuid4())
    logfile = None
    last_val_loss = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
        with open(logfile, "w"):
            pass
    checkpoint_paths = []
    checkpoint_records = []

    def record_path(path: str) -> str:
        checkpoint_dir = args.output_dir if args.output_dir else "logs"
        try:
            return os.path.relpath(path, checkpoint_dir)
        except ValueError:
            return os.path.basename(path)

    def checkpoint_args() -> dict:
        saved_args = {key: value for key, value in args.__dict__.items() if not key.startswith("_")}
        if not is_d3pm_objective(args.objective):
            saved_args.pop("num_diffusion_steps", None)
        return saved_args

    def save_checkpoint(tag: str, step: int, timed_elapsed: float, wall_elapsed: float) -> str:
        if args.skip_checkpoint_save:
            return ""
        checkpoint_dir = args.output_dir if args.output_dir else "logs"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_{tag}.pt")
        log = dict(
            model=checkpoint_model.state_dict(),
            ema_model=None,
            code=code,
            args={**checkpoint_args(), "recipe": args.objective},
            trainer="dlm",
            checkpoint_tag=tag,
            checkpoint_step=step,
            timed_training_seconds=timed_elapsed,
            total_wallclock_seconds=wall_elapsed,
            total_tokens_processed=tokens_per_step * step,
        )
        torch.save(log, checkpoint_path)
        metadata_path = record_path(checkpoint_path)
        checkpoint_paths.append(metadata_path)
        checkpoint_records.append(
            dict(
                path=metadata_path,
                tag=tag,
                step=step,
                timed_training_seconds=timed_elapsed,
                total_wallclock_seconds=wall_elapsed,
                total_tokens_processed=tokens_per_step * step,
            )
        )
        return checkpoint_path

    if args.checkpoint_steps:
        checkpoint_steps = {int(item) for item in args.checkpoint_steps.split(",") if item.strip()}
    else:
        checkpoint_steps = {int(round(k * args.num_iterations / args.num_checkpoints)) for k in range(1, args.num_checkpoints + 1)}
    checkpoint_steps = {step for step in checkpoint_steps if 0 < step <= args.num_iterations}

    timings = []
    timed_training_seconds = 0.0
    norm = -1.0
    lossf = None
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step) and val_loader is not None:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    clean_val = val_loader.next_batch()
                    time_batch = sample_training_time(clean_val.size(0), clean_val.device)
                    noisy_val = corrupt_batch(clean_val, time_batch)
                    with ctx:
                        logits = forward_noisy(noisy_val, time_batch)
                        loss, _ = get_loss(logits, clean_val, noisy_val, time_batch)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            last_val_loss = float(val_loss)
            print0(f"val loss {val_loss}")
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")

        if last_step:
            break

        t0 = time.time()
        model.train()
        args._current_step = step
        time_batch = sample_training_time(x.size(0), x.device)
        noisy_x = corrupt_batch(x, time_batch)
        with ctx:
            logits = forward_noisy(noisy_x, time_batch)
            loss, loss_terms = get_loss(logits, x, noisy_x, time_batch)
        x = train_loader.next_batch()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()
        lossf = loss.item()
        timed_training_seconds += t1 - t0
        tokens_per_second = ddp_world_size * batch_size * seq_len / (t1 - t0)
        metrics_text = " | ".join(f"{name} {value:.6f}" for name, value in sorted(loss_terms.items()))
        print0(
            f"step {step + 1:4d}/{args.num_iterations} | train loss {lossf:.6f} | {metrics_text} | "
            f"norm {norm:.4f} | lr {lr:.2e} | ({(t1 - t0) * 1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        )
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trl:{lossf}\n")
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)
        if master_process and (step + 1) in checkpoint_steps and (step + 1) != args.num_iterations:
            save_checkpoint(
                f"step{step + 1:06d}",
                step + 1,
                timed_training_seconds,
                time.perf_counter() - overall_t0,
            )

    timings = timings[-20:]
    avg_last_step_ms = float(np.mean(timings) * 1000) if timings else None
    if avg_last_step_ms is None:
        print0("final 0 iters avg: n/a")
    else:
        print0(f"final {len(timings)} iters avg: {avg_last_step_ms:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master_process:
        final_checkpoint_wallclock_seconds = time.perf_counter() - overall_t0
        checkpoint_path = save_checkpoint(
            "final",
            args.num_iterations,
            timed_training_seconds,
            final_checkpoint_wallclock_seconds,
        )
        total_wallclock_seconds = time.perf_counter() - overall_t0
        summary = dict(
            trainer="dlm",
            run_id=run_id,
            checkpoint_path=record_path(checkpoint_path) if checkpoint_path else "",
            checkpoint_paths=checkpoint_paths,
            checkpoint_records=checkpoint_records,
            timed_training_seconds=timed_training_seconds,
            total_wallclock_seconds=total_wallclock_seconds,
            avg_last_step_ms=avg_last_step_ms,
            final_train_loss=lossf,
            final_val_loss=last_val_loss,
            tokens_per_step=tokens_per_step,
            total_tokens_processed=tokens_per_step * args.num_iterations,
            num_iterations=args.num_iterations,
            model=args.model,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            total_batch_size=args.total_batch_size,
            train_files=args.train_files,
            val_files=args.val_files,
            peak_memory_mib=int(torch.cuda.max_memory_allocated() // 1024 // 1024),
            **objective_summary_fields(args),
            learning_rate=args.learning_rate,
            lr_schedule=args.lr_schedule,
            time_conditioning=args.time_conditioning,
            cond_dim=args.cond_dim,
            dropout=args.dropout,
            antithetic_sampling=args.antithetic_sampling,
            warmup_iters=args.warmup_iters,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
            num_parameters=int(num_parameters),
            non_embedding_parameters=int(non_embedding_parameters),
        )
        print0("TRAIN_RESULT " + json.dumps(summary, sort_keys=True))

    destroy_process_group()


if __name__ == "__main__":
    main()
