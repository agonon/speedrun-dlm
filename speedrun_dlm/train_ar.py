import argparse
import glob
import json
import math
import os
import time
import uuid
from dataclasses import dataclass

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


def rmsnorm(x0: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # no kv cache here this is the simple baseline path
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.c_proj(y)
        y = y / math.sqrt(24)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # not initialized, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) and not hasattr(module, "LLMC_SKIP_INIT"):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # for position embeddings, std=0.02 to match scale of token embeddings

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_logits: bool = True):
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])

        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)


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
            assert shard_ntok >= num_processes * batch_size * seq_len + 1
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

    def next_batch(self):
        batch_size = self.batch_size
        seq_len = self.seq_len
        buf = self.tokens[self.current_position : self.current_position + batch_size * seq_len + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(batch_size, seq_len)
        y = buf[1:].view(batch_size, seq_len)
        self.current_position += batch_size * seq_len * self.num_processes
        if self.current_position + (batch_size * seq_len * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def get_model_config(model_name: str) -> GPTConfig:
    return {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[model_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val_files", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model", choices=("d12", "d24", "d36", "d48"), default="d12")
    # AR keeps a smaller default batch than DLM because validation loss degrades at the same token budget.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--total_batch_size", type=int, default=32 * 1024 * 8)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_loss_every", type=int, default=0)
    parser.add_argument("--val_max_steps", type=int, default=20)
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=1,
        help=(
            "Total number of checkpoints to save over the run, including the final one. "
            "Checkpoints are spaced uniformly in training steps. "
            "Current checkpoints store fp32 model weights only: d12 is ~474 MiB each, "
            "d24 is ~1.32 GiB, d36 is ~2.88 GiB, d48 is ~5.80 GiB."
        ),
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=str,
        default="",
        help=(
            "Optional comma-separated optimizer step numbers to checkpoint. "
            "When set, this overrides --num_checkpoints for intermediate checkpoints. "
            "The final checkpoint is always saved after training."
        ),
    )
    args = parser.parse_args()

    batch_size, seq_len = args.batch_size, args.sequence_length
    assert 1 <= seq_len <= 1024
    assert args.model in {"d12", "d24", "d36", "d48"}
    assert args.num_checkpoints >= 1
    assert torch.cuda.is_available(), "CUDA is required for this trainer"

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
    model = GPT(get_model_config(args.model)).train().cuda()
    num_parameters = sum(p.numel() for p in model.parameters())
    non_embedding_parameters = sum(p.numel() for name, p in model.named_parameters() if "transformer.wte" not in name)
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
    x, y = train_loader.next_batch()

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )
    print0(f"Parameters: total {num_parameters:,} | non-token-embedding {non_embedding_parameters:,}")
    print0("Warming up compiled training step...")
    model.train()
    with ctx:
        _, warmup_loss = model(x, y, return_logits=False)
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    overall_t0 = time.perf_counter()

    def get_lr(it: int) -> float:
        assert it <= args.num_iterations
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        return (0.1 + (1 - decay_ratio)) / 1.1 * args.learning_rate

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

    def checkpoint_state_dict() -> dict[str, torch.Tensor]:
        checkpoint_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        return checkpoint_model.state_dict()

    def save_checkpoint(tag: str, step: int, timed_elapsed: float, wall_elapsed: float) -> str:
        checkpoint_dir = args.output_dir if args.output_dir else "logs"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_{tag}.pt")
        log = dict(
            model=checkpoint_state_dict(),
            code=code,
            args=args.__dict__,
            trainer="ar",
            checkpoint_tag=tag,
            checkpoint_step=step,
            timed_training_seconds=timed_elapsed,
            total_wallclock_seconds=wall_elapsed,
            total_tokens_processed=tokens_per_step * step,
            num_parameters=int(num_parameters),
            non_embedding_parameters=int(non_embedding_parameters),
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

        # Validate at step 0 and once more after the final update.
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step) and val_loader is not None:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            last_val_loss = float(val_loss)
            print0(f"val loss {val_loss}")
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"s:{step} tel:{val_loss}\n")

        if last_step:
            break

        # training section
        t0 = time.time()
        model.train()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        x, y = train_loader.next_batch()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # end of training section, rest is logging etc.

        torch.cuda.synchronize()
        t1 = time.time()
        lossf = loss.item()
        timed_training_seconds += t1 - t0
        tokens_per_second = ddp_world_size * batch_size * seq_len / (t1 - t0)
        print0(
            f"step {step + 1:4d}/{args.num_iterations} | train loss {lossf:.6f} | "
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

    timings = timings[-20:] # print avg of last 20 steps to smooth it out
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
            trainer="ar",
            run_id=run_id,
            checkpoint_path=record_path(checkpoint_path),
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
            peak_memory_mib=int(torch.cuda.max_memory_allocated() // 1024 // 1024),
            seed=args.seed,
            num_parameters=int(num_parameters),
            non_embedding_parameters=int(non_embedding_parameters),
        )
        print0("TRAIN_RESULT " + json.dumps(summary, sort_keys=True))

    destroy_process_group()


if __name__ == "__main__":
    main()
