import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint_utils import load_snapshot, resolve_amp_dtype
from .sample_text import (
    DLM_SAMPLER_CHOICES,
    checkpoint_objective_and_noise,
    resolve_dlm_sampler,
    sample_ar,
    sample_dlm,
)


def checkpoint_label(path: str) -> str:
    return Path(path).name or path


class ForwardCounter:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.original_forward = model.forward
        self.calls = 0

    def __enter__(self) -> "ForwardCounter":
        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return self.original_forward(*args, **kwargs)

        self.model.forward = wrapped_forward  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.model.forward = self.original_forward  # type: ignore[method-assign]


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_generator(device: str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(seed)
    return generator


def run_one_sample(
    snapshot,
    tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    amp_dtype: str,
    seed: int,
    dlm_sampler: str,
    num_sampling_steps: int,
    sampling_eps: float,
) -> list[int]:
    generator = make_generator(device, seed)
    if snapshot.trainer == "ar":
        return sample_ar(
            snapshot.model,
            [50256],
            tokens,
            temperature,
            top_k,
            device,
            amp_dtype,
            generator,
        )
    if snapshot.trainer == "dlm":
        return sample_dlm(
            snapshot.model,
            snapshot.args,
            tokens,
            temperature,
            top_k,
            device,
            amp_dtype,
            generator,
            dlm_sampler,
            num_sampling_steps,
            sampling_eps,
        )
    raise ValueError(f"Unsupported trainer: {snapshot.trainer!r}")


def profiler_activities(device: str) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_samples(
    snapshot,
    tokens: int,
    measured_samples: int,
    seed: int,
    temperature: float,
    top_k: int,
    device: str,
    amp_dtype: str,
    dlm_sampler: str,
    num_sampling_steps: int,
    sampling_eps: float,
) -> tuple[int, float, int]:
    with ForwardCounter(snapshot.model) as counter:
        cuda_sync(device)
        start = time.perf_counter()
        with torch.profiler.profile(
            activities=profiler_activities(device),
            with_flops=True,
            profile_memory=False,
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for offset in range(measured_samples):
                run_one_sample(
                    snapshot,
                    tokens=tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                    amp_dtype=amp_dtype,
                    seed=seed + offset,
                    dlm_sampler=dlm_sampler,
                    num_sampling_steps=num_sampling_steps,
                    sampling_eps=sampling_eps,
                )
        cuda_sync(device)
        elapsed = time.perf_counter() - start
    flops = int(sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages()))
    return flops, elapsed, counter.calls


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Inference Cost Measurement",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- checkpoint variant: `{payload['checkpoint_variant']}`",
        f"- parameters: `{payload['num_parameters']}`",
        f"- non-embedding parameters: `{payload['non_embedding_parameters']}`",
        f"- generated tokens per sample: `{payload['tokens_per_sample']}`",
        f"- measured samples: `{payload['measured_samples']}`",
        f"- warmup samples: `{payload['warmup_samples']}`",
        f"- temperature: `{payload['temperature']}`",
        f"- top_k: `{payload['top_k']}`",
        f"- DLM sampler: `{payload['dlm_sampler']}`",
        f"- DLM sampling steps: `{payload['dlm_sampling_steps']}`",
        f"- DLM sampling eps: `{payload['sampling_eps']}`",
        f"- forward calls per sample: `{payload['forward_calls_per_sample']:.6g}`",
        f"- profiler FLOPs per sample: `{payload['profiler_flops_per_sample']:.6g}`",
        f"- profiler TFLOPs per sample: `{payload['profiler_tflops_per_sample']:.6g}`",
        f"- profiler FLOPs per 128-sample gate: `{payload['profiler_flops_per_128_sample_gate']:.6g}`",
        f"- profiler FLOPs per 1024 samples: `{payload['profiler_flops_per_1024_samples']:.6g}`",
        f"- wall seconds per sample: `{payload['wall_seconds_per_sample']:.6g}`",
        f"- CUDA max memory MiB: `{payload['cuda_max_memory_mib']:.1f}`",
        "",
        "## Notes",
        "",
        "- FLOPs are estimated by `torch.profiler.profile(with_flops=True)` on the actual sampler path.",
        "- The profiler only counts operators for which PyTorch reports FLOPs; keep the JSON artifact with every record.",
        "- Warmup samples run before profiling and are not included in the reported FLOPs or wall time.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure sampler inference cost for a checkpoint.")
    parser.add_argument("snapshot_path")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--measured_samples", type=int, default=1)
    parser.add_argument("--warmup_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_variant", choices=("auto", "model", "ema_model"), default="model")
    parser.add_argument("--amp_dtype", choices=("auto", "none", "bfloat16"), default="auto")
    parser.add_argument("--dlm_sampler", choices=DLM_SAMPLER_CHOICES, default="auto")
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--sampling_eps", type=float, default=1e-5)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    device = args.device
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    snapshot = load_snapshot(args.snapshot_path, device=device, checkpoint_variant=args.checkpoint_variant)
    reported_dlm_sampler = ""
    if snapshot.trainer == "dlm":
        objective, _ = checkpoint_objective_and_noise(snapshot.args)
        reported_dlm_sampler = resolve_dlm_sampler(objective, args.dlm_sampler)
    num_parameters = sum(p.numel() for p in snapshot.model.parameters())
    non_embedding_parameters = sum(
        p.numel() for name, p in snapshot.model.named_parameters() if "transformer.wte" not in name
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for offset in range(args.warmup_samples):
        run_one_sample(
            snapshot,
            tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            amp_dtype=amp_dtype,
            seed=args.seed - args.warmup_samples + offset,
            dlm_sampler=args.dlm_sampler,
            num_sampling_steps=args.num_sampling_steps,
            sampling_eps=args.sampling_eps,
        )
    cuda_sync(device)

    flops, elapsed, forward_calls = profile_samples(
        snapshot,
        tokens=args.tokens,
        measured_samples=args.measured_samples,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        amp_dtype=amp_dtype,
        dlm_sampler=args.dlm_sampler,
        num_sampling_steps=args.num_sampling_steps,
        sampling_eps=args.sampling_eps,
    )

    per_sample_flops = flops / max(args.measured_samples, 1)
    per_sample_calls = forward_calls / max(args.measured_samples, 1)
    payload: dict[str, Any] = {
        "snapshot": checkpoint_label(args.snapshot_path),
        "trainer": snapshot.trainer,
        "checkpoint_variant": snapshot.checkpoint_variant,
        "num_parameters": int(num_parameters),
        "non_embedding_parameters": int(non_embedding_parameters),
        "device": device,
        "amp_dtype": amp_dtype,
        "tokens_per_sample": args.tokens,
        "measured_samples": args.measured_samples,
        "warmup_samples": args.warmup_samples,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "dlm_sampler": reported_dlm_sampler,
        "dlm_sampling_steps": args.num_sampling_steps if snapshot.trainer == "dlm" else 0,
        "sampling_eps": args.sampling_eps if snapshot.trainer == "dlm" else 0.0,
        "forward_calls_total": forward_calls,
        "forward_calls_per_sample": per_sample_calls,
        "profiler_flops_total": flops,
        "profiler_flops_per_sample": per_sample_flops,
        "profiler_flops_per_1024_token_sample": per_sample_flops if args.tokens == 1024 else None,
        "profiler_tflops_per_sample": per_sample_flops / 1e12,
        "profiler_tflops_per_1024_token_sample": per_sample_flops / 1e12 if args.tokens == 1024 else None,
        "profiler_flops_per_128_sample_gate": per_sample_flops * 128,
        "profiler_flops_per_1024_samples": per_sample_flops * 1024,
        "wall_seconds_total": elapsed,
        "wall_seconds_per_sample": elapsed / max(args.measured_samples, 1),
        "cuda_max_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2) if device.startswith("cuda") else 0.0
        ),
        "profiler": "torch.profiler.profile(with_flops=True)",
        "cost_accounting_version": "v0.1-torch-profiler-actual-sampler",
        "profiler_caveat": "PyTorch reports FLOPs only for supported operators; keep this with sampler config.",
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {output_json}")
    else:
        print(json.dumps(payload, indent=2))

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(payload, output_md)
        print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
