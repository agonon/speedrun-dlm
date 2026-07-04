#!/usr/bin/env python3
"""Print a JSON snapshot of the environment used for a benchmark run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGES = [
    "torch",
    "triton",
    "numpy",
    "transformers",
    "huggingface-hub",
    "datasets",
    "tiktoken",
    "safetensors",
    "tqdm",
]


def run_command(command: list[str], timeout: float = 10.0) -> dict[str, Any] | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"command": command, "error": repr(exc)}
    output = {"command": command, "returncode": result.returncode}
    if result.stdout.strip():
        output["stdout"] = result.stdout.strip()
    if result.stderr.strip():
        output["stderr"] = result.stderr.strip()
    return output


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"import_error": repr(exc)}

    info: dict[str, Any] = {
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": [props.major, props.minor],
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        info["cuda_devices"] = devices
    return info


def git_environment() -> dict[str, Any]:
    head = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "branch", "--show-current"])
    status = run_command(["git", "status", "--short"])
    return {
        "head": head.get("stdout") if head and head.get("returncode") == 0 else None,
        "branch": branch.get("stdout") if branch and branch.get("returncode") == 0 else None,
        "dirty": bool(status and status.get("stdout")),
    }


def nvidia_environment() -> dict[str, Any]:
    summary = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "gpu_summary": summary,
        "nvcc": run_command(["nvcc", "--version"]),
    }


def build_snapshot(package_names: list[str]) -> dict[str, Any]:
    return {
        "schema": "speedrun-dlm-environment-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(package_names),
        "torch": torch_environment(),
        "nvidia": nvidia_environment(),
        "git": git_environment(),
        "environment": {
            key: os.environ[key]
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_IB_DISABLE",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_CUDNN_V8_API_ENABLED",
            ]
            if key in os.environ
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print environment metadata as JSON.")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Extra Python package name to include. Can be repeated.",
    )
    args = parser.parse_args()
    package_names = sorted(set(DEFAULT_PACKAGES + args.package))
    print(json.dumps(build_snapshot(package_names), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
