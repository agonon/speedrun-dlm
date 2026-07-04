"""Internal scoring implementation for the generation-quality gate.

The public entry point is python -m speedrun_dlm.score_generation_quality.
This module scores an already-generated panel and writes per-sample metrics.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
DOC_COHERENCE_KINDS = (
    "shuffle",
    "local_swap",
    "sentence_replacement",
    "block_replacement",
    "tail_splice",
)


def timing(stage: str, seconds: float, **fields: Any) -> None:
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"TIMING quality_metrics.{stage} seconds={seconds:.3f}{suffix}", flush=True)


def ngram_repeat_ratio(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def max_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    cur = 1
    prev = tokens[0]
    for token in tokens[1:]:
        if token == prev:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
            prev = token
    return max(best, cur)


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
    return [sentence for sentence in sentences if sentence]


def sentence_token_count(sentence: str) -> int:
    return len(TOKEN_RE.findall(sentence))


def deterministic_metrics(text: str) -> dict[str, float]:
    raw = text.encode("utf-8", errors="replace")
    tokens = TOKEN_RE.findall(text)
    lower_tokens = [token.lower() for token in tokens]
    chars = len(text)
    bytes_len = len(raw)
    compressed = len(zlib.compress(raw)) if raw else 0
    line_count = text.count("\n") + (1 if text else 0)
    eot_count = text.count("<|endoftext|>")
    mask_count = text.count("<|mask|>") + text.count("[MASK]") + text.count("<mask>")
    bad_char_count = text.count("\ufffd") + len(CONTROL_RE.findall(text))

    return {
        "char_count": float(chars),
        "byte_count": float(bytes_len),
        "whitespace_token_count": float(len(tokens)),
        "line_count": float(line_count),
        "unique_token_ratio": (len(set(lower_tokens)) / len(lower_tokens)) if lower_tokens else 0.0,
        "repeat_bigram_ratio": ngram_repeat_ratio(lower_tokens, 2),
        "repeat_trigram_ratio": ngram_repeat_ratio(lower_tokens, 3),
        "repeat_4gram_ratio": ngram_repeat_ratio(lower_tokens, 4),
        "max_token_run": float(max_run(lower_tokens)),
        "compression_ratio": (compressed / bytes_len) if bytes_len else 0.0,
        "eot_count": float(eot_count),
        "mask_token_count": float(mask_count),
        "bad_char_count": float(bad_char_count),
    }

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def finite(value: float | None) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16)


def load_panel(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("version") != 1:
        raise ValueError(f"Unsupported panel version in {path}")
    return data


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    raise ValueError("Sample JSON must be either a list or a dict with a 'samples' list.")


def normalize_sample(panel: dict[str, Any], item: dict[str, Any], index: int) -> dict[str, str]:
    prompt = str(item.get("prompt") or "")
    continuation = item.get("continuation")
    generated = item.get("generated_text")
    full_text = item.get("full_text")

    if continuation is None:
        if generated is not None:
            continuation = generated
        elif full_text is not None and prompt and str(full_text).startswith(prompt):
            continuation = str(full_text)[len(prompt) :]
        elif full_text is not None:
            continuation = full_text
        else:
            continuation = ""

    if full_text is None:
        if generated is not None:
            full_text = generated
        else:
            full_text = f"{prompt}{continuation}"

    protocol = str(panel.get("protocol") or item.get("sample_mode") or item.get("mode") or "unknown")
    if protocol == "unconditional":
        score_prompt = ""
        score_target = str(generated if generated is not None else full_text)
    else:
        score_prompt = prompt
        score_target = str(continuation)

    return {
        "panel": str(panel["name"]),
        "family": str(panel.get("family", "")),
        "protocol": protocol,
        "gate_label": str(panel.get("gate_label", "")),
        "subjective_tier": "" if panel.get("subjective_tier") is None else str(panel["subjective_tier"]),
        "sample_index_global": str(index),
        "prompt_index": str(item.get("prompt_index", "")),
        "sample_index": str(item.get("sample_index", "")),
        "seed": str(item.get("seed", "")),
        "prompt": prompt,
        "target_text": score_target,
        "full_text": str(full_text),
    }


def make_sample(
    panel: str,
    family: str,
    protocol: str,
    gate_label: str,
    subjective_tier: float | None,
    index: int,
    text: str,
    prompt: str = "",
) -> dict[str, str]:
    return {
        "panel": panel,
        "family": family,
        "protocol": protocol,
        "gate_label": gate_label,
        "subjective_tier": "" if subjective_tier is None else str(subjective_tier),
        "sample_index_global": str(index),
        "prompt_index": "control",
        "sample_index": str(index),
        "seed": "control",
        "prompt": prompt,
        "target_text": text,
        "full_text": f"{prompt}{text}",
    }


def window_text(text: str, max_chars: int = 4096) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars // 2:
        return cut[: last_period + 1]
    return cut


def load_fineweb_controls(
    count: int,
    seed: int,
    dataset_name: str,
    dataset_config: str,
    pinned_fineweb: str,
    allow_fallback: bool,
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split="train",
            streaming=True,
            revision=pinned_fineweb or None,
        )
        controls: list[dict[str, str]] = []
        # `skip` gives us a deterministic, cheap offset away from the first rows.
        for item in dataset.skip(seed % 1000):
            text = window_text(str(item.get("text") or ""))
            if len(text.split()) < 80:
                continue
            controls.append(
                make_sample(
                    panel="control_fineweb_real_text",
                    family="control_real_text",
                    protocol="real_text",
                    gate_label="reference_pass",
                    subjective_tier=5.0,
                    index=len(controls),
                    text=text,
                )
            )
            if len(controls) >= count:
                return controls
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Could not load FineWeb controls. Set --allow_fineweb_fallback only for local smoke tests."
            ) from exc

    fallback = [
        "The city council released the annual transport report after months of public consultation. "
        "The document compares ridership, maintenance costs, and accident reports across several "
        "neighborhoods, and recommends adding bus lanes on corridors where average speeds have "
        "fallen during the morning commute. Officials said the proposal would be reviewed again "
        "after the summer construction season.",
        "Researchers studying coastal wetlands described a gradual recovery in several marshes "
        "after restoration crews reopened tidal channels and removed invasive plants. The report "
        "notes that bird counts remain below historical levels, but water quality and juvenile "
        "fish surveys improved for the third consecutive year.",
    ]
    return [
        make_sample(
            panel="control_fallback_real_text",
            family="control_real_text",
            protocol="real_text",
            gate_label="reference_pass",
            subjective_tier=5.0,
            index=i,
            text=window_text(text),
        )
        for i, text in enumerate((fallback * ((count // len(fallback)) + 1))[:count])
    ]

class ReferenceLMScorer:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        max_eval_tokens: int,
        score_batch_size: int,
        trust_remote_code: bool,
        device_map: str,
        pinned_reference_model: str = "",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_eval_tokens = max_eval_tokens
        self.score_batch_size = max(1, score_batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=pinned_reference_model or None,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if pinned_reference_model:
            load_kwargs["revision"] = pinned_reference_model
        if device_map:
            load_kwargs["device_map"] = device_map
            load_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not device_map:
            self.model.to(device)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        model_context_candidates = [
            getattr(self.model.config, "max_position_embeddings", None),
            getattr(self.model.config, "n_positions", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        model_contexts = [
            int(value)
            for value in model_context_candidates
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        if model_contexts:
            self.max_eval_tokens = min(self.max_eval_tokens, min(model_contexts))
            print(
                f"Reference LM context cap for {model_name}: "
                f"using max_eval_tokens={self.max_eval_tokens}",
                flush=True,
            )

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def score_targets_batch(
        self,
        contexts: list[str],
        targets: list[str],
        prepend_bos_if_empty: bool,
    ) -> list[dict[str, float]]:
        torch = self.torch
        if len(contexts) != len(targets):
            raise ValueError("contexts and targets must have the same length")
        encoded_contexts = [self.encode(context) if context else [] for context in contexts]
        encoded_targets = [self.encode(target) for target in targets]
        bos_id = self.tokenizer.eos_token_id

        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        target_bytes: list[int] = []
        target_counts: list[int] = []
        for context_ids, target_ids, target in zip(encoded_contexts, encoded_targets, targets, strict=True):
            context_ids = list(context_ids)
            target_ids = list(target_ids)
            if prepend_bos_if_empty and not context_ids and bos_id is not None:
                context_ids = [int(bos_id)]
            if len(target_ids) < 1:
                sequences.append(context_ids or ([int(bos_id)] if bos_id is not None else [0]))
                labels.append([-100] * len(sequences[-1]))
                target_bytes.append(max(1, len(target.encode("utf-8", errors="replace"))))
                target_counts.append(0)
                continue

            max_len = self.max_eval_tokens
            # Preserve the full target sentence whenever possible, trimming only
            # the left side of the context. Sentence-order contrast is local, so
            # losing very old context is less harmful than truncating candidates.
            if len(context_ids) + len(target_ids) > max_len:
                room_for_context = max(1, max_len - len(target_ids))
                context_ids = context_ids[-room_for_context:]
            if len(context_ids) + len(target_ids) > max_len:
                target_ids = target_ids[: max(1, max_len - len(context_ids))]

            input_ids = context_ids + target_ids
            label_ids = [-100] * len(context_ids) + target_ids
            if len(input_ids) < 2:
                if bos_id is not None:
                    input_ids = [int(bos_id)] + input_ids
                    label_ids = [-100] + label_ids
                else:
                    input_ids = input_ids + [input_ids[-1] if input_ids else 0]
                    label_ids = label_ids + [-100]
            sequences.append(input_ids)
            labels.append(label_ids)
            target_bytes.append(max(1, len(target.encode("utf-8", errors="replace"))))
            target_counts.append(len(target_ids))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        nlls: list[float] = []
        scored_tokens: list[int] = []
        for start in range(0, len(sequences), self.score_batch_size):
            batch_sequences = sequences[start : start + self.score_batch_size]
            batch_labels = labels[start : start + self.score_batch_size]
            max_batch_len = max(len(seq) for seq in batch_sequences)
            input_batch = []
            label_batch = []
            attention_batch = []
            for seq, label in zip(batch_sequences, batch_labels, strict=True):
                pad = max_batch_len - len(seq)
                input_batch.append(seq + [int(pad_id)] * pad)
                label_batch.append(label + [-100] * pad)
                attention_batch.append([1] * len(seq) + [0] * pad)

            ids_tensor = torch.tensor(input_batch, device=self.input_device)
            labels_tensor = torch.tensor(label_batch, device=self.input_device)
            attention_tensor = torch.tensor(attention_batch, device=self.input_device)
            with torch.no_grad():
                logits = self.model(ids_tensor, attention_mask=attention_tensor).logits[:, :-1, :]
                shifted_labels = labels_tensor[:, 1:]
                keep = shifted_labels != -100
                safe_labels = shifted_labels.masked_fill(~keep, 0)
                losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    safe_labels.reshape(-1),
                    reduction="none",
                ).reshape_as(safe_labels)
                losses = losses.masked_fill(~keep, 0.0)
                nlls.extend(losses.sum(dim=1).detach().cpu().tolist())
                scored_tokens.extend(keep.sum(dim=1).detach().cpu().tolist())

        out: list[dict[str, float]] = []
        for nll, scored, total_target_tokens, n_bytes in zip(
            nlls,
            scored_tokens,
            target_counts,
            target_bytes,
            strict=True,
        ):
            scored_i = int(scored)
            nll_f = float(nll)
            if scored_i == 0:
                bpb = float("nan")
                nll_per_token = float("nan")
            else:
                bpb = nll_f / (n_bytes * math.log(2))
                nll_per_token = nll_f / scored_i
            out.append(
                {
                    "target_token_count": float(total_target_tokens),
                    "scored_token_count": float(scored_i),
                    "nll": nll_f,
                    "nll_per_token": nll_per_token,
                    "bpb": bpb,
                }
            )
        return out

    def score(self, prompt: str, target: str) -> dict[str, float]:
        # Full-sample BPB uses sliding windows; the batched helper above is for short contrast candidates.
        torch = self.torch
        prompt_ids = self.encode(prompt) if prompt else []
        target_ids = self.encode(target)
        if len(target_ids) < 1:
            return {
                "ref_token_count": 0.0,
                "ref_scored_token_count": 0.0,
                "ref_nll": float("nan"),
                "ref_nll_per_token": float("nan"),
                "ref_ppl": float("nan"),
                "ref_bpb": float("nan"),
            }

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) < 2:
            return {
                "ref_token_count": float(len(input_ids)),
                "ref_scored_token_count": 0.0,
                "ref_nll": float("nan"),
                "ref_nll_per_token": float("nan"),
                "ref_ppl": float("nan"),
                "ref_bpb": float("nan"),
            }

        max_len = self.max_eval_tokens
        total_nll = 0.0
        total_tokens = 0
        start = 0
        while start < len(input_ids) - 1:
            end = min(len(input_ids), start + max_len)
            chunk_ids = input_ids[start:end]
            chunk_labels = labels[start:end]
            if start > 0 and chunk_labels:
                chunk_labels[0] = -100
            ids_tensor = torch.tensor([chunk_ids], device=self.input_device)
            labels_tensor = torch.tensor(chunk_labels, device=self.input_device)
            with torch.no_grad():
                logits = self.model(ids_tensor).logits[0, :-1, :]
                shifted_labels = labels_tensor[1:]
                keep = shifted_labels != -100
                if keep.any():
                    losses = torch.nn.functional.cross_entropy(
                        logits[keep],
                        shifted_labels[keep],
                        reduction="none",
                    )
                    total_nll += float(losses.sum().detach().cpu())
                    total_tokens += int(keep.sum().detach().cpu())
            if end == len(input_ids):
                break
            start = end - 1

        if total_tokens == 0:
            nll_per_token = float("nan")
            ppl = float("nan")
            bpb = float("nan")
        else:
            nll_per_token = total_nll / total_tokens
            ppl = math.exp(min(80.0, nll_per_token))
            target_bytes = max(1, len(target.encode("utf-8", errors="replace")))
            bpb = total_nll / (target_bytes * math.log(2))

        return {
            "ref_token_count": float(len(input_ids)),
            "ref_scored_token_count": float(total_tokens),
            "ref_nll": total_nll,
            "ref_nll_per_token": nll_per_token,
            "ref_ppl": ppl,
            "ref_bpb": bpb,
        }

    def score_order_contrast(
        self,
        sample: dict[str, str],
        sample_index: int,
        sentence_cache: list[list[str]],
        candidate_pool: list[tuple[int, str]],
        pairs_per_sample: int,
        distractors: int,
        context_sentences: int,
        min_sentence_tokens: int,
        max_sentence_tokens: int,
        seed: int,
    ) -> dict[str, float]:
        sentences = sentence_cache[sample_index]
        if pairs_per_sample <= 0 or len(sentences) < 2:
            return empty_order_metrics()

        def valid(sentence: str) -> bool:
            n_tokens = sentence_token_count(sentence)
            return min_sentence_tokens <= n_tokens <= max_sentence_tokens

        pair_indices = [
            i
            for i in range(len(sentences) - 1)
            if valid(sentences[i]) and valid(sentences[i + 1])
        ]
        if not pair_indices:
            return empty_order_metrics(sentence_count=len(sentences))
        rng = random.Random(seed + stable_int(f"{sample['panel']}::{sample['sample_index_global']}::{sample['seed']}"))
        rng.shuffle(pair_indices)
        pair_indices = pair_indices[:pairs_per_sample]

        external_pool = [(idx, sent) for idx, sent in candidate_pool if idx != sample_index and valid(sent)]
        same_doc_valid = [j for j, sent in enumerate(sentences) if valid(sent)]

        cross_contexts: list[str] = []
        cross_targets: list[str] = []
        cross_slices: list[tuple[int, int]] = []
        shuffle_contexts: list[str] = []
        shuffle_targets: list[str] = []
        shuffle_slices: list[tuple[int, int]] = []

        for pair_index in pair_indices:
            context = " ".join(sentences[max(0, pair_index + 1 - context_sentences) : pair_index + 1])
            actual = sentences[pair_index + 1]
            external_candidates: list[str] = []
            if external_pool:
                pool = list(external_pool)
                rng.shuffle(pool)
                external_candidates = [sent for _, sent in pool[:distractors]]
            candidates = [actual] + external_candidates
            if len(candidates) > 1:
                start = len(cross_targets)
                cross_contexts.extend([context] * len(candidates))
                cross_targets.extend(candidates)
                cross_slices.append((start, len(cross_targets)))

            shuffled_options = [
                j
                for j in same_doc_valid
                if j not in {pair_index, pair_index + 1}
                and abs(j - pair_index) > 1
            ]
            if shuffled_options:
                shuffled_target = sentences[rng.choice(shuffled_options)]
                candidates = [actual, shuffled_target]
                start = len(shuffle_targets)
                shuffle_contexts.extend([context, context])
                shuffle_targets.extend(candidates)
                shuffle_slices.append((start, len(shuffle_targets)))

        cross_wins = 0
        cross_margins: list[float] = []
        cross_actual_gains: list[float] = []
        cross_distractor_gains: list[float] = []
        if cross_targets:
            conditional = self.score_targets_batch(
                cross_contexts,
                cross_targets,
                prepend_bos_if_empty=True,
            )
            standalone = self.score_targets_batch(
                [""] * len(cross_targets),
                cross_targets,
                prepend_bos_if_empty=True,
            )
            gains = [
                standalone_row["bpb"] - conditional_row["bpb"]
                for conditional_row, standalone_row in zip(conditional, standalone, strict=True)
            ]
            for start, end in cross_slices:
                pair_gains = gains[start:end]
                actual_gain = pair_gains[0]
                distractor_gains = [gain for gain in pair_gains[1:] if finite(gain)]
                if finite(actual_gain) and distractor_gains:
                    best_distractor = max(distractor_gains)
                    cross_actual_gains.append(actual_gain)
                    cross_distractor_gains.append(mean(distractor_gains))
                    margin = actual_gain - best_distractor
                    cross_margins.append(margin)
                    if margin > 0:
                        cross_wins += 1

        shuffle_wins = 0
        shuffle_margins: list[float] = []
        shuffle_actual_gains: list[float] = []
        shuffle_distractor_gains: list[float] = []
        if shuffle_targets:
            conditional = self.score_targets_batch(
                shuffle_contexts,
                shuffle_targets,
                prepend_bos_if_empty=True,
            )
            standalone = self.score_targets_batch(
                [""] * len(shuffle_targets),
                shuffle_targets,
                prepend_bos_if_empty=True,
            )
            gains = [
                standalone_row["bpb"] - conditional_row["bpb"]
                for conditional_row, standalone_row in zip(conditional, standalone, strict=True)
            ]
            for start, end in shuffle_slices:
                pair_gains = gains[start:end]
                if len(pair_gains) != 2:
                    continue
                actual_gain, shuffled_gain = pair_gains
                if finite(actual_gain) and finite(shuffled_gain):
                    shuffle_actual_gains.append(actual_gain)
                    shuffle_distractor_gains.append(shuffled_gain)
                    margin = actual_gain - shuffled_gain
                    shuffle_margins.append(margin)
                    if margin > 0:
                        shuffle_wins += 1

        cross_pairs = len(cross_margins)
        shuffle_pairs = len(shuffle_margins)
        return {
            "order_sentence_count": float(len(sentences)),
            "order_pair_count": float(cross_pairs),
            "order_contrast_acc": (cross_wins / cross_pairs) if cross_pairs else float("nan"),
            "order_contrast_margin_bpb_mean": mean(cross_margins) if cross_margins else float("nan"),
            "order_contrast_margin_bpb_median": median(cross_margins) if cross_margins else float("nan"),
            "order_context_gain_bpb_mean": mean(cross_actual_gains) if cross_actual_gains else float("nan"),
            "order_distractor_gain_bpb_mean": mean(cross_distractor_gains) if cross_distractor_gains else float("nan"),
            "order_shuffle_pair_count": float(shuffle_pairs),
            "order_shuffle_acc": (shuffle_wins / shuffle_pairs) if shuffle_pairs else float("nan"),
            "order_shuffle_margin_bpb_mean": mean(shuffle_margins) if shuffle_margins else float("nan"),
            "order_shuffle_margin_bpb_median": median(shuffle_margins) if shuffle_margins else float("nan"),
            "order_shuffle_actual_gain_bpb_mean": mean(shuffle_actual_gains) if shuffle_actual_gains else float("nan"),
            "order_shuffle_distractor_gain_bpb_mean": mean(shuffle_distractor_gains)
            if shuffle_distractor_gains
            else float("nan"),
        }

    def score_document_coherence(
        self,
        sample: dict[str, str],
        sample_index: int,
        sentence_cache: list[list[str]],
        candidate_docs: list[tuple[int, list[str]]],
        min_sentences: int,
        max_sentences: int,
        max_sentence_tokens: int,
        shuffles: int,
        swaps: int,
        replacements: int,
        block_replacements: int,
        tail_splices: int,
        replacement_sentences: int,
        block_sentences: int,
        seed: int,
    ) -> dict[str, float]:
        sentences = sentence_cache[sample_index]

        def valid(sentence: str) -> bool:
            n_tokens = sentence_token_count(sentence)
            return 1 <= n_tokens <= max_sentence_tokens

        valid_sentences = [sentence for sentence in sentences if valid(sentence)]
        if len(valid_sentences) < min_sentences:
            return empty_document_coherence_metrics(sentence_count=len(sentences))

        rng = random.Random(seed + stable_int(f"{sample['panel']}::{sample['sample_index_global']}::{sample['seed']}"))
        if len(valid_sentences) > max_sentences:
            # Use a deterministic contiguous window so corruptions preserve local topic and style.
            start = rng.randrange(0, len(valid_sentences) - max_sentences + 1)
            doc_sentences = valid_sentences[start : start + max_sentences]
        else:
            doc_sentences = valid_sentences

        def join_doc(items: list[str]) -> str:
            return " ".join(items)

        corruptions: list[tuple[str, str]] = []
        for _ in range(max(0, shuffles)):
            shuffled = list(doc_sentences)
            rng.shuffle(shuffled)
            if shuffled != doc_sentences:
                corruptions.append(("shuffle", join_doc(shuffled)))

        for _ in range(max(0, swaps)):
            swapped = list(doc_sentences)
            if len(swapped) >= 2:
                n_swaps = max(1, len(swapped) // 4)
                used: set[int] = set()
                for _ in range(n_swaps):
                    candidates = [i for i in range(len(swapped) - 1) if i not in used and i + 1 not in used]
                    if not candidates:
                        break
                    i = rng.choice(candidates)
                    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                    used.add(i)
                    used.add(i + 1)
                if swapped != doc_sentences:
                    corruptions.append(("local_swap", join_doc(swapped)))

        external_docs = [
            (idx, [sentence for sentence in sentences_ if valid(sentence)])
            for idx, sentences_ in candidate_docs
            if idx != sample_index
        ]
        external_docs = [(idx, sentences_) for idx, sentences_ in external_docs if sentences_]
        external_pool = [
            (idx, sentence)
            for idx, sentences_ in external_docs
            for sentence in sentences_
        ]
        for _ in range(max(0, replacements)):
            if not external_pool:
                break
            replaced = list(doc_sentences)
            positions = list(range(len(replaced)))
            rng.shuffle(positions)
            pool = list(external_pool)
            rng.shuffle(pool)
            for pos, (_, sentence) in zip(positions[: max(1, replacement_sentences)], pool, strict=False):
                replaced[pos] = sentence
            if replaced != doc_sentences:
                corruptions.append(("sentence_replacement", join_doc(replaced)))

        for _ in range(max(0, block_replacements)):
            donor_docs = [
                donor_sentences
                for _, donor_sentences in external_docs
                if donor_sentences and len(doc_sentences) >= 2
            ]
            if not donor_docs:
                break
            donor = rng.choice(donor_docs)
            block_len = min(max(1, block_sentences), len(doc_sentences), len(donor))
            start = rng.randrange(0, len(doc_sentences) - block_len + 1)
            donor_start = rng.randrange(0, len(donor) - block_len + 1)
            replaced = list(doc_sentences)
            replaced[start : start + block_len] = donor[donor_start : donor_start + block_len]
            if replaced != doc_sentences:
                corruptions.append(("block_replacement", join_doc(replaced)))

        for _ in range(max(0, tail_splices)):
            if len(doc_sentences) < 4:
                break
            min_tail = max(1, len(doc_sentences) // 3)
            max_tail = max(min_tail, (2 * len(doc_sentences)) // 3)
            cut = rng.randrange(len(doc_sentences) - max_tail, len(doc_sentences) - min_tail + 1)
            tail_len = len(doc_sentences) - cut
            donor_docs = [donor_sentences for _, donor_sentences in external_docs if len(donor_sentences) >= tail_len]
            if not donor_docs:
                break
            donor = rng.choice(donor_docs)
            donor_start = rng.randrange(0, len(donor) - tail_len + 1)
            spliced = doc_sentences[:cut] + donor[donor_start : donor_start + tail_len]
            if spliced != doc_sentences:
                corruptions.append(("tail_splice", join_doc(spliced)))

        if not corruptions:
            return empty_document_coherence_metrics(sentence_count=len(sentences))

        original_doc = join_doc(doc_sentences)
        texts = [original_doc] + [text for _, text in corruptions]
        scores = self.score_targets_batch(
            [""] * len(texts),
            texts,
            prepend_bos_if_empty=True,
        )
        original_bpb = scores[0]["bpb"]
        margins_by_kind: dict[str, list[float]] = {}
        wins_by_kind: dict[str, int] = {}
        counts_by_kind: dict[str, int] = {}
        margins: list[float] = []
        wins = 0
        for (kind, _), score in zip(corruptions, scores[1:], strict=True):
            corrupt_bpb = score["bpb"]
            if not finite(original_bpb) or not finite(corrupt_bpb):
                continue
            margin = corrupt_bpb - original_bpb
            margins.append(margin)
            margins_by_kind.setdefault(kind, []).append(margin)
            counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
            if margin > 0:
                wins += 1
                wins_by_kind[kind] = wins_by_kind.get(kind, 0) + 1

        count = len(margins)
        out = {
            "doc_coherence_sentence_count": float(len(doc_sentences)),
            "doc_coherence_corruption_count": float(count),
            "doc_coherence_original_bpb": original_bpb,
            "doc_coherence_win_rate": (wins / count) if count else float("nan"),
            "doc_coherence_margin_bpb_mean": mean(margins) if margins else float("nan"),
            "doc_coherence_margin_bpb_median": median(margins) if margins else float("nan"),
            "doc_coherence_margin_bpb_min": min(margins) if margins else float("nan"),
        }
        for kind in DOC_COHERENCE_KINDS:
            kind_margins = margins_by_kind.get(kind, [])
            kind_count = counts_by_kind.get(kind, 0)
            out[f"doc_coherence_{kind}_count"] = float(kind_count)
            out[f"doc_coherence_{kind}_win_rate"] = (
                wins_by_kind.get(kind, 0) / kind_count
                if kind_count
                else float("nan")
            )
            out[f"doc_coherence_{kind}_margin_bpb_mean"] = (
                mean(kind_margins) if kind_margins else float("nan")
            )
            out[f"doc_coherence_{kind}_margin_bpb_median"] = (
                median(kind_margins) if kind_margins else float("nan")
            )
            out[f"doc_coherence_{kind}_margin_bpb_min"] = (
                min(kind_margins) if kind_margins else float("nan")
            )
        return out


def empty_order_metrics(sentence_count: int = 0) -> dict[str, float]:
    return {
        "order_sentence_count": float(sentence_count),
        "order_pair_count": 0.0,
        "order_contrast_acc": float("nan"),
        "order_contrast_margin_bpb_mean": float("nan"),
        "order_contrast_margin_bpb_median": float("nan"),
        "order_context_gain_bpb_mean": float("nan"),
        "order_distractor_gain_bpb_mean": float("nan"),
        "order_shuffle_pair_count": 0.0,
        "order_shuffle_acc": float("nan"),
        "order_shuffle_margin_bpb_mean": float("nan"),
        "order_shuffle_margin_bpb_median": float("nan"),
        "order_shuffle_actual_gain_bpb_mean": float("nan"),
        "order_shuffle_distractor_gain_bpb_mean": float("nan"),
    }


def empty_document_coherence_metrics(sentence_count: int = 0) -> dict[str, float]:
    out = {
        "doc_coherence_sentence_count": float(sentence_count),
        "doc_coherence_corruption_count": 0.0,
        "doc_coherence_original_bpb": float("nan"),
        "doc_coherence_win_rate": float("nan"),
        "doc_coherence_margin_bpb_mean": float("nan"),
        "doc_coherence_margin_bpb_median": float("nan"),
        "doc_coherence_margin_bpb_min": float("nan"),
    }
    for kind in DOC_COHERENCE_KINDS:
        out[f"doc_coherence_{kind}_count"] = 0.0
        out[f"doc_coherence_{kind}_win_rate"] = float("nan")
        out[f"doc_coherence_{kind}_margin_bpb_mean"] = float("nan")
        out[f"doc_coherence_{kind}_margin_bpb_median"] = float("nan")
        out[f"doc_coherence_{kind}_margin_bpb_min"] = float("nan")
    return out

def iter_samples(
    panel_json: Path,
    limit_per_panel: int,
) -> Iterable[dict[str, str]]:
    data = load_panel(panel_json)
    root = panel_json.parent
    for panel in data["panels"]:
        source = root / panel["sample_path"]
        payload = json.loads(source.read_text())
        items = normalize_payload(payload)
        if limit_per_panel > 0:
            items = items[:limit_per_panel]
        for i, item in enumerate(items):
            yield normalize_sample(panel, item, i)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_models", default="")
    parser.add_argument("--pinned_reference_model", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device_map",
        default="",
        help="Optional Hugging Face device_map, e.g. 'auto', for sharded reference-model scoring.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max_eval_tokens", type=int, default=4096)
    parser.add_argument("--score_batch_size", type=int, default=4)
    parser.add_argument("--limit_samples_per_panel", type=int, default=0)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    parser.add_argument("--fineweb_dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--fineweb_config", default="sample-10BT")
    parser.add_argument("--pinned_fineweb", default="9bb295ddab0e05d785b879661af7260fed5140fc")
    parser.add_argument(
        "--allow_fineweb_fallback",
        action="store_true",
        help="Use small built-in real-text controls if FineWeb cannot be loaded. Intended for local smoke tests only.",
    )
    parser.add_argument(
        "--order_contrast",
        action="store_true",
        help="Score sentence-order contextual contrast for each sample with each reference LM.",
    )
    parser.add_argument("--order_contrast_pairs_per_sample", type=int, default=8)
    parser.add_argument("--order_contrast_distractors", type=int, default=4)
    parser.add_argument("--order_contrast_context_sentences", type=int, default=3)
    parser.add_argument("--order_contrast_min_sentence_tokens", type=int, default=5)
    parser.add_argument("--order_contrast_max_sentence_tokens", type=int, default=80)
    parser.add_argument("--order_contrast_seed", type=int, default=1337)
    parser.add_argument(
        "--order_contrast_distractor_source",
        choices=("panel", "fineweb_controls", "panel_plus_fineweb_controls"),
        default="panel",
        help=(
            "Sentence source for external order-contrast distractors. "
            "Use fineweb_controls for release/crossing runs so scores do not depend on which model panels are bundled together."
        ),
    )
    parser.add_argument("--order_contrast_distractor_fineweb_count", type=int, default=128)
    parser.add_argument("--order_contrast_distractor_seed", type=int, default=-1)
    parser.add_argument(
        "--document_coherence",
        action="store_true",
        help="Score original documents against corrupted sentence-order/replacement variants.",
    )
    parser.add_argument("--doc_coherence_min_sentences", type=int, default=5)
    parser.add_argument("--doc_coherence_max_sentences", type=int, default=12)
    parser.add_argument("--doc_coherence_max_sentence_tokens", type=int, default=80)
    parser.add_argument("--doc_coherence_shuffles", type=int, default=1)
    parser.add_argument("--doc_coherence_swaps", type=int, default=1)
    parser.add_argument("--doc_coherence_replacements", type=int, default=1)
    parser.add_argument("--doc_coherence_block_replacements", type=int, default=1)
    parser.add_argument("--doc_coherence_tail_splices", type=int, default=1)
    parser.add_argument("--doc_coherence_replacement_sentences", type=int, default=2)
    parser.add_argument("--doc_coherence_block_sentences", type=int, default=3)
    parser.add_argument("--doc_coherence_seed", type=int, default=2027)
    args = parser.parse_args()

    total_start = time.perf_counter()
    panel_json = Path(args.panel_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    base_samples = list(iter_samples(panel_json, args.limit_samples_per_panel))
    timing("load_panel_samples", time.perf_counter() - stage_start, samples=len(base_samples))
    sentence_cache: list[list[str]] = []
    candidate_pool: list[tuple[int, str]] = []
    candidate_docs: list[tuple[int, list[str]]] = []
    if args.order_contrast or args.document_coherence:
        stage_start = time.perf_counter()
        use_panel_distractors = args.order_contrast_distractor_source in {
            "panel",
            "panel_plus_fineweb_controls",
        }
        for sample_index, sample in enumerate(base_samples):
            sentences = split_sentences(sample["target_text"])
            sentence_cache.append(sentences)
            if use_panel_distractors:
                candidate_docs.append((sample_index, sentences))
            if use_panel_distractors:
                for sentence in sentences:
                    candidate_pool.append((sample_index, sentence))
        if args.order_contrast_distractor_source in {"fineweb_controls", "panel_plus_fineweb_controls"}:
            distractor_seed = (
                1337
                if args.order_contrast_distractor_seed < 0
                else args.order_contrast_distractor_seed
            )
            control_start = time.perf_counter()
            distractor_samples = load_fineweb_controls(
                count=args.order_contrast_distractor_fineweb_count,
                seed=distractor_seed,
                dataset_name=args.fineweb_dataset,
                dataset_config=args.fineweb_config,
                pinned_fineweb=args.pinned_fineweb,
                allow_fallback=args.allow_fineweb_fallback,
            )
            timing(
                "load_fineweb_controls",
                time.perf_counter() - control_start,
                controls=len(distractor_samples),
                source=args.order_contrast_distractor_source,
            )
            base_offset = len(base_samples)
            for distractor_index, sample in enumerate(distractor_samples):
                sentences = split_sentences(sample["target_text"])
                candidate_docs.append((base_offset + distractor_index, sentences))
                for sentence in sentences:
                    candidate_pool.append((base_offset + distractor_index, sentence))
        print(
            "Prepared sentence-order contrast pool: "
            f"{len(candidate_pool)} sentences from {args.order_contrast_distractor_source}",
            flush=True,
        )
        timing(
            "prepare_sentence_pool",
            time.perf_counter() - stage_start,
            sentences=len(candidate_pool),
            docs=len(candidate_docs),
        )
    stage_start = time.perf_counter()
    deterministic_rows: list[dict[str, Any]] = []
    for sample in base_samples:
        row: dict[str, Any] = {
            key: value
            for key, value in sample.items()
            if key not in {"prompt", "target_text", "full_text"}
        }
        row["reference_model"] = "deterministic"
        row.update(deterministic_metrics(sample["target_text"]))
        deterministic_rows.append(row)
    timing("deterministic_metrics", time.perf_counter() - stage_start, samples=len(base_samples))

    all_rows = list(deterministic_rows)
    models = [model.strip() for model in args.reference_models.split(",") if model.strip()]
    for model_name in models:
        print(f"Loading reference LM: {model_name}", flush=True)
        model_load_start = time.perf_counter()
        scorer = ReferenceLMScorer(
            model_name,
            device=args.device,
            dtype=args.dtype,
            max_eval_tokens=args.max_eval_tokens,
            score_batch_size=args.score_batch_size,
            trust_remote_code=not args.no_trust_remote_code,
            device_map=args.device_map,
            pinned_reference_model=args.pinned_reference_model,
        )
        print(f"Loaded {model_name}", flush=True)
        timing("reference_model_load", time.perf_counter() - model_load_start, model=model_name)
        model_score_start = time.perf_counter()
        ref_score_seconds = 0.0
        order_seconds = 0.0
        doc_seconds = 0.0
        for i, sample in enumerate(base_samples, start=1):
            row = {
                key: value
                for key, value in sample.items()
                if key not in {"prompt", "target_text", "full_text"}
            }
            row["reference_model"] = model_name
            row.update(deterministic_metrics(sample["target_text"]))
            section_start = time.perf_counter()
            row.update(scorer.score(sample["prompt"], sample["target_text"]))
            ref_score_seconds += time.perf_counter() - section_start
            if args.order_contrast:
                section_start = time.perf_counter()
                row.update(
                    scorer.score_order_contrast(
                        sample=sample,
                        sample_index=i - 1,
                        sentence_cache=sentence_cache,
                        candidate_pool=candidate_pool,
                        pairs_per_sample=args.order_contrast_pairs_per_sample,
                        distractors=args.order_contrast_distractors,
                        context_sentences=args.order_contrast_context_sentences,
                        min_sentence_tokens=args.order_contrast_min_sentence_tokens,
                        max_sentence_tokens=args.order_contrast_max_sentence_tokens,
                        seed=args.order_contrast_seed,
                    )
                )
                order_seconds += time.perf_counter() - section_start
            if args.document_coherence:
                section_start = time.perf_counter()
                row.update(
                    scorer.score_document_coherence(
                        sample=sample,
                        sample_index=i - 1,
                        sentence_cache=sentence_cache,
                        candidate_docs=candidate_docs,
                        min_sentences=args.doc_coherence_min_sentences,
                        max_sentences=args.doc_coherence_max_sentences,
                        max_sentence_tokens=args.doc_coherence_max_sentence_tokens,
                        shuffles=args.doc_coherence_shuffles,
                        swaps=args.doc_coherence_swaps,
                        replacements=args.doc_coherence_replacements,
                        block_replacements=args.doc_coherence_block_replacements,
                        tail_splices=args.doc_coherence_tail_splices,
                        replacement_sentences=args.doc_coherence_replacement_sentences,
                        block_sentences=args.doc_coherence_block_sentences,
                        seed=args.doc_coherence_seed,
                    )
                )
                doc_seconds += time.perf_counter() - section_start
            all_rows.append(row)
            if i % 25 == 0:
                elapsed = time.perf_counter() - model_score_start
                print(f"Scored {i}/{len(base_samples)} samples with {model_name}", flush=True)
                timing(
                    "reference_model_score_progress",
                    elapsed,
                    model=model_name,
                    samples=i,
                    total=len(base_samples),
                    avg_sample_seconds=f"{elapsed / i:.3f}",
                )
        timing("reference_model_ref_bpb", ref_score_seconds, model=model_name)
        timing("reference_model_order_contrast", order_seconds, model=model_name)
        timing("reference_model_document_coherence", doc_seconds, model=model_name)
        timing("reference_model_score_total", time.perf_counter() - model_score_start, model=model_name)
    stage_start = time.perf_counter()
    write_csv(output_dir / "sample_scores.csv", all_rows)
    (output_dir / "README.md").write_text(
        "# Generation Quality Scores\n\n"
        f"- samples scored: `{len(base_samples)}`\n"
        f"- reference models: `{', '.join(models) if models else 'deterministic only'}`\n"
        f"- sample scores: `sample_scores.csv`\n"
    )
    timing("write_outputs", time.perf_counter() - stage_start, rows=len(all_rows))
    print(f"Wrote {output_dir / 'sample_scores.csv'}")
    print(f"Wrote {output_dir / 'README.md'}")
    timing("total", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
