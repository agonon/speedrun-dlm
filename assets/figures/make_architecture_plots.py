#!/usr/bin/env python3
"""Regenerate the public nanoGPT-2 vs DDiT architecture comparison SVGs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "assets" / "figures"
AR_COST = ROOT / "records" / "ar-baseline" / "auxiliary_metrics.json"
DDIT_ARCHITECTURE_COST = {
    "forward_calls_per_sample": 622.0,
    "num_parameters": 169_626_449,
    "profiler_tflops_per_1024_token_sample": 157.557743865198,
}
AR_NUM_PARAMETERS = 124_318_464

SEQ = 1024
LAYERS = 12
D_MODEL = 768
COND_DIM = 128
VOCAB = 50257
TIME_FREQ_DIM = 256

INK = "#1f2933"
MUTED = "#5b6773"
GRID = "#d9dee5"
BG = "#ffffff"
BLUE = "#4e79a7"
GREEN = "#59a14f"
ORANGE = "#f28e2b"
RED = "#e15759"
PURPLE = "#b07aa1"
GRAY = "#bab0ac"
LIGHT_GRAY = "#edf1f5"


def load_cost(path: Path) -> dict:
    return json.loads(path.read_text())["inference_cost"]


def load_ar_cost() -> dict:
    return load_cost(AR_COST)


def load_ddit_architecture_cost() -> dict:
    return DDIT_ARCHITECTURE_COST


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            "<defs>",
            "<style>",
            "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }",
            ".title { font-size: 26px; font-weight: 700; }",
            ".subtitle { font-size: 14px; fill: #5b6773; }",
            ".label { font-size: 14px; font-weight: 600; }",
            ".small { font-size: 12px; fill: #5b6773; }",
            ".tiny { font-size: 11px; fill: #5b6773; }",
            ".mono { font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #1f2933; }",
            ".value { font-size: 13px; font-weight: 650; }",
            "</style>",
            '<pattern id="diag" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
            '<rect width="8" height="8" fill="#edf1f5"/>',
            '<line x1="0" y1="0" x2="0" y2="8" stroke="#8b96a3" stroke-width="3"/>',
            "</pattern>",
            "</defs>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def text(x: float, y: float, value: object, klass: str = "", anchor: str = "start", color: str | None = None) -> str:
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"']
    if klass:
        attrs.append(f'class="{klass}"')
    if color:
        attrs.append(f'style="fill: {color}"')
    return f"<text {' '.join(attrs)}>{esc(value)}</text>"


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'width="{max(w, 0):.1f}"',
        f'height="{h:.1f}"',
        f'fill="{fill}"',
    ]
    if stroke:
        attrs.append(f'stroke="{stroke}"')
    return f"<rect {' '.join(attrs)}/>"


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'


def stacked_bar(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    total_scale: float,
    segments: list[tuple[str, float, str]],
    label_min_px: float = 55,
    unit: str = "",
) -> None:
    cursor = x
    for _, value, color in segments:
        w = width * value / total_scale if total_scale else 0
        if w >= 0.5:
            body.append(rect(cursor, y, w, height, color))
            if w >= label_min_px and value > 0:
                body.append(text(cursor + w / 2, y + height / 2 + 5, f"{value:.1f}{unit}", "value", "middle", "#ffffff"))
        cursor += w
    body.append(rect(x, y, width * sum(value for _, value, _ in segments) / total_scale, height, "none", INK))


def legend(body: list[str], x: float, y: float, items: list[tuple[str, str]]) -> None:
    cursor = x
    for name, color in items:
        body.append(rect(cursor, y - 10, 14, 14, color))
        body.append(text(cursor + 20, y + 1, name, "small"))
        cursor += 20 + len(name) * 7.0 + 28


def attention_score_value_estimate(backend: str) -> float:
    ddit_calls = float(load_ddit_architecture_cost()["forward_calls_per_sample"])
    if backend == "gpt":
        causal_pairs_over_prefixes = sum(t * (t + 1) / 2 for t in range(1, SEQ + 1))
        return 4 * D_MODEL * causal_pairs_over_prefixes / 1e12
    if backend == "ddit":
        return 4 * D_MODEL * ddit_calls * SEQ * SEQ / 1e12
    raise ValueError(backend)


def make_params() -> None:
    width, height = 1000, 545
    body: list[str] = [
        text(40, 48, "nanoGPT-2 vs DDiT parameters", "title"),
        text(40, 75, "d12 config: 12 layers, d=768, vocab=50,257, cond_dim=128", "subtitle"),
    ]

    gpt_token = VOCAB * D_MODEL / 1e6
    gpt_pos = SEQ * D_MODEL / 1e6
    gpt_attn = 4 * D_MODEL * D_MODEL / 1e6
    gpt_mlp = 8 * D_MODEL * D_MODEL / 1e6

    ddit_input = (VOCAB + 1) * D_MODEL / 1e6
    ddit_attn = gpt_attn
    ddit_mlp = (8 * D_MODEL * D_MODEL + 5 * D_MODEL) / 1e6
    ddit_adaln = (COND_DIM * 6 * D_MODEL + 6 * D_MODEL) / 1e6
    ddit_norms = 2 * D_MODEL / 1e6
    ddit_block = ddit_attn + ddit_mlp + ddit_adaln + ddit_norms
    ddit_time = (TIME_FREQ_DIM * COND_DIM + COND_DIM + COND_DIM * COND_DIM + COND_DIM) / 1e6
    ddit_output = (
        D_MODEL
        + D_MODEL * VOCAB
        + VOCAB
        + COND_DIM * 2 * D_MODEL
        + 2 * D_MODEL
    ) / 1e6

    body.append(text(40, 116, "Total parameters", "label"))
    x, bar_w, h = 220, 650, 38
    max_total = 175.0
    body.append(text(130, 157, "nanoGPT-2", "label", "end"))
    stacked_bar(
        body,
        x,
        132,
        bar_w,
        h,
        max_total,
        [
            ("token emb / tied head", gpt_token, BLUE),
            ("position emb", gpt_pos, GRAY),
            ("12 transformer blocks", LAYERS * (gpt_attn + gpt_mlp), GREEN),
        ],
        unit="M",
    )
    body.append(text(x + bar_w + 14, 157, f"{AR_NUM_PARAMETERS / 1e6:.1f}M", "value"))
    body.append(text(130, 214, "DDiT", "label", "end"))
    stacked_bar(
        body,
        x,
        189,
        bar_w,
        h,
        max_total,
        [
            ("input emb", ddit_input, BLUE),
            ("12 DDiT blocks", LAYERS * ddit_block, GREEN),
            ("time map", ddit_time, PURPLE),
            ("untied output layer", ddit_output, RED),
        ],
        unit="M",
    )
    body.append(text(x + bar_w + 14, 214, f"{load_ddit_architecture_cost()['num_parameters'] / 1e6:.1f}M", "value"))
    legend(body, 220, 258, [("embeddings", BLUE), ("blocks", GREEN), ("time/conditioning", PURPLE), ("DDiT output head", RED)])

    body.append(line(40, 305, 960, 305, LIGHT_GRAY, 2))
    body.append(text(40, 350, "One transformer block", "label"))
    x2, bar_w2 = 220, 570
    max_layer = 8.2
    body.append(text(130, 393, "nanoGPT-2 block", "label", "end"))
    stacked_bar(body, x2, 368, bar_w2, h, max_layer, [("attention", gpt_attn, ORANGE), ("MLP", gpt_mlp, GREEN)], unit="M")
    body.append(text(x2 + bar_w2 + 14, 393, f"{gpt_attn + gpt_mlp:.2f}M", "value"))
    body.append(text(130, 450, "DDiT block", "label", "end"))
    stacked_bar(
        body,
        x2,
        425,
        bar_w2,
        h,
        max_layer,
        [("attention", ddit_attn, ORANGE), ("MLP", ddit_mlp, GREEN), ("adaLN", ddit_adaln, PURPLE), ("norms", ddit_norms, GRAY)],
        unit="M",
    )
    body.append(text(x2 + bar_w2 + 14, 450, f"{ddit_block:.2f}M", "value"))
    legend(body, 220, 494, [("attention", ORANGE), ("MLP", GREEN), ("adaLN", PURPLE), ("norms", GRAY)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "architecture-params-per-layer.svg").write_text(svg(width, height, body))


def make_params_total() -> None:
    width, height = 1000, 300
    body: list[str] = [
        text(40, 48, "nanoGPT-2 vs DDiT parameters", "title"),
        text(40, 75, "d12 config: 12 layers, d=768, vocab=50,257", "subtitle"),
    ]

    gpt_token = VOCAB * D_MODEL / 1e6
    gpt_pos = SEQ * D_MODEL / 1e6
    gpt_attn = 4 * D_MODEL * D_MODEL / 1e6
    gpt_mlp = 8 * D_MODEL * D_MODEL / 1e6

    ddit_input = (VOCAB + 1) * D_MODEL / 1e6
    ddit_attn = gpt_attn
    ddit_mlp = (8 * D_MODEL * D_MODEL + 5 * D_MODEL) / 1e6
    ddit_adaln = (COND_DIM * 6 * D_MODEL + 6 * D_MODEL) / 1e6
    ddit_norms = 2 * D_MODEL / 1e6
    ddit_block = ddit_attn + ddit_mlp + ddit_adaln + ddit_norms
    ddit_time = (TIME_FREQ_DIM * COND_DIM + COND_DIM + COND_DIM * COND_DIM + COND_DIM) / 1e6
    ddit_output = (
        D_MODEL
        + D_MODEL * VOCAB
        + VOCAB
        + COND_DIM * 2 * D_MODEL
        + 2 * D_MODEL
    ) / 1e6

    x, bar_w, h = 220, 650, 42
    max_total = 175.0
    body.append(text(130, 137, "nanoGPT-2", "label", "end"))
    stacked_bar(
        body,
        x,
        112,
        bar_w,
        h,
        max_total,
        [
            ("token emb / tied head", gpt_token, BLUE),
            ("position emb", gpt_pos, GRAY),
            ("12 transformer blocks", LAYERS * (gpt_attn + gpt_mlp), GREEN),
        ],
        unit="M",
    )
    body.append(text(x + bar_w + 14, 137, f"{AR_NUM_PARAMETERS / 1e6:.1f}M", "value"))

    body.append(text(130, 198, "DDiT", "label", "end"))
    stacked_bar(
        body,
        x,
        173,
        bar_w,
        h,
        max_total,
        [
            ("input emb", ddit_input, BLUE),
            ("12 DDiT blocks", LAYERS * ddit_block, GREEN),
            ("time map", ddit_time, PURPLE),
            ("untied output layer", ddit_output, RED),
        ],
        unit="M",
    )
    body.append(text(x + bar_w + 14, 198, f"{load_ddit_architecture_cost()['num_parameters'] / 1e6:.1f}M", "value"))
    legend(body, 220, 252, [("embeddings", BLUE), ("blocks", GREEN), ("time/conditioning", PURPLE), ("DDiT output head", RED)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "architecture-params-total.svg").write_text(svg(width, height, body))


def make_flops_per_layer() -> None:
    dlm = load_ddit_architecture_cost()
    ddit_calls = float(dlm["forward_calls_per_sample"])
    prefix_positions = SEQ * (SEQ + 1) / 2
    ddit_positions = ddit_calls * SEQ
    ar_attn = 8 * D_MODEL * D_MODEL * prefix_positions / 1e12
    ar_mlp = 16 * D_MODEL * D_MODEL * prefix_positions / 1e12
    ddit_attn = 8 * D_MODEL * D_MODEL * ddit_positions / 1e12
    ddit_mlp = 16 * D_MODEL * D_MODEL * ddit_positions / 1e12
    ddit_adaln = 2 * COND_DIM * 6 * D_MODEL * ddit_calls / 1e12
    ar_attn_math = attention_score_value_estimate("gpt")
    ddit_attn_math = attention_score_value_estimate("ddit")

    width, height = 1000, 475
    body: list[str] = [
        text(40, 48, "Per-block FLOPs", "title"),
        text(40, 75, "Measured: nn.Linear via torch.profiler. Estimated: SDPA by formula.", "subtitle"),
    ]

    # Top: measured nn.Linear FLOPs
    body.append(text(40, 112, "Measured nn.Linear FLOPs", "label"))
    x, y0, bar_w, h = 220, 136, 430, 36
    max_flops = 12.0

    body.append(text(170, y0 + 24, "nanoGPT-2 block", "label", "end"))
    stacked_bar(
        body,
        x,
        y0,
        bar_w,
        h,
        max_flops,
        [
            ("c_attn + c_proj", ar_attn, ORANGE),
            ("MLP linears", ar_mlp, GREEN),
        ],
        unit="T",
    )
    body.append(text(x + bar_w + 14, y0 + 24, f"{ar_attn + ar_mlp:.2f} TF", "value"))

    body.append(text(170, y0 + 82, "DDiT block", "label", "end"))
    stacked_bar(
        body,
        x,
        y0 + 58,
        bar_w,
        h,
        max_flops,
        [
            ("c_attn + c_proj", ddit_attn, ORANGE),
            ("MLP linears", ddit_mlp, GREEN),
            ("adaLN", ddit_adaln, PURPLE),
        ],
        unit="T",
    )
    body.append(text(x + bar_w + 14, y0 + 82, f"{ddit_attn + ddit_mlp + ddit_adaln:.2f} TF", "value"))

    legend(
        body,
        220,
        248,
        [
            ("c_attn(x), c_proj(y)", ORANGE),
            ("MLP linears", GREEN),
            ("adaLN", PURPLE),
        ],
    )

    # Divider
    body.append(line(40, 280, 960, 280, LIGHT_GRAY, 2))

    # Bottom: estimated SDPA FLOPs
    body.append(text(40, 316, "Estimated SDPA FLOPs", "label"))
    y1 = 350
    sdpa_scale = 2.4

    body.append(text(170, y1 + 23, "nanoGPT-2 causal", "label", "end"))
    body.append(rect(x, y1, bar_w * ar_attn_math / sdpa_scale, h, "url(#diag)", INK))
    body.append(text(x + bar_w * ar_attn_math / sdpa_scale + 14, y1 + 23, f"{ar_attn_math:.2f} TF", "value"))

    body.append(text(170, y1 + 81, "DDiT non-causal", "label", "end"))
    body.append(rect(x, y1 + 58, bar_w * ddit_attn_math / sdpa_scale, h, "url(#diag)", INK))
    body.append(text(x + bar_w * ddit_attn_math / sdpa_scale + 14, y1 + 81, f"{ddit_attn_math:.2f} TF", "value"))

    # One compact note box only
    body.append(rect(745, 122, 205, 225, LIGHT_GRAY))
    body.append(text(762, 148, "torch.profiler", "label"))
    body.append(text(762, 171, "qkv = c_attn(x)", "mono"))
    body.append(text(762, 190, "y = c_proj(y)", "mono"))
    body.append(text(762, 209, "MLP, adaLN", "mono"))

    body.append(text(762, 246, "SDPA estimate", "label"))
    body.append(text(762, 269, "4 x B x H x P x d_h", "tiny"))
    body.append(text(762, 291, "P = T(T+1)/2 causal", "tiny"))
    body.append(text(762, 311, "P = T^2 non-causal", "tiny"))
    body.append(text(762, 334, "SDPA shown analytically", "tiny"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "architecture-flops-per-layer.svg").write_text(svg(width, height, body))


def make_extra_flops() -> None:
    ar = load_ar_cost()
    dlm = load_ddit_architecture_cost()
    ddit_calls = float(dlm["forward_calls_per_sample"])
    observed_delta = dlm["profiler_tflops_per_1024_token_sample"] - ar["profiler_tflops_per_1024_token_sample"]

    prefix_positions = SEQ * (SEQ + 1) / 2
    block_extra = (
        ddit_calls * LAYERS * 24 * SEQ * D_MODEL * D_MODEL
        - LAYERS * 24 * D_MODEL * D_MODEL * prefix_positions
    ) / 1e12
    head_extra = (ddit_calls * 2 * SEQ * D_MODEL * VOCAB - 2 * SEQ * D_MODEL * VOCAB) / 1e12
    cond_extra = (
        ddit_calls
        * (
            LAYERS * 2 * COND_DIM * 6 * D_MODEL
            + 2 * COND_DIM * 2 * D_MODEL
            + 2 * TIME_FREQ_DIM * COND_DIM
            + 2 * COND_DIM * COND_DIM
        )
    ) / 1e12
    other = max(observed_delta - block_extra - head_extra - cond_extra, 0.0)
    attn_extra = LAYERS * (attention_score_value_estimate("ddit") - attention_score_value_estimate("gpt"))

    ar_total = float(ar["profiler_tflops_per_1024_token_sample"])
    dlm_total = float(dlm["profiler_tflops_per_1024_token_sample"])
    total_extra = observed_delta + attn_extra

    width, height = 1000, 320
    body: list[str] = [
        text(40, 48, "Inference FLOPs for one 1024-token sample", "title"),
        text(40, 75, "Measured: nn.Linear via torch.profiler. Estimated: SDPA by formula.", "subtitle"),
        text(
            40,
            101,
            f"nanoGPT-2 {ar_total:.2f} TF | DDiT {dlm_total:.2f} TF | extra {observed_delta:.2f} TF measured + {attn_extra:.2f} TF estimated SDPA",
            "label",
        ),
    ]

    x, y, bar_w, h = 70, 132, 850, 56
    body.append(text(x, y - 14, "Extra FLOPs breakdown", "label"))

    measured_components = [
        ("DDiT output head", head_extra, RED),
        ("extra full-seq block linears", block_extra, GREEN),
        ("adaLN/time-conditioning", cond_extra, PURPLE),
        ("other measured ops", other, GRAY),
    ]

    cursor = x
    for _, value, color in measured_components:
        w = bar_w * value / total_extra if total_extra else 0.0
        if w > 0.5:
            body.append(rect(cursor, y, w, h, color))
            if w >= 75 and value > 0:
                body.append(text(cursor + w / 2, y + h / 2 + 5, f"{value:.1f}T", "value", "middle", "#ffffff"))
        cursor += w

    est_w = bar_w * attn_extra / total_extra if total_extra else 0.0
    body.append(rect(cursor, y, est_w, h, "url(#diag)", INK))
    if est_w >= 70:
        body.append(text(cursor + est_w / 2, y + h / 2 + 5, f"{attn_extra:.1f}T", "value", "middle"))

    body.append(rect(x, y, bar_w, h, "none", INK))
    body.append(text(x, y + 78, "0 TF", "small"))
    body.append(text(x + bar_w, y + 78, f"{total_extra:.2f} TF", "small", "end"))

    def legend_item(x0: float, y0: float, label: str, fill: str, stroke: str | None = None) -> None:
        body.append(rect(x0, y0 - 10, 14, 14, fill, stroke))
        body.append(text(x0 + 20, y0 + 1, label, "small"))

    legend_item(70, 235, f"DDiT output head {100 * head_extra / total_extra:.1f}%", RED)
    legend_item(300, 235, f"extra full-seq block linears {100 * block_extra / total_extra:.1f}%", GREEN)
    legend_item(610, 235, f"estimated SDPA {100 * attn_extra / total_extra:.1f}%", "url(#diag)", INK)

    legend_item(70, 264, f"adaLN/time-conditioning {100 * cond_extra / total_extra:.1f}%", PURPLE)
    legend_item(360, 264, f"other measured ops {100 * other / total_extra:.1f}%", GRAY)

    body.append(
        text(
            70,
            300,
            "SDPA estimate: 4 x B x H x P x d_h; P=T(T+1)/2 causal, P=T^2 non-causal.",
            "tiny",
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "architecture-additional-flops.svg").write_text(svg(width, height, body))

def main() -> None:
    make_params_total()
    make_params()
    make_flops_per_layer()
    make_extra_flops()


if __name__ == "__main__":
    main()
