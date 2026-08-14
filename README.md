# speedrun-dlm: A speedrun benchmark for diffusion language models

**Train a diffusion language model (DLM) as fast as possible to generate decent text.**

This repository asks how quickly one can train a diffusion language model on
*FineWeb*, using *8 H100s*, until it generates 1024-token samples that are good
enough to pass a simple quality gate.

**The leaderboard is based on generated samples passing this gate (using Gen. PPL plus additional checks for failure modes of PPL alone, like token repetition and coherence).**

The **gate** checks for:

- too much repetition,
- basic document coherence,
- quality according to a pinned reference LLM (`Qwen2.5-7B`),
- other simple heuristics (more details below).

The project is inspired by Keller Jordan and collaborators' **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun** and adapts it to diffusion language models, for which many core design choices are still open.

We fix a small DDiT backbone (170M parameters) and include re-implementations of **7 common single-block diffusion objectives** in `speedrun_dlm/train_dlm.py`.

> Note. Original methods and ideas belong to their respective authors. This is an independent project and is not affiliated with modded-nanogpt, its contributors, or the authors of the papers we cite.


## Contents

- [Leaderboard](#leaderboard)
- [Rules](#rules)
- [Model](#model)
- [Quality gate](#quality-gate)
- [How to submit](#how-to-submit)
- [Run](#run)
- [Check a candidate](#check-a-candidate)
- [Appendix](APPENDIX.md)
- [Acknowledgements](#acknowledgements)

## Leaderboard

**Runs are ranked by training time to pass the sample-quality gate.**

To appear on the leaderboard, a DLM training-and-sampling recipe must:

- pass the quality gate (see [Rules](#rules) for the exact criterion), and
- have a mean generation cost below 95.86 TFLOPs per 1024-token sample (this threshold is explained in the [appendix](APPENDIX.md)).

Small tuning changes, new samplers, and new objectives are all welcome.

The FLOP cap is meant to keep samplers reasonable in terms of inference cost. A diffusion sampler could otherwise try to pass the gate by using an arbitrarily large number of denoising steps.

| Rank by train time | Entry | Mean train time (used for ranking) | Gate result (need ≥4881/6400) | Mean 1024-token generation cost (need <95.86 TFLOPs) | Mean model calls for 1024 tokens | Training tokens | Training steps | Record |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `subs_mask` | [27m45.2s](records/0001-subs-mask-s344/training_runs.jsonl) | [5385/6400](records/0001-subs-mask-s344/quality_gate.json) | [95.41 TFLOPs](records/0001-subs-mask-s344/auxiliary_metrics.json) | [326.84](records/0001-subs-mask-s344/auxiliary_metrics.json) | [3.28B](records/0001-subs-mask-s344/training_runs.jsonl) | [8000](records/0001-subs-mask-s344/training_runs.jsonl) | [0001](records/0001-subs-mask-s344/) |

Inference TFLOPs are FLOPs counted by PyTorch while sampling, plus the attention FLOPs PyTorch did not count. See the [appendix](APPENDIX.md) for the formula.

## Rules

- Same *data*: pinned FineWeb GPT-2 tokenized data.
- Same *scorer*: pinned `Qwen/Qwen2.5-7B`.
- Same *hardware*: 8 H100 80GB GPUs.
- Same per-sample *quality rule*: a conjunction of checks for repetition, surface validity, reference-model BPB (Gen. PPL), sentence order, and document coherence, defined exactly in [`generation_quality_rule.json`](generation_quality_rule.json) and [`score_generation_quality.py`](speedrun_dlm/score_generation_quality.py).
- Same 50 *training seeds*: every official entry must run each integer seed from `1` through `50`.
- Same training *budget across seeds*: all 50 runs for an entry must use the same number of training steps and process the same number of tokens.
- Checkpoint-derived *evaluation randomness*: separate seeds for sampling, inference-cost profiling, and randomized quality checks are automatically derived from a hash of the checkpoint to evaluate. This follows a helpful suggestion by [@JCBrouwer in issue #1](https://github.com/agonon/speedrun-dlm/issues/1). For reproducibility, text is always sampled with a fixed batch size of 16.
- From each of the 50 trained checkpoints, generate 128 sequences of 1024 tokens, starting from a fully noised state (all mask tokens for mask-noise models or uniformly random tokens for uniform-noise models). (For comparison, AR models can start from a single token; we used GPT-2's beginning-of-text token for the AR model mentioned in the [Appendix](APPENDIX.md).)
- Of the resulting `50 × 128 = 6400` sequences, at least `4881` must pass the per-sample quality rule. See the [Appendix](APPENDIX.md#statistical-interpretation-of-the-quality-gate) for the statistical interpretation as a one-sided binomial test.
- The mean inference cost across the 50 checkpoint-derived profiles, using the same sampler configuration as the quality gate, must be less than 95.86 TFLOPs per 1024-token sample.
- Runs are ranked by mean training time over 50 seeds.

While architectural modifications are acceptable, the DDiT backbone parameter count should remain comparable.

## Model

The benchmark fixes a small DDiT backbone at roughly nanoGPT-2 scale:

- 12 layers,
- width 768,
- 12 heads,
- context length 1024,
- GPT-2 tokenizer.

The diffusion-specific changes are:

- non-causal attention,
- denoising-time conditioning through adaptive layer norm,
- untied input embedding and output head.

See the [appendix](APPENDIX.md) for the short comparison with the AR reference used to set the FLOP cap.

## Quality gate

**DLMs are not easy to compare** because:

- there is no direct access to the **sequence-level probability** assigned by a model: unlike for an AR model, this requires estimating the sum over all possible denoising paths to reach a given sample, which is expensive and leads to high-variance estimates,
- the **different losses** in current popular recipes are not directly comparable (even when they constitute variational bounds on the negative log-likelihood, these bounds may have different tightness across parameterizations),
- the **sampler** used at inference time must be taken into account, since the same DLM checkpoint can produce different text under different sampling procedures.

This benchmark **evaluates a trained model together with a sampler** by:

1. Generating 128 unconditional samples of 1024 tokens for each of the 50 training seeds, following the initialization protocol specified in the [Rules](#rules).
2. Applying the quality rule to every sample (repetition, fluency, etc.).
3. Deciding whether the DLM training-and-sampling recipe passes by counting how many of the `50 × 128 = 6400` samples pass the quality rule. At least `4881` must pass for the entry to be eligible for the speedrun leaderboard.

See [`score_generation_quality.py`](speedrun_dlm/score_generation_quality.py) and [`generation_quality_rule.json`](generation_quality_rule.json) for details.

## How to submit

To get your method on the leaderboard, **open a pull request** with:

- a new folder under `records/`,
- a new row in [`records/records.csv`](records/records.csv),
- any code needed to run the method.

If you use the current trainer, sampler, and scorer outputs, you can create the record folder with [`records/make_record.py`](records/make_record.py):

```bash
gate_args=()
inference_args=()
for seed in $(seq 1 50); do
  gate_args+=(--gate-dir "results/my_method/gate_seed_${seed}")
  inference_args+=(--inference-cost-json "results/my_method/inference_cost_seed_${seed}.json")
done

python3 records/make_record.py \
  --record-dir records/nextfreeid-my-method \
  --record-id nextfreeid \
  --entry my_method \
  --trainer dlm \
  --sampler S=mysamplingsteps \
  --training-runs-jsonl results/my_method/training_runs.jsonl \
  "${gate_args[@]}" \
  "${inference_args[@]}" \
  --environment-json results/my_method/train_seed_1/environment.json \
  --records-csv records/records.csv
```

Each submitted record must satisfy the [rules](#rules) above. We will check the timing and record artifacts on our cluster for each PR.

## Run

![Edit, run, check, record](assets/demo.gif)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -r requirements.txt

bash prepare_data.sh
bash run_ar.sh
bash run_dlm.sh
```

## Check a candidate

```bash
python3 -m speedrun_dlm.score_generation_quality path/to/checkpoint.pt \
  --samples 128 \
  --tokens 1024 \
  --top_k 0 \
  --num_sampling_steps 344 \
  --sampling_eps 1e-3 \
  --output_dir results/gate
```

This command records the sample-level outcomes for one checkpoint. The official pass/fail result is produced by `records/make_record.py` after all 50 outputs are provided.

```bash
python3 -m speedrun_dlm.measure_inference_cost path/to/checkpoint.pt \
  --tokens 1024 \
  --top_k 0 \
  --num_sampling_steps 344 \
  --sampling_eps 1e-3 \
  --output_json results/inference_cost_seed_1.json
```

Run the profiler once for each training checkpoint. The record builder verifies that the profiles and quality gates use the same 50 checkpoints and sampler configurations, then averages their inference costs.

## Appendix

**More details on objectives, samplers, architecture, and figures are in [the appendix](APPENDIX.md).**

## Acknowledgements

This repo builds on a lot of useful public work. In particular, we used:

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) for the speedrun setup and AR reference,
- [MDLM](https://github.com/kuleshov-group/mdlm), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), [D3PM](https://github.com/google-research/google-research/tree/master/d3pm), and [DUO](https://github.com/s-sahoo/duo) for the different DLM objectives and samplers,
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2), and [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) for the data and Gen PPL scorer.

## Cite as

The first version of this benchmark was created by Antoine Gonon, Adrian Müller,
Léon Zheng, Zebang Shen, Clément Lalanne, Ya-Ping Hsieh, Anthony Bardou, and
Nicolas Boumal.

```bibtex
@misc{speedrun_dlm,
  title = {speedrun-dlm: A speedrun benchmark for diffusion language models},
  author = {Gonon, Antoine and Müller, Adrian and Zheng, Léon and Shen, Zebang and Lalanne, Clément and Hsieh, Ya-Ping and Bardou, Anthony and Boumal, Nicolas},
  year = {2026}
}
```

## Contact

For private questions or feedback, feel free to reach out to `antoine.gonon@epfl.ch`.

For bugs, reproducibility issues, or proposed changes, please open a GitHub issue or PR.
