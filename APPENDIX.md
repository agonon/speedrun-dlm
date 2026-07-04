# speedrun-dlm: Appendix


This appendix provides a few extra details about the code.

## Models

- `dlm`: small shared DDiT backbone used for all DLM objectives and samplers ([model](speedrun_dlm/train_dlm.py)).
- `ar`: nanoGPT-2 causal transformer ([model](speedrun_dlm/train_ar.py), purely for reference).

The public AR sampler has no KV cache ([sampler](speedrun_dlm/sample_text.py)). The cached AR number in the leaderboard was profiled separately on H100 80GB GPUs.

The README explains the model setup. In short, DDiT starts from the nanoGPT-2 shape and makes the usual DLM changes:

- non-causal attention,
- time conditioning,
- untied input and output embeddings.

The README figures for parameters and FLOPs are made with:

```bash
python assets/figures/make_architecture_plots.py
```

For reference, these are the DDiT parts we match. The code here is our own re-implementation, so any mistake is ours.

| Local component | Upstream anchors |
| --- | --- |
| sinusoidal time embedding + `F.silu(time_map(t))` | [MDLM](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/models/dit.py#L150-L189), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/model/transformer.py#L56-L94), [DUO](https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/models/dit.py#L176-L215) |
| non-causal DDiT block with RoPE and adaLN-Zero gates | [MDLM](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/models/dit.py#L214-L288), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/model/transformer.py#L118-L189), [DUO](https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/models/dit.py#L305-L379) |
| final LayerNorm + adaLN modulation + linear head | [MDLM](https://github.com/kuleshov-group/mdlm/blob/c112c526d193436838c98d81455ee51f90309470/models/dit.py#L302-L321), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/model/transformer.py#L207-L224), [DUO](https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/models/dit.py#L406-L428) |
| dense and weighted embedding inputs | [DUO](https://github.com/s-sahoo/duo/blob/492505208b361fa330f4703b705abc54cf7ead20/models/dit.py#L382-L403) |


## DLM objectives

`run_dlm.sh` uses `subs_mask` by default. Other objectives can be selected with `--objective`. 

When no schedule or time-conditioning flag is passed, the trainer uses the local default for that objective. 

The implementations are in [`train_dlm.py`](speedrun_dlm/train_dlm.py).

| Objective | Paper | Noise | Loss | Default sampler |
| --- | --- | --- | --- | --- |
| `subs_mask` | [SUBS](https://arxiv.org/abs/2406.07524) | mask | masked x0 weighted CE | `subs_mask_ancestral` |
| `d3pm_mask` | [D3PM](https://arxiv.org/abs/2107.03006) | mask | D3PM VB + CE with `coef_vb=0.001`, `coef_ce=1.0` | `d3pm_mask_ancestral` |
| `d3pm_uniform` | [D3PM](https://arxiv.org/abs/2107.03006) | uniform | D3PM full VB with `coef_vb=1.0`, `coef_ce=0.0` | `d3pm_uniform_ancestral` |
| `sedd_mask` | [SEDD](https://arxiv.org/abs/2310.16834) | mask | SEDD score entropy | `sedd_analytic` |
| `sedd_uniform` | [SEDD](https://arxiv.org/abs/2310.16834) | uniform | SEDD score entropy | `sedd_analytic` |
| `duo_uniform` | [DUO](https://arxiv.org/abs/2506.10892) | uniform | DUO uniform diffusion loss | `duo_ancestral` |

Default samplers are used only when `--dlm_sampler auto`.

For `duo_uniform`, the trainer downloads the pinned DUO keep-probability table if `--duo_keep_prob_table_path` is omitted. The table stores `p_clean = P[argmax gaussian latent is the original clean token]` and converts it with `keep_prob = (vocab_size * p_clean - 1) / (vocab_size - 1)`.

For D3PM, the code uses `coef_vb * VB + coef_ce * CE`. In that convention, the [official text config](https://github.com/google-research/google-research/blob/1fa17414f56c3703d5adb3818338b6e35e0fd550/d3pm/text/configs.py#L162-L176) uses `uniform=(1.0, 0.0)` and `mask=(1.0, 0.01)`. This benchmark keeps `d3pm_mask=(0.001, 1.0)` because it produced better text under the current gate.

## Samplers

Sampler names are listed in [`sample_text.py`](speedrun_dlm/sample_text.py).

| Family | Paper | Samplers |
| --- | --- | --- |
| AR | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | top-k next-token sampling |
| SUBS | [SUBS](https://arxiv.org/abs/2406.07524) | `subs_mask_ancestral` |
| D3PM | [D3PM](https://arxiv.org/abs/2107.03006) | `d3pm_mask_ancestral`, `d3pm_uniform_ancestral`, uniform checkpoints also allow DUO Psi samplers |
| SEDD | [SEDD](https://arxiv.org/abs/2310.16834) | `sedd_euler`, `sedd_analytic` |
| DUO | [DUO](https://arxiv.org/abs/2506.10892), [Psi samplers](https://arxiv.org/abs/2602.21185) | `duo_ancestral`, `duo_greedy_tail`, `duo_psi_rescale`, `duo_psi_capped`, `duo_psi_loop` |
