# Benchmark details

## HellaSwag evaluation

We use all 10,042 examples from the [pinned HellaSwag validation split](https://huggingface.co/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76), with preprocessing from the [pinned Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/4e7e0d47f33bc71070c1d38394bafbb52b117163/lm_eval/tasks/hellaswag).

For each context and possible ending, we keep the context clean and corrupt only the ending. The model's objective supplies a recovery loss for each corruption. We average these losses, divide by the ending's character count, and select the lowest-scoring ending:

```text
score(ending) = average recovery loss / number of characters in the ending
prediction = ending with the lowest score
accuracy = number of correct predictions / 10,042
```

SUBS uses the masked-token estimator used by LLaDA. D3PM and SEDD use their corresponding discrete or continuous-time loss terms. DUO uses its Gaussian relaxation and curriculum. The exact terms are in [`evaluate_hellaswag.py`](speedrun_dlm/evaluate_hellaswag.py). Raw losses are compared only among the four endings for one checkpoint.

Evaluation uses seed 314159 and a fixed hash-based random stream. Random draws are shared across checkpoints and, where their corruption processes allow it, across recipes. They do not depend on the checkpoint hash. Context plus ending fits within 256 GPT-2 tokens; the evaluator rejects overflow rather than truncating an answer.

## Training time and target

The target is **27.5% HellaSwag accuracy averaged over ten independent trainings**. Training time sums optimizer-step durations on two nodes with four GH200 GPUs each, excluding compilation, evaluation, and checkpoint writing.

Checkpoints are saved after the first optimizer step reaching each elapsed minute. We first evaluate minutes 1, 4, 7, ..., 28, every ten minutes from minute 30, and the final checkpoint with 32 corruptions per answer. Evaluations with eight corruptions guide further checks between these points. A three-point confirmation uses 128 corruptions: the previous minute must be below the target, and the selected and following minutes above it. If these scores move the crossing, the search continues with 128 corruptions. The selected minute is the first passing minute with this confirmation and no reversal among later 128-corruption evaluations. Its time is the average of the ten recorded checkpoint times, not an average of ten individual first-crossing times.

DUO with the original curriculum passes at minute 41, falls below the target at 42, and passes at 43 and 44 with 128 corruptions, so its reported time is 43 minutes. All measurements remain in the [trace](figures/production-hellaswag-measurement-trace.png), including lower-precision estimates that fall below the target later. Nearby-target comparisons use the largest complete draw count at each measured position and the first pass that stays above the target at later measured positions; only 27.5% receives adjacent-minute confirmation.

## Records

Each [record](records/records.csv) includes the ten individual results, their average, all evaluated checkpoints for that recipe, the draw counts used, training parameters, source hashes, and reproduction commands. The [measurement trace](figures/production-hellaswag-measurement-trace.png) shows those evaluations. Recipes that do not reach a target share the last rank in threshold comparisons.

The [corruption comparison](figures/corruption-variation.csv) compares the first 32 and all 128 draws within the same evaluation. The reported standard deviations describe variation across the ten training seeds. The finish line uses their average accuracy, not a confidence bound.
