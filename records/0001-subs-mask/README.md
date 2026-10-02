# SUBS

Ten independent trainings, seeds 1–10, on two nodes of four GH200 GPUs.
At the recorded position, average HellaSwag accuracy is **27.518%**,
which reaches the 27.5% target. Average training time is
**12.003 minutes**, after
**1.3608B tokens**.

Each checkpoint is evaluated on all 10,042 validation examples, with
128 corruptions per ending.

- [Training times, processed tokens, scores, and checkpoint hashes](training_runs.jsonl)
- [Average result](hellaswag.json)
- [All measured checkpoints and draw counts](measurements.csv)
- [Training configuration and source hashes](recipe.json)

## Reproduce

Prepare the pinned data as described in the [README](../../README.md#run).
The following prints the complete training command; add `--execute` to run it.
Repeat for seeds 1–10. On two nodes, set `NNODES=2`, `NPROC_PER_NODE=4`,
`NODE_RANK=0` or `1`, and `MASTER_ADDR` to the first node, then launch on both.

```bash
python records/reproduce.py records/0001-subs-mask \
  --seed 1 --output results/subs_mask/seed1

python -m speedrun_dlm.evaluate_hellaswag path/to/checkpoint.pt \
  --data data/hellaswag-validation.jsonl.gz --output results/subs_mask/seed1/hellaswag \
  --mc-samples 128 --prefixes 32,128 \
  --sampling-scheme randomized_lattice --seed 314159 --batch-size 32 \
  --max-length 256 --duo-batch-size 4 --duo-keep-prob-table data/duo-gpt2.pkl
```

To regenerate this record from the included measurement table:

```bash
python records/make_record.py --entry subs_mask --position minute:012 \
  --measurements figures/checkpoint-evaluations.csv --record-dir results/rebuilt-0001-subs-mask
```
