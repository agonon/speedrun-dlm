#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
OUTPUT_DIR=${OUTPUT_DIR:-results/ar_run}

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m speedrun_dlm.train_ar \
  --model "${MODEL:-d12}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --sequence_length "${SEQUENCE_LENGTH:-1024}" \
  --total_batch_size "${TOTAL_BATCH_SIZE:-262144}" \
  --num_iterations "${NUM_ITERATIONS:-500}" \
  --num_checkpoints "${NUM_CHECKPOINTS:-1}" \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --warmup_iters "${WARMUP_ITERS:-320}" \
  --seed "${SEED:-1337}" \
  --val_loss_every "${VAL_LOSS_EVERY:-500}" \
  --val_max_steps "${VAL_MAX_STEPS:-8}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
