#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" -m speedrun_dlm.train_dlm \
  --model "${MODEL:-d12}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --sequence_length "${SEQUENCE_LENGTH:-1024}" \
  --total_batch_size "${TOTAL_BATCH_SIZE:-524288}" \
  --num_iterations "${NUM_ITERATIONS:-6250}" \
  --num_checkpoints "${NUM_CHECKPOINTS:-1}" \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --objective "${DLM_OBJECTIVE:-subs_mask}" \
  --continuous_time_eps "${DLM_CONTINUOUS_TIME_EPS:-1e-3}" \
  --num_diffusion_steps "${DLM_NUM_DIFFUSION_STEPS:-1000}" \
  --cond_dim "${DLM_COND_DIM:-128}" \
  --dropout "${DLM_DROPOUT:-0.0}" \
  --warmup_iters "${WARMUP_ITERS:-384}" \
  --seed "${SEED:-1}" \
  --val_loss_every "${VAL_LOSS_EVERY:-6250}" \
  --val_max_steps "${VAL_MAX_STEPS:-8}" \
  --duo_curriculum_mode "${DLM_DUO_CURRICULUM_MODE:-dense_softmax}" \
  --duo_keep_prob_table_path "${DLM_DUO_KEEP_PROB_TABLE_PATH:-}" \
  --duo_top_k "${DLM_DUO_TOP_K:-32}" \
  --output_dir "${OUTPUT_DIR:-results/dlm_run}" \
  "$@"
