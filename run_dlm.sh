#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
OUTPUT_DIR=${OUTPUT_DIR:-results/dlm_run}
DLM_OBJECTIVE="${DLM_OBJECTIVE:-${OBJECTIVE:-subs_mask}}"
DLM_NOISE_SCHEDULE="${DLM_NOISE_SCHEDULE:-${NOISE_SCHEDULE:-}}"
NOISE_SCHEDULE_ARGS=()
if [[ -n "$DLM_NOISE_SCHEDULE" ]]; then
  NOISE_SCHEDULE_ARGS+=(--noise_schedule "$DLM_NOISE_SCHEDULE")
fi
D3PM_COEFF_ARGS=()
if [[ -n "${DLM_D3PM_VB_COEFF:-}" ]]; then
  D3PM_COEFF_ARGS+=(--d3pm_vb_coeff "$DLM_D3PM_VB_COEFF")
fi
if [[ -n "${DLM_D3PM_CE_COEFF:-}" ]]; then
  D3PM_COEFF_ARGS+=(--d3pm_ce_coeff "$DLM_D3PM_CE_COEFF")
fi
CHECKPOINT_STEPS_ARGS=()
if [[ -n "${CHECKPOINT_STEPS:-}" ]]; then
  CHECKPOINT_STEPS_ARGS+=(--checkpoint_steps "$CHECKPOINT_STEPS")
fi
DUO_ARGS=(
  --duo_curriculum_mode "${DLM_DUO_CURRICULUM_MODE:-dense_softmax}"
  --duo_keep_prob_table_path "${DLM_DUO_KEEP_PROB_TABLE_PATH:-}"
  --duo_top_k "${DLM_DUO_TOP_K:-32}"
  --duo_log_noise_ratio_min "${DLM_DUO_LOG_NOISE_RATIO_MIN:--3.55}"
  --duo_log_noise_ratio_max "${DLM_DUO_LOG_NOISE_RATIO_MAX:--1.85}"
  --duo_softmax_temperature_log10_start "${DLM_DUO_SOFTMAX_TEMPERATURE_LOG10_START:--3.0}"
  --duo_softmax_temperature_log10_end "${DLM_DUO_SOFTMAX_TEMPERATURE_LOG10_END:--3.0}"
  --duo_curriculum_start "${DLM_DUO_CURRICULUM_START:-0}"
  --duo_curriculum_end "${DLM_DUO_CURRICULUM_END:-500000}"
)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m speedrun_dlm.train_dlm \
  --model "${MODEL:-d12}" \
  --batch_size "${BATCH_SIZE:-50}" \
  --sequence_length "${SEQUENCE_LENGTH:-1024}" \
  --total_batch_size "${TOTAL_BATCH_SIZE:-409600}" \
  --num_iterations "${NUM_ITERATIONS:-5040}" \
  --num_checkpoints "${NUM_CHECKPOINTS:-1}" \
  "${CHECKPOINT_STEPS_ARGS[@]}" \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8 \
  --objective "$DLM_OBJECTIVE" \
  "${NOISE_SCHEDULE_ARGS[@]}" \
  --continuous_time_eps "${DLM_CONTINUOUS_TIME_EPS:-1e-3}" \
  --geometric_noise_level_min "${DLM_GEOMETRIC_NOISE_LEVEL_MIN:-0.0001}" \
  --geometric_noise_level_max "${DLM_GEOMETRIC_NOISE_LEVEL_MAX:-20.0}" \
  --num_diffusion_steps "${DLM_NUM_DIFFUSION_STEPS:-1000}" \
  --cond_dim "${DLM_COND_DIM:-128}" \
  --dropout "${DLM_DROPOUT:-0.1}" \
  --warmup_iters "${WARMUP_ITERS:-384}" \
  --seed "${SEED:-1337}" \
  --val_loss_every "${VAL_LOSS_EVERY:-5040}" \
  --val_max_steps "${VAL_MAX_STEPS:-8}" \
  --output_dir "$OUTPUT_DIR" \
  "${D3PM_COEFF_ARGS[@]}" \
  "${DUO_ARGS[@]}" \
  "$@"
