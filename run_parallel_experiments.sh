#!/bin/bash

set -e

ENV="cheetah-run"
SEED=0
MAX_EPISODES=200
EVAL_INTERVAL=20

# Reduced batch size for parallel execution (2 experiments sharing 24GB VRAM)
BATCH_SIZE=48  # 48*2 = ~same memory as 64 sequential

COMMON="--env_id $ENV \
    --seed $SEED \
    --max_episodes $MAX_EPISODES \
    --eval_interval $EVAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --seq_len 50 \
    --replay_buff_capacity 400000 \
    --train_steps 50 \
    --plan_horizon 12 \
    --plan_candidates 500 \
    --plan_iters 8 \
    --plan_top_k 50"

echo "==================================="
echo "Parallel Experiment Runner"
echo "Running 2 experiments at a time"
echo "==================================="
echo ""

START_TIME=$(date +%s)

# Function to run experiment and log time
run_exp() {
    local name=$1
    shift
    echo "Starting: $name"
    python train.py $COMMON "$@" --run_name "$name" &
}

# Batch 1: Baseline experiments (2 parallel)
echo "=== Batch 1/4: Baseline experiments ==="
run_exp "planet_baseline_seed${SEED}"
run_exp "featdec_masked_seed${SEED}" --use_feature_decoder --mask_ratio 0.5 --feature_rec_scale 10.0
wait
echo "Batch 1 complete"
echo ""

# Batch 2: Simple geo experiments (2 parallel)
echo "=== Batch 2/4: Geometric experiments ==="
run_exp "geo_mu_only_seed${SEED}" --use_geo_decoder --feature_norm_mode layernorm --lambda_geo_rec 10.0
run_exp "geo_bisim_seed${SEED}" --use_geo_decoder --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 --lambda_bisim_geo 0.5 \
    --geo_knn_k 8 --geo_time_subsample 8 \
    --bisimulation_warmup 2000 --bisimulation_ramp 5000
wait
echo "Batch 2 complete"
echo ""

# Batch 3: Full system experiments (2 parallel)
echo "=== Batch 3/4: Full system experiments ==="
run_exp "geo_full_no_sigma_seed${SEED}" \
    --use_geo_decoder --use_masked_branch --geo_cube_masking \
    --geo_cube_t 2 --geo_cube_hw 8 --geo_mask_mode zero --mask_ratio 0.5 \
    --feature_norm_mode layernorm --lambda_geo_rec 10.0 --lambda_bisim_geo 0.5 \
    --geo_knn_k 8 --geo_time_subsample 8 \
    --bisimulation_warmup 2000 --bisimulation_ramp 5000

run_exp "geo_full_sigma_penalty_seed${SEED}" \
    --use_geo_decoder --use_masked_branch --geo_cube_masking \
    --geo_cube_t 2 --geo_cube_hw 8 --geo_mask_mode zero --mask_ratio 0.5 \
    --feature_norm_mode layernorm --lambda_geo_rec 10.0 --lambda_bisim_geo 0.5 \
    --lambda_sigma_plan 0.1 \
    --geo_knn_k 8 --geo_time_subsample 8 \
    --bisimulation_warmup 2000 --bisimulation_ramp 5000
wait
echo "Batch 3 complete"
echo ""

# Batch 4: Final experiment (1 alone for stability)
echo "=== Batch 4/4: Final experiment ==="
python train.py $COMMON \
    --use_geo_decoder --use_masked_branch --geo_cube_masking \
    --geo_cube_t 2 --geo_cube_hw 8 --geo_mask_mode zero --mask_ratio 0.5 \
    --feature_norm_mode layernorm --lambda_geo_rec 10.0 --lambda_bisim_geo 0.5 \
    --sigma_threshold 2.0 \
    --geo_knn_k 8 --geo_time_subsample 8 \
    --bisimulation_warmup 2000 --bisimulation_ramp 5000 \
    --run_name "geo_full_sigma_constraint_seed${SEED}"
echo "Batch 4 complete"
echo ""

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo "==================================="
echo "All experiments complete!"
echo "Total time: $((TOTAL_TIME/60))m $((TOTAL_TIME%60))s"
echo "==================================="
echo ""
echo "View results: tensorboard --logdir runs/"
