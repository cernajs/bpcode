#!/bin/bash


set -e

# ============ CONFIGURATION ============
ENV="cheetah-run"
SEED=0

# Fast iteration settings (adjust for full runs)
MAX_EPISODES=200          # Reduce for quick ablations (use 1000+ for full runs)
EVAL_INTERVAL=20          # Less frequent evaluation

# Hardware-optimized settings for RTX 3090 + 16GB RAM
BATCH_SIZE=64             # Increased (RTX 3090 can handle this)
SEQ_LEN=50                # Standard
REPLAY_CAPACITY=500000    # Reduced for 16GB RAM (default 1M)
TRAIN_STEPS=50            # Steps per collect interval

# Planning - faster with slightly reduced quality
PLAN_HORIZON=12
PLAN_CANDIDATES=500       # Reduced from 1000 (still good quality)
PLAN_ITERS=8              # Reduced from 10
PLAN_TOP_K=50             # Reduced from 100

# Geometric settings
GEO_KNN_K=8
GEO_TIME_SUBSAMPLE=8      # Reduced from 16 for faster bisim

# Common args for all experiments
COMMON_ARGS="--env_id $ENV \
    --seed $SEED \
    --max_episodes $MAX_EPISODES \
    --eval_interval $EVAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --replay_buff_capacity $REPLAY_CAPACITY \
    --train_steps $TRAIN_STEPS \
    --plan_horizon $PLAN_HORIZON \
    --plan_candidates $PLAN_CANDIDATES \
    --plan_iters $PLAN_ITERS \
    --plan_top_k $PLAN_TOP_K"

echo "==================================="
echo "Geometric World Model Experiments"
echo "==================================="
echo "Hardware: RTX 3090 / Ryzen 5 5600 / 16GB RAM"
echo "Batch size: $BATCH_SIZE | Episodes: $MAX_EPISODES"
echo "==================================="
echo ""

# Track total time
START_TIME=$(date +%s)

# ============================================================
# Experiment 1: Baseline PlaNet (No Geometric Features)
# ============================================================
echo "[1/7] Running baseline PlaNet..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --run_name planet_baseline_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 2: FeatureDecoder with masking (no pullback)
# ============================================================
echo "[2/7] Running masked feature decoder (no pullback)..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_feature_decoder \
    --mask_ratio 0.5 \
    --feature_rec_scale 10.0 \
    --run_name featdec_masked_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 3: Geometric μ-only (no masking, no bisim)
# ============================================================
echo "[3/7] Running geometric decoder (μ-only, no masking)..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_geo_decoder \
    --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 \
    --run_name geo_mu_only_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 4: Geometric μ-bisim (no masking)
# ============================================================
echo "[4/7] Running geometric μ-bisim (no masking)..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_geo_decoder \
    --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 \
    --lambda_bisim_geo 0.5 \
    --geo_knn_k $GEO_KNN_K \
    --geo_time_subsample $GEO_TIME_SUBSAMPLE \
    --bisimulation_warmup 2000 \
    --bisimulation_ramp 5000 \
    --run_name geo_bisim_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 5: Full System (masking + μ-bisim, no σ-planning)
# ============================================================
echo "[5/7] Running full geometric system (no σ-planning)..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_geo_decoder \
    --use_masked_branch \
    --geo_cube_masking \
    --geo_cube_t 2 \
    --geo_cube_hw 8 \
    --geo_mask_mode zero \
    --mask_ratio 0.5 \
    --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 \
    --lambda_bisim_geo 0.5 \
    --geo_knn_k $GEO_KNN_K \
    --geo_time_subsample $GEO_TIME_SUBSAMPLE \
    --bisimulation_warmup 2000 \
    --bisimulation_ramp 5000 \
    --run_name geo_full_no_sigma_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 6: Full System with σ-penalty planning
# ============================================================
echo "[6/7] Running full system with σ-penalty planning..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_geo_decoder \
    --use_masked_branch \
    --geo_cube_masking \
    --geo_cube_t 2 \
    --geo_cube_hw 8 \
    --geo_mask_mode zero \
    --mask_ratio 0.5 \
    --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 \
    --lambda_bisim_geo 0.5 \
    --lambda_sigma_plan 0.1 \
    --geo_knn_k $GEO_KNN_K \
    --geo_time_subsample $GEO_TIME_SUBSAMPLE \
    --bisimulation_warmup 2000 \
    --bisimulation_ramp 5000 \
    --run_name geo_full_sigma_penalty_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Experiment 7: Full System with σ-constraint planning
# ============================================================
echo "[7/7] Running full system with σ-constraint planning..."
EXP_START=$(date +%s)
python train.py $COMMON_ARGS \
    --use_geo_decoder \
    --use_masked_branch \
    --geo_cube_masking \
    --geo_cube_t 2 \
    --geo_cube_hw 8 \
    --geo_mask_mode zero \
    --mask_ratio 0.5 \
    --feature_norm_mode layernorm \
    --lambda_geo_rec 10.0 \
    --lambda_bisim_geo 0.5 \
    --sigma_threshold 2.0 \
    --geo_knn_k $GEO_KNN_K \
    --geo_time_subsample $GEO_TIME_SUBSAMPLE \
    --bisimulation_warmup 2000 \
    --bisimulation_ramp 5000 \
    --run_name geo_full_sigma_constraint_seed${SEED}
echo "  Completed in $(($(date +%s) - EXP_START))s"
echo ""

# ============================================================
# Summary
# ============================================================
TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MINS=$((TOTAL_TIME / 60))
TOTAL_SECS=$((TOTAL_TIME % 60))

echo "==================================="
echo "All experiments complete!"
echo "Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "==================================="
echo ""
echo "View results with:"
echo "  tensorboard --logdir runs/"
echo ""
echo "Quick comparison (eval returns):"
echo "  for d in runs/*/; do echo \"\$d\"; done"
echo ""

# ============================================================
# FULL RUN SETTINGS (uncomment for production runs)
# ============================================================
# For full runs, change these at the top:
#   MAX_EPISODES=1000
#   EVAL_INTERVAL=10
#   PLAN_CANDIDATES=1000
#   PLAN_ITERS=10
#   PLAN_TOP_K=100
#   GEO_TIME_SUBSAMPLE=16
#   bisimulation_warmup 5000
#   bisimulation_ramp 10000
