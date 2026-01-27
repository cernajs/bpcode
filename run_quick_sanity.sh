#!/bin/bash

# QUICK SANITY CHECK - Run all configurations in ~10-15 minutes total
# Use this to verify everything works before running full experiments

set -e

ENV="cheetah-run"  # Or use Pendulum-v1 if dm_control not installed
SEED=0

# Ultra-fast settings for sanity checking
COMMON="--env_id $ENV \
    --seed $SEED \
    --max_episodes 5 \
    --seed_episodes 2 \
    --eval_interval 3 \
    --batch_size 32 \
    --seq_len 30 \
    --replay_buff_capacity 50000 \
    --train_steps 10 \
    --collect_interval 50 \
    --plan_horizon 8 \
    --plan_candidates 200 \
    --plan_iters 5 \
    --plan_top_k 20"

echo "============================================"
echo "Quick Sanity Check (~10-15 min total)"
echo "============================================"
echo ""

START=$(date +%s)

echo "[1/5] Baseline..."
python train.py $COMMON --run_name sanity_baseline

echo "[2/5] Feature Decoder..."
python train.py $COMMON --use_feature_decoder --mask_ratio 0.5 --run_name sanity_featdec

echo "[3/5] Geo Decoder..."
python train.py $COMMON --use_geo_decoder --lambda_geo_rec 10.0 --run_name sanity_geo

echo "[4/5] Geo + Masked Branch..."
python train.py $COMMON --use_geo_decoder --use_masked_branch --geo_cube_masking \
    --lambda_geo_rec 10.0 --run_name sanity_geo_masked

echo "[5/5] Full System..."
python train.py $COMMON --use_geo_decoder --use_masked_branch --geo_cube_masking \
    --lambda_geo_rec 10.0 --lambda_bisim_geo 0.1 --lambda_sigma_plan 0.05 \
    --geo_knn_k 4 --geo_time_subsample 4 --run_name sanity_full

ELAPSED=$(($(date +%s) - START))
echo ""
echo "============================================"
echo "Sanity check complete in $((ELAPSED/60))m $((ELAPSED%60))s"
echo "============================================"
echo ""
echo "If all passed, run full experiments with:"
echo "  bash run_geometric_experiments.sh"
