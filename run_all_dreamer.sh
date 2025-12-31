#!/usr/bin/env bash

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa  # helps PyOpenGL choose the right backend

set -e  # stop on error

# ensure venv is active
source venv/bin/activate

#echo "=== Run 1: baseline ==="
#python geojacobreg.py --bisimulation_weight 0.0 --pb_curvature_weight 0.0

echo "=== Run 2: bisimulation only ==="
python geojacobreg.py --bisimulation_weight 0.05 --pullback_bisim

echo "=== Run 3: curvature only ==="
#python geojacobreg.py --bisimulation_weight 0.0 --pb_curvature_weight 0.1 --pb_curvature_projections 2 --pb_detach_features

echo "=== Run 4: both regularizers ==="
#python geojacobreg.py --bisimulation_weight 0.5 --pb_curvature_weight 0.1 --pb_curvature_projections 2 --pb_detach_features

echo "=== All runs completed. ==="
