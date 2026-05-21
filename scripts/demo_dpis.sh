#!/bin/bash
# DPIS-SQ & DPIS-MI demo test script
# Usage: bash scripts/demo_dpis.sh
# Prerequisites: conda environment 'dpimagebench', GPUs with >=32GB VRAM

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dpimagebench

# Common params for minimal full-pipeline demo
# Reduce n_epochs/batch_size for quick testing; increase for production
COMMON_PARAMS="
  setup.run_type=torchrun
  pretrain.n_epochs=1
  train.n_epochs=1
  train.n_splits=1
  gen.data_num=50
  pretrain.batch_size=128
  train.batch_size=64
  train.max_physical_batch_size=64
  pretrain.fid_freq=1000000
  pretrain.snapshot_freq=1000000
  pretrain.save_freq=1000000
  train.fid_freq=1000000
  train.snapshot_freq=1000000
  train.save_freq=1000000
"

DATASET="mnist_28"
EPSILON="1.0"
GPU_DEVICES="1,2"
NPROC=2
MASTER_PORT=6030

echo "========================================"
echo "  DPIS Demo Script"
echo "  Dataset: $DATASET  Epsilon: $EPSILON"
echo "  GPUs: $GPU_DEVICES (x$NPROC)"
echo "========================================"

# ---- DPIS-SQ ----
echo ""
echo ">>> [1/2] Running DPIS-SQ demo..."
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  run.py $COMMON_PARAMS \
  -m DPIS-SQ -dn $DATASET -e $EPSILON -ed demo

echo ">>> DPIS-SQ demo completed."

# ---- DPIS-MI ----
MASTER_PORT=$((MASTER_PORT + 1))
echo ""
echo ">>> [2/2] Running DPIS-MI demo..."
CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  run.py $COMMON_PARAMS \
  -m DPIS-MI -dn $DATASET -e $EPSILON -ed demo

echo ">>> DPIS-MI demo completed."

# ---- Summary ----
echo ""
echo "========================================"
echo "  Demo Complete"
echo "========================================"
echo "Results:"
for method in dpis-sq dpis-mi; do
  latest=$(ls -td exp/$method/*demo* 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    echo "  $method: $latest"
    grep -E "best acc.*from (resnet|wrn|resnext)|average.*acc|FID.*synthetic" "$latest/stdout.txt" 2>/dev/null | tail -5
  fi
done
