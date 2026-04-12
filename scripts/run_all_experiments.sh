#!/bin/bash
# MergeSafe — Master Experiment Runner
# Runs ALL experiments sequentially: Qwen/SST2, Llama/SST2, Qwen/AGNews, Llama/AGNews
# Then adaptive attacks, then figure generation
# Started: Apr 12, 2026
# Expected: ~12-18 hours total

set -e
cd /Users/rono/Garage/phd/projects/mergesafe

LOG=/tmp/mergesafe_master.log
echo "=== MergeSafe Master Runner Started: $(date) ===" | tee -a $LOG

# phase 1: full matrix (4 model-dataset combos)
echo "[Phase 1] Starting Qwen/SST-2 matrix..." | tee -a $LOG
uv run python scripts/run_matrix.py --models qwen --datasets sst2 2>&1 | tee -a $LOG
echo "[Phase 1] Qwen/SST-2 DONE: $(date)" | tee -a $LOG

echo "[Phase 1] Starting Llama/SST-2 matrix..." | tee -a $LOG
uv run python scripts/run_matrix.py --models llama --datasets sst2 2>&1 | tee -a $LOG
echo "[Phase 1] Llama/SST-2 DONE: $(date)" | tee -a $LOG

echo "[Phase 1] Starting Qwen/AGNews matrix..." | tee -a $LOG
uv run python scripts/run_matrix.py --models qwen --datasets agnews 2>&1 | tee -a $LOG
echo "[Phase 1] Qwen/AGNews DONE: $(date)" | tee -a $LOG

echo "[Phase 1] Starting Llama/AGNews matrix..." | tee -a $LOG
uv run python scripts/run_matrix.py --models llama --datasets agnews 2>&1 | tee -a $LOG
echo "[Phase 1] Llama/AGNews DONE: $(date)" | tee -a $LOG

# clean merged models to save disk
echo "[Cleanup] Removing merged model caches..." | tee -a $LOG
find results/matrix/runs -name "merged" -type d -exec rm -rf {} + 2>/dev/null || true
echo "[Cleanup] Disk freed: $(df -h /Users/rono | tail -1 | awk '{print $4}')" | tee -a $LOG

# phase 2: adaptive attacks
echo "[Phase 2] Starting adaptive attacks..." | tee -a $LOG

# Spectral flattening attack
echo "[Phase 2a] Spectral flattening (alpha=0.1)..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset sst2 \
  --attack badnets \
  --mode spectral \
  --alpha 0.1 \
  --merge-method linear 2>&1 | tee -a $LOG

echo "[Phase 2a] Spectral flattening (alpha=1.0)..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset sst2 \
  --attack badnets \
  --mode spectral \
  --alpha 1.0 \
  --merge-method linear 2>&1 | tee -a $LOG

# Weight distribution matching attack
echo "[Phase 2b] Weight distribution matching (beta=0.1)..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset sst2 \
  --attack badnets \
  --mode weight_dist \
  --beta 0.1 \
  --merge-method linear 2>&1 | tee -a $LOG

echo "[Phase 2b] Weight distribution matching (beta=1.0)..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset sst2 \
  --attack badnets \
  --mode weight_dist \
  --beta 1.0 \
  --merge-method linear 2>&1 | tee -a $LOG

# Combined evasion
echo "[Phase 2c] Combined evasion (alpha=0.5, beta=0.5)..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset sst2 \
  --attack badnets \
  --mode combined \
  --alpha 0.5 \
  --beta 0.5 \
  --merge-method linear 2>&1 | tee -a $LOG

# Full sweep on Llama too
echo "[Phase 2d] Adaptive sweep on Llama..." | tee -a $LOG
uv run python scripts/run_adaptive.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset sst2 \
  --attack badnets \
  --mode combined \
  --alpha 0.5 \
  --beta 0.5 \
  --merge-method linear 2>&1 | tee -a $LOG

echo "[Phase 2] Adaptive attacks DONE: $(date)" | tee -a $LOG

# phase 3: generate figures
echo "[Phase 3] Generating figures..." | tee -a $LOG
uv run python scripts/generate_figures.py 2>&1 | tee -a $LOG
echo "[Phase 3] Figures DONE: $(date)" | tee -a $LOG

# final cleanup
echo "[Cleanup] Final merged model cleanup..." | tee -a $LOG
find results/matrix/runs -name "merged" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== MergeSafe Master Runner COMPLETE: $(date) ===" | tee -a $LOG
echo "Results: $(wc -l results/matrix/results.jsonl)" | tee -a $LOG
echo "Disk: $(df -h /Users/rono | tail -1 | awk '{print $4}') free" | tee -a $LOG
