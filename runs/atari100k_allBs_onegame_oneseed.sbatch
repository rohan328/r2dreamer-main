#!/bin/bash
#SBATCH --job-name=atari100k_6run
#SBATCH --partition=gpuql
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# One array task at a time avoids two jobs landing on the same node when
# Slurm cannot reserve a whole node (no --exclusive on this cluster).
# Raise % (e.g. %3) only if you confirm each task gets its own GPU cleanly.
#SBATCH --array=0-5%4
#SBATCH --output=logs/slurm-%x-%A_%a.out

set -euo pipefail

# Optional: activate your environment here if needed.
# source ~/.bashrc
# conda activate dreamer

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Unbuffered python output so logs are realtime in Slurm output files.
export PYTHONUNBUFFERED=1
# Replay on CPU avoids multi-GB GPU residency for LazyTensorStorage (max_size 5e5).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Sweep settings ----------
DATE="$(date +%Y%m%d)"
SEED_START=0
SEED_END=0
SEED_STEP=100

BACKBONES=(
  "gru"
  "s4"
  "s5"
  "mamba2"
  "transformer"
  "transformer_cpc"
)

TASKS=(
  "atari_alien"


)

SEEDS_PER_TASK=$(( (SEED_END - SEED_START) / SEED_STEP + 1 ))
TASKS_PER_BACKBONE=${#TASKS[@]}
RUNS_PER_BACKBONE=$(( TASKS_PER_BACKBONE * SEEDS_PER_TASK ))
TOTAL_RUNS=$(( ${#BACKBONES[@]} * RUNS_PER_BACKBONE ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is intended to run as a Slurm array job."
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_RUNS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, expected [0, $((TOTAL_RUNS - 1))]"
  exit 1
fi

idx=${SLURM_ARRAY_TASK_ID}
backbone_idx=$(( idx / RUNS_PER_BACKBONE ))
rem=$(( idx % RUNS_PER_BACKBONE ))
task_idx=$(( rem / SEEDS_PER_TASK ))
seed_idx=$(( rem % SEEDS_PER_TASK ))

backbone="${BACKBONES[$backbone_idx]}"
task="${TASKS[$task_idx]}"
seed=$(( SEED_START + seed_idx * SEED_STEP ))

run_name="${DATE}_${backbone}_${task#atari_}_${seed}"
run_logdir="logdir/${run_name}"

echo "Starting Atari100k array task"
echo "Job: ${SLURM_JOB_ID:-na}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_JOB_NODELIST:-unknown-node}"
echo "Config: backbone=${backbone} task=${task} seed=${seed}"
echo "Logdir: ${run_logdir}"

python -u train.py \
  env=atari100k \
  env.task="${task}" \
  backbone="${backbone}" \
  seed="${seed}" \
  model=size200M \
  model.compile=True \
  device=cuda:0 \
  buffer.storage_device=cuda:0 \
  logdir="${run_logdir}"

echo "Completed ArrayTask=${SLURM_ARRAY_TASK_ID}"
