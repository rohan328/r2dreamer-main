#!/bin/bash
#SBATCH --job-name=atari100k_full
#SBATCH --partition=gpuql
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-779%20
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Optional: activate your environment here if needed.
# source ~/.bashrc
# conda activate dreamer

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Unbuffered python output so logs are realtime in Slurm output files.
export PYTHONUNBUFFERED=1

# ---------- Sweep settings ----------
DATE="$(date +%Y%m%d)"
SEED_START=0
SEED_END=400
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
  "atari_amidar"
  "atari_assault"
  "atari_asterix"
  "atari_bank_heist"
  "atari_battle_zone"
  "atari_boxing"
  "atari_breakout"
  "atari_chopper_command"
  "atari_crazy_climber"
  "atari_demon_attack"
  "atari_freeway"
  "atari_frostbite"
  "atari_gopher"
  "atari_hero"
  "atari_jamesbond"
  "atari_kangaroo"
  "atari_krull"
  "atari_kung_fu_master"
  "atari_ms_pacman"
  "atari_pong"
  "atari_private_eye"
  "atari_qbert"
  "atari_road_runner"
  "atari_seaquest"
  "atari_up_n_down"
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
