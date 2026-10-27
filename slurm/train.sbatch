#!/bin/bash
#SBATCH --job-name=ploutos-train
#SBATCH --account=cv
#SBATCH --gres=gpu:1
#SBATCH --exclude=node089        # hangs jobs silently (seen on the thesis runs)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# usage: sbatch slurm/train.sbatch <sft|dtw> [extra args...]
LOSS=${1:-sft}; shift || true

source "${PLOUTOS_VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate"
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1
mkdir -p logs

# same requeue setup as the thesis - USR1 arrives 5 min before the time limit,
# requeue and the script picks up from the latest checkpoint-*
trap 'echo "[slurm] USR1 -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

nvidia-smi --query-gpu=name,memory.total --format=csv
python train/sft_train.py --max-len 1024 --loss "$LOSS" "$@" &
wait
