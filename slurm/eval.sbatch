#!/bin/bash
#SBATCH --job-name=ploutos-eval
#SBATCH --account=cv
#SBATCH --gres=gpu:1
#SBATCH --exclude=node089        # hangs jobs silently (seen on the thesis runs)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# usage: sbatch slurm/eval.sbatch <adapter_dir> [extra args...]
ADAPTER=$1; shift || true

source "${PLOUTOS_VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate"
cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export PYTHONUNBUFFERED=1
mkdir -p logs

# per-row predictions are saved next to the adapter. --resume uses the --dump
# file to skip rows already done, so resubmitting carries on where it stopped
nvidia-smi --query-gpu=name,memory.total --format=csv
python train/eval_student.py --adapter "$ADAPTER" \
  --test data/splits/test.jsonl --stratified 300 --seed 42 --resume \
  --dump "${ADAPTER%/}/eval_test_strat300_seed42.jsonl" "$@"
