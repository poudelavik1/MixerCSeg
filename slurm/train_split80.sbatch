#!/bin/bash
# Train MixerCSeg (official model, official recipe) on the 16 Nov split_80_10_10 on the ASL cluster. From the repo folder:
#     DATA=~/split_80_10_10 OUT=~/mixercseg_runs/split80 sbatch slurm/train_split80.sbatch
# Trains, picks the best epoch on val (mIoU), scores test. Resubmitting the same command after a time-out resumes
# from last.pt. Optional: ENV_DIR (conda env, default ~/.conda/envs/mixercseg), EXTRA_ARGS (e.g. "--augment",
# "--batch-size 8 --lr 1e-3", "--epochs 30").
#SBATCH --job-name=mixercseg_s80
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=mixercseg_s80_%j.out

set -uo pipefail
: "${DATA:?Set DATA=<split_80_10_10 folder>}"
: "${OUT:?Set OUT=<run folder>}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
[ -f "$DATA/split_manifest.csv" ] || { echo "No split_manifest.csv in $DATA" >&2; exit 1; }

# Compute nodes have no conda: put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/mixercseg}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

echo "===== MixerCSeg split_80_10_10 ${SLURM_JOB_ID:-} -> $OUT on $(hostname) | $(date) ====="
echo "python : $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
python -c "import selective_scan_cuda_oflex" 2>/dev/null && echo "selective_scan kernel: ok" \
    || echo "WARNING: selective_scan CUDA kernel not built (VMamba/models/kernels/selective_scan: python setup.py install); the PyTorch scan is very slow"

python -u train_mixercseg_split80.py train --data "$DATA" --out "$OUT" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA_ARGS:-}
STATUS=$?
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
