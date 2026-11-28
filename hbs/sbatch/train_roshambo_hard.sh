#!/bin/bash -l
#SBATCH --job-name=rxn-hard
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --time=9:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --signal=B:USR1@300

# The three most out-of-distribution roshambo2 queries, both arms, three seeds:
# 18 runs as one array so Slurm starts each as a GPU frees rather than making us
# queue 18 separate submissions.
#
#   sbatch hbs/sbatch/train_roshambo_hard.sh                    # all 18
#   STEPS=10 sbatch --array=0 hbs/sbatch/train_roshambo_hard.sh # one smoke run
#   sbatch --array=6-11 hbs/sbatch/train_roshambo_hard.sh       # one target only
#
# Index layout -- target varies slowest, so --array=0-5 is one target's full pair:
#   0-5    gsm_ro7185876      6-11   irak4_8ucc      12-17  gsm_pf-06648671
# and within each block: 0-2 uniform seeds 1-3, 3-5 cas seeds 1-3.
#
# Resources, and why:
#   - 32 CPUs: 28 conformer-pool workers + 4 retrosynthesis. Sized from a
#     measured 10-step smoke run of each arm on this target at 16 CPUs (jobs
#     11891316 / 11891317), not from an estimate: ~85s per 64-molecule step, of
#     which *sampling* was 0.35s (uniform) and 0.74s (CAS). So 99% of a step is
#     conformer preparation plus roshambo scoring, it is CPU bound and
#     embarrassingly parallel, and 500 steps would take 11.8h at 16 CPUs. More
#     pool workers is the only lever; 28 of them should give ~45-50s/step and
#     ~6.5h, which also keeps a run inside the window where preemption on 'long'
#     is less likely to cost much.
#     That smoke run also settles the arms' relative cost here: 0.97x overall,
#     i.e. CAS's 2-2.8x per-step policy overhead -- real on the PMO tasks, where
#     a whole step is 1.1-5.0s -- is invisible behind an 85s reward. These runs
#     therefore compare quality, and can say nothing about CAS's efficiency.
#     Per-step cost is also very uneven (23.7s to 304.3s over ten steps) because
#     some molecules are far more expensive to embed, so the mean is what the
#     wall-clock estimate rests on, not any single step.
#   - 48G rather than train_cdk8.sh's 32G: 2 stereoisomers x 25 conformers per
#     molecule already holds far more RDKit conformers at once than a descriptor
#     oracle does, and 28 pool workers hold more of them simultaneously than 12.
#     The smoke runs peaked at 3.4GB (uniform) and 6.3GB (CAS) of GPU memory.
#   - l40s. roshambo2's CUDA backend is verified on sm_89 here, and every other
#     run in this package used l40s, so arms and targets share a GPU type. The
#     GPU is mostly idle while the CPUs build conformers, so a faster card would
#     buy little; VRAM need is ~2GB.
#   - 9h for an expected ~6.5h. A run cannot be resumed into an existing output
#     directory (see "Resuming" in SETUP.md), so chaining shorter jobs would mean
#     restarting from a checkpoint rather than continuing; checkpoint_every is 100
#     in these configs and USR1 is trapped below, so a preemption loses at most
#     100 steps and is logged as such.

set -euo pipefail

STEPS="${STEPS:-500}"
TARGETS=(gsm_ro7185876 irak4_8ucc gsm_pf-06648671)
ARMS=(uniform uniform uniform cas cas cas)
SEEDS=(1 2 3 1 2 3)

INDEX="${SLURM_ARRAY_TASK_ID:-${1:?usage: sbatch train_roshambo_hard.sh  (or pass an index to run directly)}}"
TARGET="${TARGETS[$(( INDEX / 6 ))]}"
ARM="${ARMS[$(( INDEX % 6 ))]}"
SEED="${SEEDS[$(( INDEX % 6 ))]}"

case "$ARM" in
  uniform) ARM_FLAGS=() ;;
  cas)     ARM_FLAGS=(--cas) ;;
  *) echo "bad arm '$ARM'" >&2; exit 2 ;;
esac

ROOT="${RXNFLOW_ROOT:-/network/scratch/d/diego.calanzone/ift6760b-project}"
cd "$ROOT"
CONFIG="hbs/configs/roshambo_${TARGET}.yaml"
test -f "$CONFIG" || { echo "no config at $ROOT/$CONFIG" >&2; exit 2; }

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in .venv-hbs/bin/python .venv/bin/python; do
    [ -x "$candidate" ] && PYTHON="$candidate" && break
  done
fi
test -n "$PYTHON" && test -x "$PYTHON" || {
  echo "no usable interpreter in $ROOT (.venv-hbs or .venv); see SETUP.md" >&2; exit 2; }
mkdir -p logs || true

module load cuda/12.4.1

export PYTHONUNBUFFERED=1
export ROSHAMBO_NUM_CPUS="${ROSHAMBO_NUM_CPUS:-$(( ${SLURM_CPUS_PER_TASK:-16} > 5 ? ${SLURM_CPUS_PER_TASK:-16} - 4 : 1 ))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

OUTPUT_DIR="${OUTPUT_DIR:-runs/hard/roshambo_${TARGET}_${ARM}_s${SEED}}"

echo "=== $(date -Is) ==="
echo "array idx  : $INDEX of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "target     : $TARGET"
echo "arm        : $ARM"
echo "seed       : $SEED"
echo "config     : $CONFIG"
echo "output     : $OUTPUT_DIR"
echo "steps      : $STEPS"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?} (roshambo pool $ROSHAMBO_NUM_CPUS + 4 retrosynthesis)"
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Preemption on 'long' arrives as USR1 300s before the kill. Log it so a short
# run in the output is distinguishable from a crash when the results are read.
trap 'echo "[preempted] $(date -Is) at $OUTPUT_DIR" >&2' USR1

"$PYTHON" -m hbs.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --num-workers 4 \
  "${ARM_FLAGS[@]}" \
  "$@" &
wait $!
echo "finished=$(date -Is) rc=$?"
