#!/bin/bash -l
#SBATCH --job-name=pmo-sweep
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --partition=short-unkillable
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# One arm of a PMO seed sweep: 4 seeds in parallel, one per GPU, on one node.
#
#   sbatch hbs/sbatch/sweep_pmo.sh uniform              # upstream subsampling
#   sbatch hbs/sbatch/sweep_pmo.sh hbs                  # hierarchical binary search
#   TASK=gsk3b sbatch hbs/sbatch/sweep_pmo.sh hbs
#   HBS_NOISE_SIGMA=0.8 sbatch hbs/sbatch/sweep_pmo.sh hbs
#
# Submit both arms at once for the full 4 x 2 sweep. The short-partition QoS caps
# a user at 4 GPUs, so the second job starts as the first finishes -- which is
# the intended shape, not a queueing accident: two waves of four.
#
#     sbatch hbs/sbatch/sweep_pmo.sh uniform
#     HBS_NOISE_SIGMA=0.8 sbatch hbs/sbatch/sweep_pmo.sh hbs
#
# This script is right for a protocol that fits in 3h. Measured on an L40S, a
# step costs 1.6s uniform and 4.5s HBS, so that is roughly BUDGET=10000. The
# configs set 50000, which can take a HBS run past 3h; use
# hbs/sbatch/train_pmo.sh --array for that, and pass BUDGET explicitly here.
#
# Why a whole node rather than eight 1-GPU jobs:
#   - short-unkillable is non-preemptible and high priority (PriorityTier 1000),
#     but its QoS sets MinTRES gres/gpu=4: a job there *must* take all four GPUs
#     of a node. Taking them and running four seeds at once uses the allocation
#     fully instead of idling three GPUs.
#   - A PMO run is short -- a 10,000-call budget is ~200 steps -- so two
#     non-preemptible waves of four finish sooner, and more predictably, than
#     eight preemptible jobs on `long` that each queue on their own.
#   - Each run still gets a GPU to itself. fwd/metrics.py records per-trajectory
#     latency_ms, which is only comparable between runs that were not sharing a
#     device, so packing two runs onto one GPU would corrupt a measurement the
#     project reports.
#
# Resources: 24 CPUs and 96G, i.e. 6 CPUs and 24G per run, the same per-run
# allocation as hbs/sbatch/train_pmo.sh and for the same reasons (no conformer
# pool; 4 retrosynthesis workers plus the main process; a replay buffer that
# grows to capacity). An l40s node has 48 CPUs, so this leaves headroom rather
# than claiming cores nothing would use.

set -euo pipefail

ARM="${1:?usage: sbatch sweep_pmo.sh <uniform|hbs> [extra args for hbs/train.py]}"
shift || true

TASK="${TASK:-jnk3}"
STEPS="${STEPS:-1200}"
BUDGET="${BUDGET:-10000}"  # not the configs' 50000: see the note above
NUM_SEEDS="${NUM_SEEDS:-4}"
FIRST_SEED="${FIRST_SEED:-1}"
CONFIG="${CONFIG:-hbs/configs/pmo_${TASK}.yaml}"
HBS_MIN_ROWS="${HBS_MIN_ROWS:-256}"
HBS_NOISE_SIGMA="${HBS_NOISE_SIGMA:-0.0}"
METRICS_EVERY="${METRICS_EVERY:-10}"
CPUS_PER_RUN="${CPUS_PER_RUN:-6}"

case "$ARM" in
  uniform)
    ARM_FLAGS=()
    SUFFIX=""
    ;;
  hbs)
    ARM_FLAGS=(--hbs --hbs-min-rows "$HBS_MIN_ROWS")
    SUFFIX=""
    # Reported as sigNNN in the run name, as in runs/seeds, so a σ>0 sweep does
    # not overwrite the σ=0 one. awk rather than bc: bc is not guaranteed on a
    # compute node, and a missing comparison tool must not be allowed to quietly
    # drop --hbs-noise-sigma and mislabel the run as the σ=0 arm.
    SIGMA_POSITIVE=$(awk -v s="$HBS_NOISE_SIGMA" 'BEGIN{ if (s+0 > 0) print 1; else print 0 }')
    if [ "$SIGMA_POSITIVE" = "1" ]; then
      ARM_FLAGS+=(--hbs-noise-sigma "$HBS_NOISE_SIGMA")
      SUFFIX=$(awk -v s="$HBS_NOISE_SIGMA" 'BEGIN{ printf "_sig%03.0f", s*100 }')
    elif [ "$SIGMA_POSITIVE" != "0" ]; then
      echo "could not evaluate HBS_NOISE_SIGMA='$HBS_NOISE_SIGMA'" >&2; exit 2
    fi
    ;;
  *) echo "arm must be 'uniform' or 'hbs', got '$ARM'" >&2; exit 2 ;;
esac

ROOT="${RXNFLOW_ROOT:-/network/scratch/d/diego.calanzone/ift6760b-project}"
cd "$ROOT"
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
export TDC_ORACLE_DIR="${TDC_ORACLE_DIR:-$ROOT/oracle}"
# Four runs share the node's cores, so each one's libraries must stay inside its
# own share or they would all oversubscribe the same 24 CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

RUN_NAME="$(basename "${CONFIG%.yaml}")"
RUN_ROOT="${RUN_ROOT:-runs/pmo}"
# See hbs/sbatch/train_pmo.sh: distinguishes two protocols on the same task, so
# the second sweep does not land on the first one's directories.
RUN_SUFFIX="${RUN_SUFFIX:-}"

BUDGET_FLAGS=()
[ "$BUDGET" -gt 0 ] && BUDGET_FLAGS=(--pmo-budget "$BUDGET")

echo "=== $(date -Is) ==="
echo "arm        : $ARM$SUFFIX"
echo "task       : $TASK"
echo "config     : $CONFIG"
echo "seeds      : $FIRST_SEED..$(( FIRST_SEED + NUM_SEEDS - 1 )) (one GPU each, in parallel)"
echo "steps      : $STEPS"
echo "budget     : ${BUDGET} oracle calls (0 = unlimited)"
echo "python     : $PYTHON"
echo "oracle dir : $TDC_ORACLE_DIR"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?} total, $CPUS_PER_RUN per run"
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

pids=()
names=()
for offset in $(seq 0 $(( NUM_SEEDS - 1 ))); do
  SEED=$(( FIRST_SEED + offset ))
  OUTPUT_DIR="${RUN_ROOT}/${RUN_NAME}_${ARM}${SUFFIX}_s${SEED}${RUN_SUFFIX}"
  LOG="logs/${SLURM_JOB_NAME:-pmo-sweep}-${SLURM_JOB_ID:-local}-${ARM}${SUFFIX}-s${SEED}.out"
  echo "--> GPU $offset: seed $SEED -> $OUTPUT_DIR (log $LOG)"
  # One run per GPU: CUDA_VISIBLE_DEVICES makes each process see exactly one, so
  # torch's "cuda" is that device and nothing has to be told which.
  CUDA_VISIBLE_DEVICES="$offset" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  "$PYTHON" -m hbs.train \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --device cuda \
    --num-workers 4 \
    --metrics-every "$METRICS_EVERY" \
    "${BUDGET_FLAGS[@]}" \
    "${ARM_FLAGS[@]}" \
    "$@" > "$LOG" 2>&1 &
  pids+=("$!")
  names+=("${ARM}${SUFFIX}_s${SEED}")
done

# Wait for all four and report each one's exit status, so one failed seed is
# visible in the job's own log rather than only in its run directory.
status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[ok]   ${names[$index]}"
  else
    code=$?
    echo "[FAIL] ${names[$index]} exited $code"
    status=1
  fi
done
echo "=== $(date -Is) done, status $status ==="
exit "$status"
