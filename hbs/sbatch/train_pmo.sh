#!/bin/bash -l
#SBATCH --job-name=rxnflow-pmo
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# RxnFlow on one PMO task, one run on one GPU. Works as a single job or as a job
# array.
#
# This is the runner for the 50,000-call protocol the configs set: measured on an
# L40S, a step costs 1.6s uniform and 4.5s HBS, and 50,000 distinct molecules
# take 800-2,200 steps, so a HBS run is 1-3h and can pass short-unkillable's 3h
# cap. Eight 1-GPU jobs on `long` all run at once and have no such cliff.
#
# For a shorter protocol (BUDGET=10000, ~160-700 steps) prefer
# hbs/sbatch/sweep_pmo.sh instead: it takes a whole 4-GPU node on
# short-unkillable, which is non-preemptible and high priority.
#
# Submit from the repository root: the job resolves the root from RXNFLOW_ROOT
# below, but --output above is resolved against the submission directory, so
# `logs/` has to exist there.
#
#   # one run
#   sbatch hbs/sbatch/train_pmo.sh uniform                  # upstream subsampling
#   sbatch hbs/sbatch/train_pmo.sh hbs                      # hierarchical binary search
#   TASK=gsk3b SEED=3 sbatch hbs/sbatch/train_pmo.sh hbs
#
#   # 4 seeds x 2 arms = 8 runs, all at once
#   TASK=jnk3 HBS_NOISE_SIGMA=0.8 sbatch --array=0-7 hbs/sbatch/train_pmo.sh
#
#   # a second protocol on the same task: fixed gradient steps, no oracle cap.
#   # RUN_SUFFIX keeps it off the first sweep's directories (see below).
#   TASK=jnk3 HBS_NOISE_SIGMA=0.8 BUDGET=0 STEPS=1000 RUN_SUFFIX=_st1000 \
#     sbatch --array=0-7 hbs/sbatch/train_pmo.sh
#
# In array mode the arm and the seed come from SLURM_ARRAY_TASK_ID and any
# positional arm is ignored: index 0-3 are uniform seeds 1-4, 4-7 are hbs seeds
# 1-4. Anything after the arm is passed through to hbs/train.py, so the two arms
# are the same command with and without --hbs.
#
# Resources, and why they are so much smaller than hbs/sbatch/train_cdk8.sh:
#   - 6 CPUs. The PMO oracles are RDKit descriptors and pickled sklearn models;
#     measured here, a 64-molecule batch costs 27ms (celecoxib_rediscovery) to
#     512ms (gsk3b), against ~40s for roshambo2 on the same batch. There is no
#     conformer pool to feed, so this is 4 retrosynthesis workers plus the main
#     process plus one spare, and the reward is no longer what sets the wall
#     time -- generation and the loss pass are.
#   - 24G. The CDK8 runs peaked at 4.9GB RSS at 1,200 steps *including* an
#     8-process conformer pool; without it the footprint is the prepared
#     environment, the HBS reservoir (482,070 x 256 floats, ~0.5GB), the
#     oracle's forests, and the replay buffer, which is the one that grows --
#     these runs go past the 100,000-trajectory capacity the CDK8 runs never
#     reached. 24G is several times the measured peak rather than a guess.
#   - l40s. These runs need little VRAM, so by the usual rule the most abundant
#     GPU is the right one -- and it is the type the CDK8 runs used, which
#     matters because fwd/metrics.py's latency_ms is only comparable across runs
#     on the same hardware. For the same reason one run gets one GPU to itself:
#     packing runs onto a shared GPU would make those numbers meaningless.
#   - long, 12h for a run expected to take 1-3h. `long` is preemptible, which is
#     the price of having no 3h cap and no 4-GPU-per-user limit; the
#     checkpoint_every=200 in the configs caps what a preemption costs at 200
#     steps. short-unkillable is not available to a 1-GPU job at all: its QoS
#     sets MinTRES gres/gpu=4, so a job there has to take a whole node.

set -euo pipefail

STEPS="${STEPS:-4000}"
TASK="${TASK:-jnk3}"
# Oracle-call budget; 0 means unlimited, leaving --steps to end the run. STEPS
# above is only a ceiling: at 64 online samples a step and 23-64 of them distinct,
# 4,000 steps reaches 50,000 calls with room to spare, so the budget is what ends
# a run and both arms stop at the same number of scored molecules.
BUDGET="${BUDGET:-50000}"
CONFIG="${CONFIG:-hbs/configs/pmo_${TASK}.yaml}"
# The HBS arm's settings. These are the seed sweep's, not hbs/train.py's
# defaults, so the PMO arms match the CDK8 arms they will be read beside.
HBS_MIN_ROWS="${HBS_MIN_ROWS:-256}"
HBS_NOISE_SIGMA="${HBS_NOISE_SIGMA:-0.0}"
METRICS_EVERY="${METRICS_EVERY:-10}"
CAS_ROWS_PER_CELL="${CAS_ROWS_PER_CELL:-235}"
CAS_TAU="${CAS_TAU:-1.0}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # 0-3 -> uniform seeds 1-4, 4-7 -> hbs seeds 1-4.
  INDEX="$SLURM_ARRAY_TASK_ID"
  NUM_SEEDS="${NUM_SEEDS:-4}"
  if [ "$INDEX" -lt "$NUM_SEEDS" ]; then
    ARM="uniform"
    SEED=$(( INDEX + 1 ))
  else
    ARM="hbs"
    SEED=$(( INDEX - NUM_SEEDS + 1 ))
  fi
else
  ARM="${1:?usage: sbatch train_pmo.sh <uniform|hbs> [extra args for hbs/train.py]}"
  shift || true
  SEED="${SEED:-1}"
fi

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
  cas)
    # RxnFlow-CAS: clustered action sampling. rows_per_cell sets nlist per library,
    # tau the temperature of the cell distribution.
    ARM_FLAGS=(--cas --cas-rows-per-cell "$CAS_ROWS_PER_CELL" --cas-tau "$CAS_TAU")
    SUFFIX=""
    ;;
  *) echo "arm must be 'uniform', 'hbs' or 'cas', got '$ARM'" >&2; exit 2 ;;
esac

# Slurm runs a *copy* of this script from its spool directory, so the script's
# own path says nothing about where the code is. Hence an absolute default.
ROOT="${RXNFLOW_ROOT:-/network/scratch/d/diego.calanzone/ift6760b-project}"
cd "$ROOT"
test -f "$CONFIG" || { echo "no config at $ROOT/$CONFIG" >&2; exit 2; }
# Interpreter: an explicit PYTHON wins, then the environment built for this
# package, then a plain .venv. Only .venv-hbs has PyTDC; see SETUP.md.
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
# The oracle checkpoints, so a run is independent of the working directory and
# never re-downloads them onto a compute node.
export TDC_ORACLE_DIR="${TDC_ORACLE_DIR:-$ROOT/oracle}"
# sklearn and RDKit would each open a thread per core inside the reward and
# spend the time in contention; the reward is single-batch and short.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

RUN_NAME="$(basename "${CONFIG%.yaml}")"
# RUN_SUFFIX distinguishes two protocols on the same task and arm. Without it a
# second sweep over the same config writes to the first one's directories, and
# RxnFlowTrainer refuses an existing output directory, so every job fails on
# startup. Set it whenever the protocol differs from a sweep already on disk --
# e.g. RUN_SUFFIX=_st1000 for a fixed-1000-step run beside a budget-capped one.
RUN_SUFFIX="${RUN_SUFFIX:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pmo/${RUN_NAME}_${ARM}${SUFFIX}_s${SEED}${RUN_SUFFIX}}"

BUDGET_FLAGS=()
[ "$BUDGET" -gt 0 ] && BUDGET_FLAGS=(--pmo-budget "$BUDGET")

echo "=== $(date -Is) ==="
echo "arm        : $ARM"
echo "task       : $TASK"
echo "root       : $ROOT"
echo "python     : $PYTHON"
echo "config     : $CONFIG"
echo "output     : $OUTPUT_DIR"
echo "steps/seed : $STEPS / $SEED"
echo "budget     : ${BUDGET} oracle calls (0 = unlimited)"
echo "oracle dir : $TDC_ORACLE_DIR"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?} (4 retrosynthesis + main)"
echo "job        : ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

exec "$PYTHON" -m hbs.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --num-workers 4 \
  --metrics-every "$METRICS_EVERY" \
  "${BUDGET_FLAGS[@]}" \
  "${ARM_FLAGS[@]}" \
  "$@"
