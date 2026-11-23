#!/bin/bash -l
#SBATCH --job-name=rxnflow-cdk8
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# One RxnFlow run against the CDK8 roshambo2 target, on a single GPU.
#
# Submit from the baseline root (this directory's ../..): the job itself resolves
# the root from RXNFLOW_ROOT below, but --output above is resolved against the
# submission directory, so `logs/` has to exist there.
#
#   sbatch hbs/sbatch/train_cdk8.sh uniform            # upstream action subsampling
#   sbatch hbs/sbatch/train_cdk8.sh hbs                # hierarchical binary search
#   STEPS=2500 SEED=1 sbatch hbs/sbatch/train_cdk8.sh hbs --hbs-depth 10
#   CONFIG=hbs/configs/cdk8_enamine.yaml sbatch hbs/sbatch/train_cdk8.sh hbs
#
# Anything after the arm is passed through to hbs/train.py, so the two arms are
# the same command with and without --hbs.
#
# Resources, and why:
#   - 16 CPUs, 12 of them the reward's conformer pool and 4 RxnFlow's
#     retrosynthesis workers. Conformer preparation is the dominant cost of a
#     step by a wide margin and it is CPU bound, so this sets the wall time.
#     Measured here on 4 pool workers: ~110s per 64-molecule step, of which
#     generation was 0.4s. RxnFlow's three-block molecules are much more
#     expensive to embed than the small molecules the parent package profiled
#     (2 stereoisomers x 25 ETKDGv3 conformers + MMFF each), so do not expect
#     that package's 3.4s/batch here. num_confs and num_isomers are fixed at the
#     published values, so the CPU count is the only knob.
#   - 32G. Scoring a batch as 2 stereoisomers x 25 conformers holds far more
#     RDKit conformers at once than a descriptor oracle does; the parent
#     package's roshambo2 runs peaked at ~3.6GB on the Python side with
#     roshambo2 itself resident at ~2GB, and a wider pool holds more at once.
#   - l40s. roshambo2's CUDA backend is verified on sm_89 here, and l40s is what
#     the other baselines in this package used, so the arms share a GPU type.
#   - 72h for 2,500 steps: ~24-35h expected at 12 pool workers, with margin for
#     a batch of awkward conformers. Not preemptible, and a restart would need a
#     fresh output directory anyway (see "Resuming" in SETUP.md).

set -euo pipefail

ARM="${1:?usage: sbatch train_cdk8.sh <uniform|hbs> [extra args for hbs/train.py]}"
shift || true
STEPS="${STEPS:-2500}"
SEED="${SEED:-1}"
CONFIG="${CONFIG:-hbs/configs/cdk8.yaml}"

case "$ARM" in
  uniform) ARM_FLAGS=() ;;
  hbs)     ARM_FLAGS=(--hbs) ;;
  *) echo "arm must be 'uniform' or 'hbs', got '$ARM'" >&2; exit 2 ;;
esac

# Slurm runs a *copy* of this script from its spool directory, so the script's
# own path says nothing about where the code is. Hence an absolute default.
ROOT="${RXNFLOW_ROOT:-/network/scratch/d/diego.calanzone/ift6760b-project}"
cd "$ROOT"
test -f "$CONFIG" || { echo "no config at $ROOT/$CONFIG" >&2; exit 2; }
# Interpreter: an explicit PYTHON wins, then the environment built for this package,
# then a plain .venv. This checkout already had a .venv (Python 3.13, no roshambo2)
# for catalog preparation, so the two are kept apart rather than one overwriting
# the other.
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
# Conformer pool, sized to the allocation rather than to a fixed default, minus
# the four retrosynthesis workers requested below.
export ROSHAMBO_NUM_CPUS="${ROSHAMBO_NUM_CPUS:-$(( ${SLURM_CPUS_PER_TASK:-16} > 5 ? ${SLURM_CPUS_PER_TASK:-16} - 4 : 1 ))}"
# Pool workers inherit this; left unbounded, each would open a thread per core
# and preparation would spend its time in contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Named after the config, so runs on different catalogs cannot collide:
# cdk8.yaml -> runs/cdk8_<arm>_s<seed>, cdk8_enamine.yaml -> runs/cdk8_enamine_...
RUN_NAME="$(basename "${CONFIG%.yaml}")"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}_${ARM}_s${SEED}}"

echo "=== $(date -Is) ==="
echo "arm        : $ARM"
echo "root       : $ROOT"
echo "python     : $PYTHON"
echo "config     : $CONFIG"
echo "output     : $OUTPUT_DIR"
echo "steps/seed : $STEPS / $SEED"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?} (roshambo pool $ROSHAMBO_NUM_CPUS + 4 retrosynthesis)"
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

exec "$PYTHON" -m hbs.train \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --num-workers 4 \
  "${ARM_FLAGS[@]}" \
  "$@"
