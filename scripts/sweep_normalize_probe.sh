#!/bin/bash -l
#SBATCH --job-name=hbs-normprobe
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --partition=short-unkillable
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Does normalizing the probe let trajectory balance train HBS retrieval?
#
#   sbatch scripts/sweep_normalize_probe.sh
#   TASK=gsk3b SEEDS="1 2" sbatch scripts/sweep_normalize_probe.sh
#
# The hypothesis: a logit is probe . fingerprint, so with an unnormalized probe TB
# can cut its error by growing ||probe|| instead of rotating it. The hash sees
# direction only, so that radial component is a free sink and retrieval never
# improves. Normalizing makes a logit a pure cosine, leaving only a rotation.
#
# 2 settings x 2 seeds = 4 runs, one per GPU. Everything except
# --hbs-normalize-probe matches runs/pmo/pmo_jnk3_hbs_sig080_s4, so "off" is a
# replication of the existing baseline and the pair differs in one flag.
#
# Read it from the probe lines in each run's training.log:
#   probe norm      -- 1.0 when on; if it climbs when off, the sink is real
#   cosine best     -- probe vs the fingerprints it should retrieve
#   best row in bucket / recalled -- retrieval itself, against chance
# scripts/analyze_normalize_probe.py tabulates these.
#
# Sizing: the baseline spent its 50,000 oracle calls at step 1,260 in 95 min
# (4.55 s/step on an L40S), so the budget binds before --steps and a run fits the
# 3h cap. short-unkillable's QoS sets MinTRES gres/gpu=4, so a job there takes a
# whole node; running all four at once uses it instead of idling three GPUs.
# 24 CPUs and 96G is 6 CPUs and 24G per run, as in hbs/sbatch/sweep_pmo.sh.

set -euo pipefail

TASK="${TASK:-jnk3}"
SEEDS="${SEEDS:-1 2}"
STEPS="${STEPS:-4000}"          # not binding; the oracle budget ends the run
BUDGET="${BUDGET:-50000}"       # PMO protocol, as the baseline run used
MIN_ROWS="${MIN_ROWS:-256}"
NOISE_SIGMA="${NOISE_SIGMA:-0.8}"
PROBE_EVERY="${PROBE_EVERY:-100}"  # denser than the baseline's 200; no_grad only
METRICS_EVERY="${METRICS_EVERY:-10}"
CONFIG="${CONFIG:-hbs/configs/pmo_${TASK}.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/normalize_probe}"

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
mkdir -p logs "$RUN_ROOT"

module load cuda/12.4.1

export PYTHONUNBUFFERED=1
export TDC_ORACLE_DIR="${TDC_ORACLE_DIR:-$ROOT/oracle}"
# Four runs share 24 cores, so each one's libraries stay inside its own share.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date -Is) ==="
echo "task       : $TASK ($CONFIG)"
echo "arms       : hbs, normalize_probe off vs on"
echo "seeds      : $SEEDS"
echo "steps      : $STEPS (budget $BUDGET oracle calls binds first)"
echo "hbs        : min_rows $MIN_ROWS, noise_sigma $NOISE_SIGMA, probe_every $PROBE_EVERY"
echo "python     : $PYTHON"
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

gpu=0
pids=()
names=()
for seed in $SEEDS; do
  for mode in off on; do
    NORM_FLAG=()
    [ "$mode" = "on" ] && NORM_FLAG=(--hbs-normalize-probe)
    NAME="pmo_${TASK}_hbs_norm${mode}_s${seed}"
    OUTPUT_DIR="${RUN_ROOT}/${NAME}"
    LOG="logs/${SLURM_JOB_NAME:-hbs-normprobe}-${SLURM_JOB_ID:-local}-${NAME}.out"
    echo "--> GPU $gpu: $NAME -> $OUTPUT_DIR (log $LOG)"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" -m hbs.train \
      --config "$CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      --steps "$STEPS" \
      --seed "$seed" \
      --device cuda \
      --num-workers 4 \
      --metrics-every "$METRICS_EVERY" \
      --pmo-budget "$BUDGET" \
      --hbs \
      --hbs-min-rows "$MIN_ROWS" \
      --hbs-noise-sigma "$NOISE_SIGMA" \
      --hbs-probe-every "$PROBE_EVERY" \
      "${NORM_FLAG[@]}" > "$LOG" 2>&1 &
    pids+=("$!")
    names+=("$NAME")
    gpu=$(( gpu + 1 ))
  done
done

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
