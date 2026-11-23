#!/bin/bash
#SBATCH --job-name=prep-enamine-290k
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/prep-enamine-290k-%j.out
#SBATCH --error=logs/prep-enamine-290k-%j.err

# Prepare a synthon environment from the uncapped Enamine building-block catalog
# using the `explore` reaction templates (26 bimolecular + 4 unimolecular, with
# reaction-reactant compatibility constraints).
#
# The six preparation stages each drop a *.done marker in the env dir, so a
# timed-out or failed run resumes from the last completed stage on resubmission.

set -euo pipefail

PROJECT=/network/scratch/d/diego.calanzone/ift6760b-project
cd "$PROJECT"

# One thread per RDKit/NumPy process; parallelism comes from --num-workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} started=$(date -Is)"

srun .venv/bin/python scripts/prepare.py \
  --env-dir  data/envs/enamine_290k \
  --block-smi data/catalogs/bb_enamine_raw_uncapped.smi \
  --config   data/templates/explore/config.yaml \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --min-library-size 10 \
  --max-atoms 30

echo "finished=$(date -Is)"
