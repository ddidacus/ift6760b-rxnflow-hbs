#!/bin/bash
#SBATCH --job-name=prep-enamine-2m
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prep-enamine-2m-%j.out

# Prepare a synthon environment from the full Enamine building-block catalog
# (2,945,336 compounds, converted from the 6.1 GB SDF to SMILES+ID by streaming
# its <smiles> and <Catalog_ID> fields -- no RDKit parse needed, 1m42s).
#
# Sized against the 290k build, which took 8m21s on 4 CPUs and 16G: this is ~10x
# the blocks, so ~85 min is expected; 4 CPUs and 48G stay inside the per-user QOS CPU cap
# The six preparation stages each drop a *.done marker in the env dir, so a
# timed-out or failed run resumes from the last completed stage on resubmission.

set -euo pipefail

PROJECT=/network/scratch/d/diego.calanzone/ift6760b-project
cd "$PROJECT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} started=$(date -Is)"

srun .venv/bin/python scripts/prepare.py \
  --env-dir   data/envs/enamine_2m \
  --block-smi data/catalogs/bb_enamine_2m.smi \
  --config    data/templates/explore/config.yaml \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --min-library-size 10 \
  --max-atoms 30

echo "finished=$(date -Is)"
