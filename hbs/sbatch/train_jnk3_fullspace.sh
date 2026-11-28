#!/bin/bash -l
#SBATCH --job-name=jnk3-cas-fullspace
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

# CAS-FullSpace on jnk3: one k-means over all 482,070 rows, no library division.
#
# max_cells is 64, not the 16 default. Measured in a 12-step smoke at 16
# (job 11894102): legal_fraction 1.9%, cells_walked 10.5, fallback_rate 16-33%.
# The filter applies per *reaction query*, and the median query's libraries hold
# only 2.38% of the catalog -- 5.6 legal rows in a 235-row cell -- so the walk
# needs room. The residual fallback at 64 is structural: a library holding 0.01%
# of the catalog has its rows spread over 2,048 cells, so no single cell contains
# one, and that is the real cost of dropping the division.
set -euo pipefail
cd /network/scratch/d/diego.calanzone/ift6760b-project
export PYTHONUNBUFFERED=1
module load cuda/12.4.1
exec .venv-hbs/bin/python -m hbs.train \
  --config hbs/configs/pmo_jnk3.yaml \
  --output-dir runs/pmo/pmo_jnk3_cas_fullspace_s1 \
  --steps 4000 --seed 1 --num-workers 4 \
  --metrics-every 10 \
  --cas --cas-fullspace --cas-max-cells 64 --cas-min-legal 32
