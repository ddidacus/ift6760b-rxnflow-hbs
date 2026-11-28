#!/usr/bin/env bash
# Copy the parts of RxnFlow we build on into this repo, rewriting import paths only.
# Re-runnable (overwrites the vendored files). Run it on a clean tree or a vendor/* branch
# so that `git diff` shows exactly what changed.
set -euo pipefail

# The vendored files have since been modified (backward policy, config, trainer, ...), so a
# re-run would silently overwrite that work. Use it only on a clean vendor/* branch to
# re-sync a new upstream release, then merge/rebase the changes deliberately.
if [ "${1:-}" != "--force" ]; then
  echo "refusing to run: this overwrites modified files. Re-run with --force on a clean vendor/* branch." >&2
  exit 1
fi

: "${REFS_DIR:?export REFS_DIR first (the folder that contains RxnFlow/)}"
UP="$REFS_DIR/RxnFlow"
TAG="v2.1.1"
REV="8f7c9975a4f8f12e79c9753428a4c0d71de3563a"
ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"

# "path inside RxnFlow/ : path inside this repo"
FILES=(
  "src/rxnflow/__version__.py:src/IFT6760/__version__.py"
  "src/rxnflow/core/errors.py:src/IFT6760/core/errors.py"
  "src/rxnflow/core/molecule.py:src/IFT6760/core/molecule.py"
  "src/rxnflow/core/reaction.py:src/IFT6760/core/reaction.py"
  "src/rxnflow/core/synthon.py:src/IFT6760/core/synthon.py"
  "src/rxnflow/core/types.py:src/IFT6760/core/types.py"
  "src/rxnflow/envs/env.py:src/IFT6760/envs/env.py"
  "src/rxnflow/envs/features.py:src/IFT6760/envs/features.py"
  "src/rxnflow/envs/graph.py:src/IFT6760/envs/graph.py"
  "src/rxnflow/envs/library.py:src/IFT6760/envs/library.py"
  "src/rxnflow/envs/retrosynthesis.py:src/IFT6760/envs/retrosynthesis.py"
  "src/rxnflow/envs/prepare/build_action_space.py:src/IFT6760/envs/prepare/build_action_space.py"
  "src/rxnflow/envs/prepare/convert_synthons.py:src/IFT6760/envs/prepare/convert_synthons.py"
  "src/rxnflow/envs/prepare/main.py:src/IFT6760/envs/prepare/main.py"
  "src/rxnflow/envs/prepare/prepare_metadata.py:src/IFT6760/envs/prepare/prepare_metadata.py"
  "src/rxnflow/envs/prepare/preprocess_features.py:src/IFT6760/envs/prepare/preprocess_features.py"
  "src/rxnflow/envs/prepare/read_blocks.py:src/IFT6760/envs/prepare/read_blocks.py"
  "src/rxnflow/models/__init__.py:src/IFT6760/models/__init__.py"
  "src/rxnflow/models/encoders.py:src/IFT6760/models/encoders.py"
  "src/rxnflow/models/mpnn.py:src/IFT6760/models/mpnn.py"
  "src/rxnflow/models/nn.py:src/IFT6760/models/nn.py"
  "src/rxnflow/models/rxnflow.py:src/IFT6760/models/rxnflow.py"
  "src/rxnflow/gflownet/policy.py:src/IFT6760/policies/policy.py"
  "src/rxnflow/gflownet/trajectory_balance.py:src/IFT6760/objectives/trajectory_balance.py"
  "src/rxnflow/gflownet/replay.py:src/IFT6760/sampling/replay.py"
  "src/rxnflow/gflownet/conditioning.py:src/IFT6760/sampling/condition_sampler.py"
  "src/rxnflow/conditioning.py:src/IFT6760/sampling/conditioning.py"
  "src/rxnflow/sampler.py:src/IFT6760/sampling/sampler.py"
  "src/rxnflow/gflownet/property_penalty.py:src/IFT6760/rewards/property_penalty.py"
  "src/rxnflow/reward.py:src/IFT6760/rewards/reward.py"
  "src/rxnflow/config.py:src/IFT6760/training/config.py"
  "src/rxnflow/trainer.py:src/IFT6760/training/trainer.py"
  "examples/qed.py:scripts/train_qed.py"
  "scripts/prepare.py:scripts/prepare.py"
  "scripts/sample.py:scripts/sample.py"
  "configs/qed.yaml:configs/experiments/qed.yaml"
  "configs/template.yaml:configs/template.yaml"
)

# Only dotted module paths are rewritten. Do NOT blanket-replace "rxnflow": the prepared
# environment stores the strings "rxnflow_version" and "rxnflow-synthon-feature".
REWRITE=(
  -e 's/\brxnflow\.__version__\b/IFT6760.__version__/'
  -e 's/\brxnflow\.conditioning\b/IFT6760.sampling.conditioning/'
  -e 's/\brxnflow\.config\b/IFT6760.training.config/'
  -e 's/\brxnflow\.reward\b/IFT6760.rewards.reward/'
  -e 's/\brxnflow\.sampler\b/IFT6760.sampling.sampler/'
  -e 's/\brxnflow\.trainer\b/IFT6760.training.trainer/'
  -e 's/\brxnflow\.gflownet\.policy\b/IFT6760.policies.policy/'
  -e 's/\brxnflow\.gflownet\.trajectory_balance\b/IFT6760.objectives.trajectory_balance/'
  -e 's/\brxnflow\.gflownet\.replay\b/IFT6760.sampling.replay/'
  -e 's/\brxnflow\.gflownet\.conditioning\b/IFT6760.sampling.condition_sampler/'
  -e 's/\brxnflow\.gflownet\.property_penalty\b/IFT6760.rewards.property_penalty/'
  -e 's/\brxnflow\.(core|envs|models)\b/IFT6760.\1/'
  -e 's/^from rxnflow import/from IFT6760 import/'
)

for pair in "${FILES[@]}"; do
  src_rel="${pair%%:*}"
  dst_rel="${pair##*:}"
  dst="$ROOT/$dst_rel"
  mkdir -p "$(dirname "$dst")"
  {
    # Attribution lives in NOTICE.md and THIRD_PARTY_LICENSES/, not per file.
    printf '# Source: %s\n' "$src_rel"
    # tr -d '\r': a Windows checkout of upstream has CRLF line endings; keep this repo LF.
    sed -E "${REWRITE[@]}" "$UP/$src_rel" | tr -d '\r'
  } > "$dst"
done

# Data for the smoke test (git-ignored, see data/README.md) and the upstream licence text.
mkdir -p "$ROOT/data/envs" "$ROOT/data/templates" "$ROOT/data/building_blocks" "$ROOT/THIRD_PARTY_LICENSES"
cp -r "$UP/data/envs/example_5k" "$ROOT/data/envs/"
cp -r "$UP/data/templates/." "$ROOT/data/templates/"
cp -r "$UP/data/building_blocks/." "$ROOT/data/building_blocks/"
cp "$UP/LICENSE" "$ROOT/THIRD_PARTY_LICENSES/RxnFlow-LICENSE"

echo "vendored ${#FILES[@]} files from RxnFlow $TAG ($REV)"
