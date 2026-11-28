# PMO benchmark runs

The Practical Molecular Optimization benchmark (Gao et al., NeurIPS 2022,
[*Sample Efficiency Matters*](https://arxiv.org/abs/2206.12411)) is 23
single-objective molecular design tasks, each scoring in `[0, 1]`, each evaluated
under a fixed oracle-call budget. It is the second objective family in this
package, beside the roshambo2 CDK8 target, and it is here for one reason: the
CDK8 runs compare the two action-selection arms on a single expensive 3D
objective, where the reward swamps everything else and only one target is
measured. PMO is cheap, has published baselines, and comes in 23 flavours, so the
same comparison can be made across reward shapes and against other methods.

## What is set up

Four tasks have configs, chosen to span the two reward shapes that matter here:

| Config | Task | Oracle | Shape |
| --- | --- | --- | --- |
| [`pmo_gsk3b.yaml`](pmo_gsk3b.yaml) | `gsk3b` | random forest on Morgan bits | sparse — near-0 almost everywhere |
| [`pmo_jnk3.yaml`](pmo_jnk3.yaml) | `jnk3` | random forest on Morgan bits | sparse, and harder than `gsk3b` |
| [`pmo_drd2.yaml`](pmo_drd2.yaml) | `drd2` | SVC on count-based Morgan | sparse |
| [`pmo_amlodipine_mpo.yaml`](pmo_amlodipine_mpo.yaml) | `amlodipine_mpo` | RDKit descriptors | dense — every molecule scores something |

Three sparse activity classifiers and one dense multi-property objective. The
sparse ones are where candidate selection should matter: if HBS retrieval puts
better building blocks in front of the policy, a reward that is zero almost
everywhere is where that shows. The dense one is the control — it carries a
gradient from any starting point, so both arms should make progress on it, and a
difference there is about optimization rather than about finding anything.

The other 19 tasks need no new code: copy a config and change `task` to any name
in `PMO_TASKS` (`hbs/pmo.py`). Only `gsk3b`, `jnk3` and `drd2` need a download
(~74MB, into `oracle/`); the rest are pure RDKit.

## Protocol

```bash
# one run
sbatch hbs/sbatch/train_pmo.sh uniform              # upstream subsampling
TASK=gsk3b SEED=2 sbatch hbs/sbatch/train_pmo.sh hbs

# 4 seeds x 2 arms = 8 runs, 1 GPU each, all at once
TASK=jnk3 HBS_NOISE_SIGMA=0.8 sbatch --array=0-7 hbs/sbatch/train_pmo.sh

# a short protocol instead: a whole 4-GPU node, non-preemptible, two waves
BUDGET=10000 sbatch hbs/sbatch/sweep_pmo.sh uniform
BUDGET=10000 HBS_NOISE_SIGMA=0.8 sbatch hbs/sbatch/sweep_pmo.sh hbs
```

**The budget, not the step count, ends a run.** RxnFlow scores only unique
uncached SMILES, so an oracle call is a distinct molecule, which is the same
accounting PMO uses. The reward raises `OracleBudgetExhausted` when the budget is
spent and `hbs/train.py` records the run as complete, so `--steps` only has to be
large enough not to end the run first. Set `BUDGET=0` to remove the cap and let
`--steps` decide.

The configs set **50,000 calls**, not PMO's published 10,000. Two reasons: a
synthesis-constrained generator starts from a far smaller reachable set than the
unconstrained baselines PMO tabulates, and AUC top-10 is normalised by the budget,
so it stays on the same scale at any budget. Set `max_oracle_calls: 10000` (or
`BUDGET=10000`) for a number directly comparable with the published table.

**Measured cost**, on one L40S over the Enamine catalog with `jnk3`: a step is
**1.6s** in the uniform arm and **4.5s** in the HBS arm, of which sampling is
0.32s and 1.22s. The reward is 0.37s per 64-molecule batch and no longer sets the
wall time, which is the opposite of the CDK8 runs. 50,000 distinct molecules take
800-2,200 steps depending on how much of each batch repeats a molecule already
scored, so a run is roughly 20-60 min uniform and 1-3h HBS.

**The metric is AUC top-10**, the area under the (oracle calls, mean reward of
the 10 best distinct molecules) curve divided by the budget. It rewards finding
good molecules early rather than merely finding them, which is the whole point of
a fixed budget. `hbs/analyze.py` already computes it — it was written for the
CDK8 runs and needs no changes:

```bash
.venv-hbs/bin/python -m hbs.analyze runs/pmo/pmo_jnk3_uniform_s*/ runs/pmo/pmo_jnk3_hbs_sig080_s*/
```

**These configs set no `property_penalty`**, unlike the CDK8 ones. PMO scores a
molecule by its oracle alone, and with no penalty and a single objective the
`reward` recorded per molecule *is* the oracle score, so the AUC is on the
benchmark's scale and comparable with published numbers. Adding a penalty would
quietly put every result on a different scale.

Everything else is deliberately identical to [`cdk8_enamine.yaml`](cdk8_enamine.yaml):
the same prepared Enamine catalog (388 libraries, 482,070 synthons), the same
`beta: 32`, the same `sampling_ratio: 0.1, min_sampling: 50`. So a PMO pair and a
CDK8 pair differ in the objective and in nothing else, and the HBS arm faces the
same subsampling it is replacing.

## Caveat worth stating

RxnFlow generates only what its catalog and reaction templates can synthesise, and
the published PMO baselines are mostly unconstrained SMILES or graph generators.
On the rediscovery and similarity tasks a synthesis-constrained model may be
unable to reach the target molecule at all, so its ceiling is lower for reasons
that have nothing to do with the optimizer. That makes **arm-versus-arm within
this package** the sound comparison; the published leaderboard is context, not a
like-for-like baseline. The four tasks above were picked partly because none of
them is a rediscovery task.
