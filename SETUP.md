# RxnFlow — vendored copy, environment, and the CDK8 roshambo2 runs

Upstream: <https://github.com/SeonghwanSeo/RxnFlow>

| | |
|---|---|
| Commit | `0693e9825d89552ef5dc5ea8f265a2581dfa62a6` (`main`, "add replay sampling option") |
| Authored | 2026-11-13 |
| Cloned | 2026-11-17, `--depth 1` |
| Version | `rxnflow 2.1.1` |

The `.git` directory was removed so this does not become a repository nested inside the
parent. **Nothing in the upstream tree was modified**: `README.md`, `LICENSE`, `pyproject.toml`
and every file under `src/`, `configs/`, `examples/`, `scripts/` and `data/` are as cloned.
Everything added for this package lives in `hbs/`, and reaches RxnFlow by subclassing it:

```
hbs/reservoir.py            hierarchical binary search over fixed features, hashed once
hbs/policy.py               RxnFlowPolicy with no subsampling: candidates are a bucket
hbs/metrics.py              per-trajectory diversity and latency, identical in both arms
hbs/trainer.py              the two trainers: metrics-only, and metrics + HBS
hbs/reward.py               roshambo2 shape/colour reward, CDK8 by default
hbs/train.py                entry point for both arms
hbs/analyze.py              quality and diversity of a finished run
hbs/monitor.py              side-by-side status of running runs
hbs/configs/cdk8.yaml       the run configuration, shared by both arms
hbs/configs/cdk8_enamine.yaml  the same, over the prepared 290k Enamine catalog
hbs/sbatch/train_cdk8.sh    one run on one GPU
hbs/requirements-frozen.txt the installed environment
```

The baseline arm therefore runs upstream code exactly as cloned, which is the point: the two
runs differ in one mechanism and in nothing else.

## Environment

```bash
module load cuda/12.4.1        # roshambo2 builds a CUDA extension; nvcc must be on PATH
uv venv --python 3.12          # roshambo2 requires >=3.11, rxnflow >=3.10
uv pip install -e .
uv pip install roshambo2==1.0.0
uv pip install pandas          # roshambo2 imports it without declaring it
```

`roshambo2` is not an extra of upstream's `pyproject.toml` and was not added to it, so the
core install stays exactly as cloned; the frozen set is in `hbs/requirements-frozen.txt`
(`torch 2.9.0+cu128`, `numpy 2.4.4`, `rdkit 2025.9.1`, `roshambo2 1.0.0`). There is no
`uv.lock` here for the same reason: the environment is `pyproject.toml` plus those three
commands.

## Prepared synthesis environment

The default building blocks and the default (`explore`) templates, as shipped:

```bash
.venv/bin/python scripts/prepare.py \
  --block-smi ./data/building_blocks/example_5k.smi \
  --env-dir ./data/envs/example_5k/ \
  --config ./data/templates/explore/config.yaml \
  --num-workers 2
```

27s, and it is not tracked — regenerate it rather than copy it. It yields **45 synthon
libraries holding 5,864 synthons**, sizes from 10 to 1,672 (top five: 1672, 580, 567, 356,
336).

## Reward

`hbs/reward.py` scores a molecule by the best 3D overlay it achieves against a reference
ligand, with the settings the published REINVENT, Augmented Hill Climbing and NuevaMol runs
were trained on (`num_isomers=2`, `num_confs=25`, colour on, combination score, CUDA
backend), so its numbers sit on the same scale as those and as
`baselines/gflownet/gflownet/proxy/roshambo2.py`. The default target is **CDK8**
(`support/queries/CDK8_insilico.sdf`, 27 heavy atoms); `TARGET_FILES` covers the other eight
query molecules in `support/queries`, and `sdf_path` takes any other.

Anything RDKit cannot parse, and any molecule whose conformer generation exceeds
`prep_timeout` (300s), scores 0 — penalised rather than dropped. The conformer pool is built
once for the run and sized by `ROSHAMBO_NUM_CPUS`: preparation is the dominant cost and is CPU
bound, so that variable sets the wall time.

RxnFlow scores only unique, uncached SMILES, so `run_summary.json`'s `oracle_calls` counts
distinct molecules, which is what the other baselines in this package budget against.

## The two runs

```bash
sbatch hbs/sbatch/train_cdk8.sh uniform     # upstream action subsampling
sbatch hbs/sbatch/train_cdk8.sh hbs         # hierarchical binary search
```

2,500 steps of 64 online samples (`STEPS=` and `SEED=` override), so at most 160k scored
molecules — the 150k budget the published roshambo2 baselines here used.

**Where a step's time goes**, measured here on one L40S with a 4-process conformer pool:
generation 0.4s, everything else ~110s, essentially all of it the reward. RxnFlow's
three-block molecules are far more expensive to embed than the small molecules the parent
package profiled at 3.4s per 64-molecule batch (2 stereoisomers x 25 ETKDGv3 conformers +
MMFF for each, on larger and more flexible molecules with more unassigned stereocentres).
`num_confs` and `num_isomers` are fixed at the published values for comparability, so the
pool width is the only knob: the runner asks for 16 CPUs and gives 12 to the pool, which is
~24-35h for 2,500 steps, inside its 72h limit. Two consequences worth stating: the arms
cannot be compared on wall time, since the reward swamps the difference between retrieving
and sampling candidates; and a run that needs to be cheaper should cut `STEPS`, never the
conformer settings. Both arms share
`hbs/configs/cdk8.yaml`: `beta=32`, `mw<=500`, generation and model defaults, and
`sampling_ratio=0.1, min_sampling=50`. On this catalog those subsample **20 of the 45
libraries, holding 5,343 of the 5,864 synthons**, so 1,652 of 5,864 candidate actions are
scored per state and the mechanism being replaced is actually doing something. Each run
writes `training.log`, `training.jsonl` (per-step metrics), `sample.jsonl` (every molecule,
its reward and its synthesis route), `checkpoints/` and a final `run_summary.json`.

## What the HBS arm changes

[`METHODS_COMPARISON.md`](METHODS_COMPARISON.md) is the condensed side-by-side; this section
is the detail.


RxnFlow never scores a whole library. It draws K of the N rows uniformly, scores those, and
corrects the normalisation estimate by log(N/K) (`SubsamplingPolicy`). That correction is the
price of not having looked at the other N-K rows.

**The HBS arm removes the subsample.** Every row of every library is hashed once, before the
first step, in *fixed* feature space, and an action's candidates are the rows of the bucket
its query falls into. There is no K, the config's `sampling_ratio` is ignored, and
`log_importance` is 0 for every library, so the normalisation over the bucket is exact rather
than an importance-corrected estimate of the whole library.

Three decisions make that work:

1. **The reservoir is fixed, so the index never goes stale.** Upstream scores an action as
   `query . F.normalize(encode_synthon(fingerprint, properties, types))`, and that encoder is
   trained like the rest of the policy — measured between two checkpoints of the finished
   uniform run, its weights move by 0.002-0.023 per tensor, so any index over those vectors
   is wrong as soon as it is built. The reservoir here is each synthon's **precomputed
   fingerprint** instead (`library.fingerprints`, 678 bits of Morgan + MACCS written when the
   environment was prepared), L2-normalised. Those vectors never change, so the table is
   built once and is correct for the whole run.
2. **Learning stays on the query side.** Upstream's action heads are untouched; one
   `Linear(synthon_dim -> 678)` registered on the model projects their output into the
   reservoir's space, so the inner product being ranked and the angle the hash estimates live
   in the same space. Because only the query moves, nothing has to be re-encoded: the
   synthon encoder leaves the action path entirely.
3. **Hashing is about the origin, not the centroid.** `scripts/hbs.py` mean-centres both the
   reservoir and the query, which is right when queries come from the same distribution as
   the reservoir (as in its own demo). Here the query is a learned vector, and the logit
   ranks cosine *about the origin*, so that is what the hash must estimate. Measured on
   `example_5k`'s largest library, for queries that point at specific blocks, origin hashing
   finds the best row 4.9x more often than chance against the centroid's 4.3x.
   `--hbs-center` restores the reference behaviour.

`--hbs-depth` replaces the sampling ratio: it is the number of hyperplanes, so a library is
cut into 2^depth cells and a step scans roughly N / 2^depth rows. `0` (the default) picks it
per library from its size, aiming at `--hbs-target-bucket` rows, because libraries here span
128 to 92,813 rows and one depth cannot suit both ends. A bucket is
widened to its Hamming neighbours only when it holds fewer than `--hbs-min-rows` rows (default
1), which is the `hamming` import `scripts/hbs.py` never used.

### What it costs

* **A logit is now linear in the fingerprint bits**, with state-dependent coefficients, where
  upstream's was `query . MLP(bits)`. That is strictly less expressive, and it is the
  deliberate trade for an index that never needs rebuilding. The paper's own argument for a
  learned block embedding (ICLR 2025, App. B.5) is parameter economy and generalisation to
  unseen blocks; both survive here, since fingerprints are features too, but the reward-
  adapted reweighting the encoder provides does not.
* **Candidate sets are uneven.** Bucket occupancy in fingerprint space is clustered: on
  `example_5k` the measured spread is 2 to 390 rows. One state may score a few dozen
  candidates and another a few thousand.
* **Reachability is no longer guaranteed.** A row outside the query's bucket cannot be chosen
  at that step, where uniform subsampling gave every row probability K/N.
  `get_action_logits` still scores an observed action directly, so replayed and
  retrosynthesis-derived trajectories keep a finite forward logit.

### Measured retrieval quality

`--hbs-probe-every` scores one library exactly and reports how often the opened bucket held
the best row, against the chance level (which is exactly the share of the library scanned).
Offline, on `example_5k`'s largest library (N=1,672, depth 3, ~13% scanned):

| query | best row in bucket | chance | lift |
|---|---|---|---|
| projected query at initialisation | 18.0% | 12.8% | 1.40x |
| random Gaussian | 18.3% | 12.4% | 1.48x |
| pointing at a specific block | 62.0% | 12.7% | **4.87x** |

So the mechanism works when the query points at blocks, and barely better than chance when it
points anywhere. Whether training drives the projection into the first regime is the open
question these runs answer — watch the probe lines, and note that a probe fires on whatever
states are active, which can be as few as one query, so single lines are noisy and only their
average over many probes means anything (`hbs/monitor.py` aggregates them).

Replacing a *learned-embedding* index was the previous design of this arm, and its finished
comparison is in `RESULTS_26_11_18.md`; it kept K and log(N/K) and so tested which rows get
sampled, not whether subsampling is needed.

## Per-trajectory metrics

Both arms record the same three families every run, written to
`trajectory_metrics.jsonl` in the output directory, one line per step. They exist to
separate "did the mechanism change which candidates are offered" from "did that change
what got built":

| Field | Per trajectory, over | Meaning |
|---|---|---|
| `block_div_library` | the synthon rows scored at each of its steps | mean over subsampled (reaction, library) subspaces of 1 - mean pairwise Tanimoto. Uniform subsampling ignores the query, so its candidate sets should be as diverse as the library; retrieval that works should be visibly tighter |
| `block_div_union` | the same rows pooled per step | one diversity over the whole candidate set of that step |
| `route_scaffolds` | its selected synthons + obtained products | distinct Bemis-Murcko scaffolds |
| `route_div` | the same molecules | 1 - mean pairwise Tanimoto |
| `latency_ms` | - | wall clock from the start of the batch's sampling to this trajectory being complete |
| `policy_ms` | - | the share of that spent in its own forward passes |

Fingerprints are Morgan radius 2, 2,048 bits, and diversity is the definition in
`molgen/utils/diversity.py`, so these numbers are comparable with the rest of the package.
Sets larger than `--metrics-pairs` pairs are estimated from that many random pairs.

Latency is recorded every step; the chemistry is recorded every `--metrics-every` steps on
`--metrics-states` trajectories (defaults 25 and 8). RxnFlow advances a batch in lockstep,
one batched forward call per step, so `latency_ms` is the wall clock at which a trajectory
became complete *within its batch* — it includes that step's chemistry and backward
analysis, but it is not what the trajectory would cost sampled alone.

The collector only reads the action categorical the policy already built and draws from
its own generator, so turning metrics on does not change what either arm samples. With
`--metrics-every 0` and no `--hbs`, upstream's trainer runs with nothing of ours in the path.

Inspect runs while they train, which touches only the small append-only logs:

```bash
.venv/bin/python -m hbs.monitor runs/cdk8_enamine_uniform_s1 runs/cdk8_enamine_hbs_s1
.venv/bin/python -m hbs.monitor runs/*_s1 --window 200 --top-k 10   # --top-k rescans samples
```

and score finished ones (top-k reward, `div@k`, scaffolds, `#circles`, `auc_top_10`):

```bash
.venv/bin/python -m hbs.analyze runs/cdk8_uniform_s1 runs/cdk8_hbs_s1
```

## Resuming

`RxnFlowTrainer.__init__` refuses an existing output directory and `load_checkpoint` refuses
a changed config, reward class or catalog. A restart therefore needs a *new* `OUTPUT_DIR=`
plus `--resume-from-checkpoint <old>/checkpoints/latest.ckpt`. `long` is not preempted, so
this only comes up after a crash or a node failure.

## Scope

A reference implementation of one substitution, kept here so RxnFlow can be run alongside
REINVENT, Augmented Hill Climbing and the GFlowNet baseline on the same targets. Not
budget-matched to them automatically: `oracle_calls` in `run_summary.json` is the number to
compare on, since duplicate SMILES are served from the reward cache and do not spend budget.
