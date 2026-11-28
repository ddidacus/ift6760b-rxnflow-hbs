# Learned backward policy (`P_B`)

Notes on the `learned backward policy` branch of this repository, written so the feature
can be integrated here rather than read in place. Source: commit
[`f985461`](https://github.com/ddidacus/ift6760b-rxnflow-hbs/commit/f985461ff2b7218e23ce34a07e2654f2ad8e09c2)
by Rubing Zhang, 27 Nov 2026 -- an **initial commit with no parents**, 80 files and 11,765
insertions. It is a parallel re-layout of RxnFlow v2.1.1 into `src/IFT6760/`, with its own
`pyproject.toml`, `uv.lock` and config system, so **nothing in it merges into this tree**.
Integration means porting.

Its own design document is `docs/learned_backward.md` on that branch; this file covers what
that one does not: the gap against *our* layout, and what has to be built to close it.

## What the feature is

RxnFlow's `P_B` is fixed. `RxnFlowPolicy.calc_bck_logprob`
(`src/rxnflow/gflownet/policy.py:630`) runs a recursive retrosynthesis search, enumerates
*complete* routes, and weighs their first edges by a hard-coded
`backward_synthon_penalty ** -(n_synthons - 1)` -- a float from the config, so nothing is
learned.

The branch replaces it with a learned `P_B` in SynFlowNet's style, and drops the recursive
search from the training path entirely:

- **Candidates** `C(s')` are the molecule's **one-step** reverse steps `(action, parent)`,
  from a new `one_step_many`, cached by SMILES. Parents that cannot themselves be built
  from the catalog ("dead ends") are candidates too -- learning to avoid them is the point.
- `P_B(e | s') = softmax over C(s') of head([h(s'), rxn(e), type(e), synthon(e)])`, where
  `h`, `rxn` and the synthon encoder are the **forward policy's own**. `detach_shared`
  gates whether `P_B`'s gradient reaches them.
- The head's last layer is zero-initialised, so `P_B` starts exactly uniform over the
  candidates. That makes `objective: none` a clean frozen control rather than an
  approximation of one.

### The contract, in one line

`src/IFT6760/objectives/trajectory_balance.py:106-108` and `:151`:

```python
traj_log_p_B = bck_output.traj_log_p_B.detach()   # TB sees a constant
...
loss = loss + bck_output.aux_loss                 # P_B is trained only by its own loss
```

TB never trains `P_B`; an auxiliary objective does, and its loss is simply added. Everything
else is plumbing around that.

### The two objectives

Independent, never combined, differing in which trajectories they cover (SynFlowNet
Algorithms 1 and 2):

| `backward.objective` | loss | covers |
|---|---|---|
| `none` | none; `log P_B = -log\|C\|` by construction | -- |
| `mle` | `-mean_tau sum_t log P_B(e_t \| s_t)` on the edges `P_F` generated | `{tau_F}` |
| `reinforce` | `-mean(R_B * log P_B(tau)) - entropy_weight * H`, no baseline | `{tau_B} u {tau_F}` |

`R_B` is `+1` when a backward walk reaches the empty state and `illegal_reward` (`-1`)
otherwise. `{tau_B}` are walks sampled from the current `P_B` starting from replay-buffer
terminal states; `{tau_F}` is RxnFlow's whole TB batch (64 online + 64 replay), where
SynFlowNet uses only its fresh online batch.

### The counter trick

Worth understanding before porting, because it is what makes the whole thing well-defined
without a legality test. A walk carries the `(reactions, synthons)` counters of the terminal
state it started from and decrements them through `parent_counts`. Arriving at the empty
state with both at zero therefore *means* the route it found spends exactly what the
recorded one did. The same rule is applied to demonstrations in `observed()`: candidates are
filtered to the genuine parents of `(molecule, reactions, synthons)`, not of the molecule
alone, so `P_B` is normalised over the support a walk could actually sample. If the generated
edge is not rediscovered the code raises `MissingReverseEdge` instead of patching over it --
"scoring an edge the walk could never sample trains on a distribution that does not exist."

## Why it might be worth having

Measured on that branch (`example_5k`, molecules from a 200-step QED run):

| | early (steps 1-5) | late (steps 196-200) |
|---|---|---|
| one-step candidates per state (range, mean) | 1-4, 1.8 | 1-4, 2.0 |
| dead candidates among reverse steps | 16.7% | 25.6% |
| uniform-`P_B` walk success probability | 0.526 | 0.443 |
| molecules with success below 1 | 286/300 | 288/300 |
| first edges equal to the recursive search's | 300/300 | 300/300 |
| cost of one `one_step` call | 6.4 ms | 6.0 ms |

Read these two rows together before committing effort. A uniform `P_B` fails roughly half
its backward walks, which is the case *for* learning -- but a uniform `P_B` already matches
the recursive search on the **first edge for 300/300 molecules**, so what is on offer is
fixing walk survival deeper in the tree, not a better first-edge distribution. The branch
states the consequence of leakage plainly: with a leaky `P_B` the sampler targets
`R(x) * p_succ(x)` rather than `R(x)`.

## What we already have, and what is missing

Checked against this tree at the time of writing.

**Present, no work needed** -- `model.rxn_embedding`, `model.encode_synthon`,
`env.library_site_indices`, `molecule_to_graph_data` / `GraphBatch` / `GraphData`,
`State.num_reactions` / `State.num_synthons`, `step.product_smiles`,
`ReplayBuffer.sample(count, rng)` (identical signature), `BackwardTrajectory`.

**Missing -- these are the prerequisites**, in dependency order:

1. **`ActionType.cost`** (`src/rxnflow/core/types.py`). A 2-line property:
   `(0 if is_first else 1, 0 if is_unirxn else 1)`. The MDP's counting rule, stated once.
   Everything below depends on it.
2. **`SynthesisEnv.parent_counts(action, reactions, synthons)`** (`src/rxnflow/envs/env.py`).
   A static method, ~6 lines: subtract the cost, then return `(0, 0)` only if FirstSynthon
   lands exactly on the empty state, else `(reactions, synthons)` if `reactions >= 0 and
   synthons >= 1`, else `None`. This is the function that makes `P_B` a distribution over
   parents of a *state* rather than of a molecule.
3. **`one_step` / `one_step_many`** (`src/rxnflow/envs/retrosynthesis.py`). The real work.
   Our analyzer exposes only `run` / `submit` / `result` / `close` -- the recursive search.
   The branch adds a single-reverse-step enumerator (`_RetroDFS.one_step`) plus an
   LRU-cached, pool-parallel `one_step_many` that returns candidates in canonical order.
   This is the largest piece to port and the one to review most carefully, since the
   guarantee that the generating edge is always among the candidates lives here.
4. **`BackwardEdge = tuple["Action", str]`** -- a type alias; ours only has
   `BackwardTrajectory = list[tuple["Action", str]]`.
5. **`Segments`** (`src/IFT6760/utils/segments.py`, 65 lines). The ragged-softmax helper:
   `log_softmax`, `pick`, `entropy`, `sample` over variable-width candidate lists. Note it
   contains the word "backward" nowhere, so it will not turn up if you grep for the
   feature -- and nothing works without it.
6. **`MissingReverseEdge`** -- one exception class.

## Port map

Their layout is flat (`src/IFT6760/`); ours splits the harness (`hbs/`) from the package
(`src/rxnflow/`), and the forward policies now live under `src/rxnflow/gflownet/fwd/`
(`fwd/hbs/`, `fwd/cas/`). A backward policy is not a forward policy, so it belongs beside
`gflownet/`, not inside `fwd/`.

| theirs | lines | suggested destination here |
|---|---|---|
| `policies/backward.py` | 332 | `src/rxnflow/gflownet/bck/policy.py` |
| `objectives/backward.py` | 181 | `src/rxnflow/gflownet/bck/objective.py` |
| `models/backward.py` | 52 | `src/rxnflow/models/backward.py` (head only) |
| `utils/segments.py` | 65 | `src/rxnflow/utils/segments.py` |
| `evaluation/backward.py` | 372 | `hbs/eval_backward.py` (measurement, not runtime) |
| `scripts/compare_backward_policies.py` | 209 | `hbs/compare_backward.py` |
| `scripts/{benchmark,probe,measure}_backward*.py` | 850 | `hbs/` as needed |

Integration points in existing files:

- `src/rxnflow/gflownet/trajectory_balance.py` -- accept an optional `backward` objective,
  detach its `traj_log_p_B`, add its `aux_loss`. Our `TrajectoryBalance.__init__` currently
  takes `(policy, moo_scalarization, reward_floor, loss_fn)`; the branch appends
  `backward: BackwardObjective | None = None`. Keeping the default `None` is what makes the
  heuristic path bit-identical.
- `src/rxnflow/models/rxnflow.py` -- own an optional `backward_head`, zero its output.
- `src/rxnflow/gflownet/fwd/trainer.py` (or `fwd/base.py`) -- build the policy and
  objective, and give the head its own parameter group if `backward.learning_rate` is set.
- `src/rxnflow/config.py` -- a `BackwardConfig` dataclass: `mode`, `objective`, three
  weights, `illegal_reward`, `detach_shared`, `learning_rate`, `cache_size`, with
  `validate()`.
- `src/rxnflow/sampler.py` -- pass `analyze_backward=False` so sampling skips the recursive
  search when `P_B` is learned.

## Cost

On that branch's 12-core CPU a step took **9-16 s against 5 s for the heuristic**, roughly
halved once the head stopped encoding each candidate's parent (the only behaviour now). Our
runs are already step-bound -- jnk3 uniform is 843 steps for 50k oracle calls, CAS 1,530 --
so a 2-3x step cost is the dominant integration risk, not the modelling. Budget a GPU run
before trusting any wall-clock comparison against our existing arms.

## Validating a port

The branch ships the right tool for this and it is worth adopting rather than rebuilding:
`compare_backward_policies.py` scores every arm's `P_B` on **one shared held-out panel**,
and its probabilities are **exact rather than sampled** -- the one-step candidate graph is
small enough to enumerate every complete backward route, so `p_replay` is closed-form and
`solve@k = 1 - (1 - p)^k` follows. Its rows are `native` (the recursive search), `uniform`
(the floor, no network) and each trained arm. A route counts only if replaying its edges
forward through `env.step()` from the empty state is accepted and lands on the same
molecule, which is stricter than checking that counters balance.

Minimum bar for a port:

1. `mode: heuristic` reproduces a current run bit-identically (they claim this; verify it).
2. `objective: none` scores `nll == log|C|` exactly, confirming the uniform start.
3. `native` and `uniform` rows reproduce the numbers above on `example_5k`.

## Caveats to carry across

- **`bck_rollout_success` is not the `p_succ` the measurement scripts report.** Rollouts do
  not check `min_*` / `max_*` limits (as in SynFlowNet, and as in our own
  `calc_bck_logprob`, see its TODO); the scripts count success only within the forward
  limits. Easy to conflate across tables.
- In `reinforce`, `ForwardTrajectories(traj_log_p_B[valid], entropy)` pairs valid-only
  log-probabilities with the entropy over *all* observed decisions. Not a shape bug --
  entropy is only ever meaned -- but the bonus mixes in decisions from invalid
  trajectories.
- Checkpoints from before the change are rejected on that branch (`__version__` 2.2.0). A
  port here needs the same guard or it will silently load a model with no backward head.

## Relation to the HBS probe problem

Worth noting because it is the same shape of problem, already solved once. Our HBS query
projection is trained **only** by trajectory balance, and only through row logits: the
bucket choice is `.detach()`ed (`src/rxnflow/gflownet/fwd/hbs/policy.py:427`) and the
lookup is a sign hash with no gradient anywhere, so nothing in the loss ever says "you
should have opened a different bucket". `P_B` had the identical structure -- a discrete
choice TB cannot reach -- and this branch's answer is exactly the pattern to copy: detach it
inside TB, then train it with a dedicated objective whose reward is whether the discrete
choice worked out. Their `R_B` ("did the walk reach `s_0`") maps onto "did the opened bucket
contain a legal, high-value action". If we go that way, `objectives/backward.py` and
`utils/segments.py` are the parts to read rather than reinvent.
