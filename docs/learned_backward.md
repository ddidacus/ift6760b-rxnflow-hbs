# Learned backward policy (P_B)

RxnFlow v2.1.1 uses a fixed `P_B`: a recursive retrosynthesis search enumerates complete
routes and a hard-coded softmax (`backward_synthon_penalty ** -(n_synthons - 1)`) weighs
their first edges. The value is a float, so nothing is learned. This package adds a
**learned** `P_B` in the style of SynFlowNet (maximum likelihood and REINFORCE) without the
recursive search. The heuristic stays available and is bit-identical to the baseline
(`mode: heuristic`, the default).

## How it works

For a state `s'` (a molecule) the candidates `C(s')` are its **one-step reverse steps**
`(action, parent)`: a reverse template whose removed synthon is a catalogue row and whose
forward reaction reproduces `s'` (`RetroSynthesisAnalyzer.one_step_many`, cached by SMILES).
Parents that cannot themselves be built from the catalogue ("dead ends") are candidates too.

`P_B(e | s') = softmax over C(s') of  head([h(s'), rxn(e), type(e), synthon(e)])`

- `h` is the forward policy's state encoder, `rxn` and `synthon` its reaction and synthon
  embeddings (so the head shares parameters with the forward policy unless
  `detach_shared: true`).
- The head's last layer starts at zero, so `P_B` starts **uniform over the candidates**.
- TB uses `sum_t log P_B` **detached** (SynFlowNet does the same); `P_B` is trained only by
  its own objective.

Objectives (`backward.objective`):

| objective | loss |
|---|---|
| `none` | none; `P_B` stays uniform over the candidates (control) |
| `mle` | `-mean_traj sum_t log P_B(e_t | s_t)` on the edges the forward policy generated |
| `reinforce` | `J_B = -mean(R_B * log P_B(tau)) - entropy_weight * H(P_B)` over `{tau_B} u {tau_F}` |

The two are **independent and never combined**, and they differ in which trajectories the
loss covers, following SynFlowNet's Algorithms 1 and 2:

- **MLE** covers the forward trajectories `{tau_F}` only.
- **REINFORCE** covers the union of `{tau_B}`, backward trajectories sampled from the current
  `P_B` (called rollouts in the code), and `{tau_F}`. SynFlowNet draws 64 of each
  (`num_from_policy = 64`, `num_from_buffer_for_pb = 64`); here the backward trajectories
  start at molecules drawn from the replay buffer, one per trajectory in the batch, so the
  halves are equal. There is no baseline, as in SynFlowNet.

`{tau_F}` is RxnFlow's whole TB batch, 64 online plus 64 replay trajectories, for both
objectives. SynFlowNet uses only its fresh online batch; keeping RxnFlow's formulation was a
deliberate choice (notes/10 section 4.3 explains the difference and the online-only variant).

`R_B` is +1 when a walk reaches the empty state (SynFlowNet's `ends_in_s_0`; the route
length is not checked) and `illegal_reward` (-1) otherwise. A walk is cut off after
`max_reactions + 1` steps, the counterpart of SynFlowNet's `max_len`.

## Use

```bash
python scripts/train_qed.py --config configs/experiments/qed_bck_reinforce.yaml \
    --steps 1000 --output-dir ~/runs/qed-reinforce
```

Configs in `configs/experiments/`: `qed.yaml` (heuristic baseline) and
`qed_bck_{uniform,mle,reinforce}.yaml`; for Enamine 290k with the CDK8 similarity reward,
`cdk8_sim_enamine.yaml` (baseline) and `cdk8_sim_enamine_bck_{uniform,mle,reinforce}.yaml`. The
`backward:` section is documented in `training/config.py` (`BackwardConfig`). Extra metrics
in `training.jsonl`: `bck_nll`, `bck_uniform_nll` (what a uniform `P_B` would score),
`bck_p_observed` (mean probability of the generated edge), `bck_acc`, `bck_entropy`, `bck_n_cand`, `bck_observed_missing`, `bck_rollout_success`,
`bck_rollout_reward`, `tb_loss`, `bck_aux_loss`.

## Tools

- `scripts/measure_backward_candidates.py`: how many candidates are dead ends on a run's
  molecules and how often a uniform `P_B` would fail.
- `scripts/probe_backward_policy.py`: train `P_B` alone on fixed molecules and report its
  exact backward success probability on training and held-out molecules.
- `scripts/benchmark_backward.py`: held-out comparison of RxnFlow's native search with
  uniform / MLE / REINFORCE `P_B`. A product is solved when a backward route reaches the empty
  state and replays forward through `env.step()`; solve@k is exact; `--time` adds seconds per
  solved product (run it on an idle machine). Helpers live in `IFT6760/evaluation/backward.py`.
- `scripts/compare_runs.py`: reward, TB residual, `P_B` statistics and scaffold counts across
  training runs.

## What was measured (`example_5k`, molecules from a 200-step QED run)

| | early (steps 1-5) | late (steps 196-200) |
|---|---|---|
| one-step candidates per state (range, mean) | 1-4, 1.8 | 1-4, 2.0 |
| generated edge regenerated | 100% | 100% |
| dead candidates among reverse steps of final molecules | 16.7% | 25.6% |
| uniform-`P_B` success probability | 0.526 | 0.443 |
| molecules with success below 1 | 286/300 | 288/300 |
| first edges equal to the recursive search's | 300/300 | 300/300 |
| cost of one `one_step` call | 6.4 ms | 6.0 ms |

Dead ends are therefore common, a uniform `P_B` fails about half of its backward walks, and
the validity reward carries real signal on this catalogue.

## Cost and caveats

- Evaluating `P_B` re-encodes each candidate's state: on this 12-core CPU a step took
  roughly 9-16 s against 5 s for the heuristic, measured when the head also encoded each
  candidate's parent. Scoring without the parent (now the only behaviour) is about 2x
  cheaper. Real runs belong on a GPU.
- With a leaky `P_B` the sampler targets `R(x) * p_succ(x)` rather than `R(x)`, where
  `p_succ(x)` is the success probability of a backward walk from `x` (the quantity above).
- A rollout's success does not check `min_*`/`max_*`, as in SynFlowNet (RxnFlow's recursive
  search ignores the minimums too, see the TODO in `RxnFlowPolicy.calc_bck_logprob`). The
  measurement scripts still count success only within the forward limits, so their `p_succ`
  is not the quantity logged as `bck_rollout_success`.
- Checkpoints from before this change are rejected (`__version__` 2.2.0).

## Tests

`pytest -m "not slow"` (about 4 minutes) and `pytest -m slow` (end-to-end trainer runs, about
5 minutes). The slow suite includes the regression against the RxnFlow v2.1.1 reference run.
