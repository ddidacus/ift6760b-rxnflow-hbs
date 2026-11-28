# Forward x backward integration — state and next steps

Branch `feature/merge-fwd-bwd-diego`. Working notes, not documentation; fold anything
durable into `docs/backward_policy.md` and delete this file when the port is finished.

Source is commit `f985461` on `origin/yuran/learned-backward`, which is locally fetched,
so files can be lifted with `git show f985461:src/IFT6760/<path>`.

## Done

Ported verbatim with import paths rewritten (`IFT6760.policies.policy` ->
`rxnflow.gflownet.policy`, `IFT6760.sampling.replay` -> `rxnflow.gflownet.replay`,
`IFT6760.training.config` -> `rxnflow.config`, their `# Source:` header stripped):

| destination | lines | from |
|---|---|---|
| `src/rxnflow/core/errors.py` | +18 | `core/errors.py` (`MissingReverseEdge`) |
| `src/rxnflow/core/types.py` | 265 | `core/types.py` (`ActionType.cost`, `BackwardEdge`) |
| `src/rxnflow/envs/retrosynthesis.py` | 596 | `one_step` / `one_step_many` |
| `src/rxnflow/envs/env.py` | 362 | `parent_counts`, `one_step_cache_size` |
| `src/rxnflow/models/rxnflow.py` | 226 | conditional `backward_head` |
| `src/rxnflow/models/backward.py` | 52 | `BackwardHead` |
| `src/rxnflow/utils/segments.py` | 65 | `Segments` |
| `src/rxnflow/gflownet/bck/policy.py` | 332 | `BackwardPolicy` |
| `src/rxnflow/gflownet/bck/objective.py` | 181 | `BackwardObjective` |
| `src/rxnflow/gflownet/trajectory_balance.py` | — | taken wholesale; `backward=` arg |

Taking those files wholesale was safe because our versions were verified **strict
subsets**: every "ours-only" diff line was the pre-image of one of their edits. Checked
per file with `diff | grep -c '^<'` before copying.

`src/rxnflow/config.py` and `src/rxnflow/trainer.py` were **hand-merged**, because ours
diverge from theirs independently — their config lives at `training/config.py` and their
trainer imports `rxnflow.rewards.*` / `rxnflow.sampling.*` packages we do not have, and
**our trainer carries the replay-sampling checkpoint shim from `0693e98` that their
branch lacks**. Do not copy either file wholesale.

- `config.py`: `BackwardConfig` spliced in before `Config`, registered as
  `backward: BackwardConfig` and validated in `Config.validate`.
- `trainer.py`, five edits: backward imports; `one_step_cache_size=cfg.backward.cache_size`
  on the env; `cfg.backward` passed to **both** `RxnFlowModel` constructions; the
  optimizer `groups` block giving the head its own LR; `BackwardPolicy`/`BackwardObjective`
  built in `_setup_training` and handed to TB as `backward=`; and
  `analyze_backward=not cfg.backward.learned` on the training sample call.

### Verified

- All backward modules import; `compileall` clean.
- Default config is `mode: heuristic` -> `backward.learned` False -> `backward=None` in TB.
- All five forward arms still train 2 steps: uniform, `--hbs`, `--hbs --hierarchical`,
  `--cas`, `--cas --cas-fullspace`.
- The pipeline is bit-deterministic run-to-run on CPU (two identical runs -> max|diff| 0),
  so numeric comparisons below are meaningful.

### Minimum bar 1 is NOT met, but the cause is benign

`mode: heuristic` is **numerically equivalent, not bit-identical** to a pre-port run of
the same command and seed. After 2 steps the projection weights differ by 7.4e-6 (`--hbs`)
and 1.3e-5 (uniform). Pinpointed:

- Step 1 is **exact** on every logged scalar, including `traj_log_p_B`, `traj_log_p_F`,
  `logZ`, `loss` and `scaled_log_R`. So candidate generation, the forward pass and the
  heuristic `P_B` are untouched.
- The first difference is `traj_log_p_F` at step 2: -17.290951 vs -17.290953, about 1e-7
  relative — float32 epsilon.
- Identical forward loss with a slightly different gradient points at a reduction-order
  change in the wholesale-ported `trajectory_balance.py`, not a behavioural change.

Decide whether that is acceptable. If bit-identity is wanted, re-apply the TB change as a
minimal hand edit (add the `backward` parameter and the `if backward is not None` branch
to **our** file) instead of taking theirs, then re-run the comparison.

## Next, in order

1. **CLI flags** in `hbs/train.py`: `--backward-mode {heuristic,learned}`,
   `--backward-objective {none,mle,reinforce}`, the three weights, `--illegal-reward`,
   `--detach-shared`, `--backward-lr`, `--backward-cache-size`; map onto `config.backward`
   the way the CAS/HBS flags map onto their settings. Add `backward` to the
   `run_config.json` summary (it uses `asdict`, so a dataclass is fine).
2. **Smoke-test the learned path** on `example_5k`:
   `--backward-mode learned --backward-objective none` first, which must score
   `nll == log|C|` exactly (minimum bar 2), then `mle`, then `reinforce`.
3. **Then the four-way cross**: {uniform, hbs, hier, cas} x {heuristic, mle, reinforce}.
   Nothing on their branch exercises this — their configs are all uniform-forward.

## Open problems carried from the review

- **Clash 2 (unresolved, needs a decision).** Their `P_B` conditions on
  `model.encode_synthon` (`bck/policy.py:197`), which HBS and CAS removed from the forward
  action path, so under those arms the synthon encoder gets no TB gradient and
  `detach_shared=False` makes the backward objective its only trainer. Recommendation on
  record: feed `P_B` the fixed fingerprint instead, so both policies describe the same
  object on every arm. Not yet implemented.
- **Clash 3.** Their step cost is 9-16s against 5s heuristic. Our jnk3 HBS run is
  4.55 s/step, 1,260 steps, 95 min for the 50k budget -- already two thirds of
  `short-unkillable`'s 3h cap. A 2-3x step cost does not fit; plan for `long` or a
  smaller budget.
- **Clash 4.** `load_state_dict` is called without `strict=False`, and the head is
  conditional, so a heuristic run cannot resume as learned or vice versa, and existing
  checkpoints cannot resume once a head exists. Consider always creating the head
  (zero-initialised, unused under `heuristic`) to keep state dicts uniform.
- Their EMA is identical to ours (`zip(..., strict=True)`) and is satisfied because both
  models are built with `cfg.backward`. Keep it that way: create the head in
  `RxnFlowModel.__init__`, never with `add_module` after the fact the way HBS attaches
  `hbs_query_projection`. `BackwardPolicy` is built on `self.model`, so the EMA copy's
  head is unused.
