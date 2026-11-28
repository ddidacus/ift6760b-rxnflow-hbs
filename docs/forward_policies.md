# Forward policies: HBS and CAS

How this fork replaces RxnFlow's uniform action subsampling, what has been tried, what
failed, and what is in the code today.

Every number below is labelled by where it comes from:
**[measured]** from a completed run, **[computed]** from a closed-form argument,
**[untested]** believed but not yet checked.

## The problem both arms attack

Upstream scores a synthon action as `query . MLP(synthon)` and keeps it affordable by
scoring `K` of each library's `N` rows drawn uniformly, correcting the partition with
`log(N/K)`. On the largest jnk3 library that is 9,281 of 92,813 rows. The correction is
the price of never looking at the other rows: the policy cannot prefer a block it was
not offered, and the offer is independent of the state.

Both arms replace the uniform draw with a **retrieval** step over an index built once
from the environment's precomputed fingerprints, which never change during a run. The
learned part is a single `Linear(synthon_dim -> feature_dim, bias=False)` registered on
the model as `hbs_query_projection` — the *probe* — which maps an action head's query
into the fixed feature space. A row's logit is then `probe . fingerprint`.

Trajectory balance is the only loss. The residual is one scalar per trajectory,

```
tb_error = log_Z + sum_t log P_F(a_t|s_t) - sum_t log P_B - beta * log R(x)
```

with the reward evaluated on the **terminal** molecule only (`gflownet/trajectory_balance.py`).
There is no `F(s,a)` head: the flow is implicit in the TB constraint. Consequently all
credit assignment for retrieval is trajectory-level, which is the root of everything in
the HBS section below.

## HBS — hashed buckets

`fwd/hbs/reservoir.py`, `fwd/hbs/policy.py`, `fwd/hbs/trainer.py`.

Every row of every library is hashed once with `depth` random orthogonal hyperplanes
(Super-Bit LSH), giving each row a `depth`-bit code. A query is hashed into the same
space and scores the rows of the bucket it lands in.

### What failed

**Pure-bucket scoring (abandoned).** The first version scored the query's bucket and
declared to the loss that it had scored the whole library, handing upstream a
`SubsamplingPolicy` whose `log_importance` is `log(N/N) = 0`. On the largest library that
is 256 of 92,813 rows: 0.3% scanned, 100% declared. TB does not complain — it solves for
`log_Z` instead, and `tb_error` looked *better* than uniform's — while `log_Z` absorbed
the error: **−12.3 against uniform's +3.2, a ~15.5 nat gap [measured, jnk3]**, and 12% of
the top-10 AUC. Reachable with `--hbs-pure-bucket`, kept only to reproduce those runs.

**A deterministic offer collapses coverage.** With the candidate set a deterministic
function of the state, catalog usage fell to **2.3% of the largest group against the
baseline's 32% [measured]**. The fix was `--hbs-noise-sigma`, a deterministic jitter added
to the probe *before hashing only*, a function of `(seed, step, group)` so sampling and
the loss open the same buckets within a step. Read it through `code_bits_flipped`.

**Depth 16 cannot resolve a nearest neighbour, at any probe quality.** For a SimHash each
bit agrees with probability `1 - theta/pi`, so an exact `d`-bit match has probability
`(1 - theta/pi)^d`. At the probe cosine HBS actually reaches:

| probe cosine | d=8 | d=10 | d=12 | d=16 |
|---|---|---|---|---|
| 0.26 (observed) | 1.4e-2 | 4.6e-3 | 1.6e-3 | **1.8e-4** |
| 0.60 | 6.1e-2 | 3.0e-2 | 1.5e-2 | 3.7e-3 |
| 0.95 | 4.3e-1 | 3.5e-1 | 2.8e-1 | 1.8e-1 |

A 50% chance of an exact match needs cosine **0.991 at d=16, or 0.966 even at d=8**
[computed]. Depth is not an escape. The matching measurement: over 1,260 steps the probe's
cosine to the best row improved **0.155 -> 0.262**, while the best row was in the query's
bucket **0.00% of the time against 0.30% chance**, and recall of its own top rows ended at
**0.33% while the scan covered 0.276% of the library [measured, `pmo_jnk3_hbs_sig080_s4`]**.
Retrieval was indistinguishable from drawing 256 rows at random.

**`normalize_probe` is not the binding constraint.** The standing hypothesis was that with
an unnormalized probe TB cuts its error by growing `||probe||` — a radial sink the hash
cannot see, since it reads direction only. Two observations weigh against it: the cosine
*did* improve over that run without normalization, and `||probe||` starts at ~11 from
Xavier init and was flat over the first 60 steps (11.44 -> 11.47, 10.59 -> 10.44)
[measured]. A 2x2 A/B (`scripts/sweep_normalize_probe.sh`) was launched and stopped early
by choice; the arm is **[untested]** past step 60, but retrieval sat at chance in both.

### What worked

**The hybrid candidate set.** The bucket *plus* a uniform draw, each row carrying its own
log importance weight (`fwd/sampling.py`): bucket rows weigh 0, having been certain;
uniform rows outside the bucket each stand in for `(N - |B|) / |U \ B|` rows. This made the
partition honest again and restored reachability — every row keeps probability >= K/N at
every step. It is on by default; turning it off is provably biased.

**Keeping the index static.** Because only the query moves, the index built before step 1
stays valid for the whole run, and nothing in the action path is re-encoded. The cost,
accepted deliberately: the synthon encoder leaves the action path, so a logit is linear in
the fingerprint bits where upstream's was `query . MLP(bits)`.

## CAS — clustered action sampling

`fwd/cas/index.py`, `fwd/cas/policy.py`, `fwd/cas/trainer.py`, `fwd/cas/fullspace.py`.

Same probe, but the partition is **fitted** rather than random: spherical k-means over the
fixed features, with occupancy fixed (`rows_per_cell`, default 235) rather than `nlist`,
because libraries run from 12 rows to 92,813.

### What worked

**A fitted partition retrieves where the hash did not.** At probe cosine 0.21 a k-means
cell contains the target row **23.8% of the time against 0.0% for an 11-bit SimHash cell of
the same size**, with balanced occupancy (**4.5x** median-to-max against **15.9x**) and no
empty cells (against **19.6%** at d=16) [measured]. This is the single largest effect
recorded in this project, and it is a property of the partition, not of training.

**Clamping `nlist` so a cell never exceeds upstream's K.** Below ~350 rows `nlist` collapses
to 1, making the "cell" the whole library, so CAS would score *more* than upstream: 320 of
388 libraries fall there and turn a 12.7x saving on the large libraries into a **1.03x wash
overall**. With the clamp the catalog-wide figure is **2.17x [measured]**.

**Treating the cell as the action space.** `log P(row) = l_row - logsumexp over the cell`,
so `P_F` is exactly normalized over what the policy can choose, with no estimate anywhere.

### What failed

**The cell as a *sample* of the library.** Then the partition must estimate the library's
sum as `Z_cell / P(cell)`, which is Horvitz–Thompson and unbiased *only if the cell is
drawn with probability `P(cell)`*. `_open_cells` takes the argmax, which biased it by
**+0.47 nats at cosine 0.15, +0.63 at 0.21, +1.13 at 0.40 — growing with probe quality**,
so it taxed exactly the improvement the probe was trying to make. Drawing the cell removed
the bias (**±0.004 nats**) and destroyed retrieval: the drawn cell held the nearest row
**5% of the time against the argmax cell's 24%** [all measured]. Recorded conclusion:
*cell commitment and estimator validity are one knob pulled in opposite directions.* This
is the result to read before re-proposing a sampled bucket at fixed temperature.

**A global cell is mostly illegal.** One index over all 482,070 rows is the intended
design, but a global 235-row cell holds about **0.2 chemically legal rows for the median
state** (a uniform pick is legal 8.7% of the time action-weighted, 0.09% at the median
step) [measured], so almost every state would have no executable action. Hence two
variants: **CAS-Validity** keeps the per-library division and every offered row is legal by
construction; **CAS-FullSpace** clusters the whole catalog and reports `legal_fraction`
instead of enforcing it.

## What is implemented today

Both arms run through `python -m hbs.train` (see `hbs/configs/`, `scripts/`), and all four
are one flag apart: no flag (uniform), `--hbs`, `--cas`, `--cas --cas-fullspace`.
`examples/hbs_qed.py` runs any of them on QED with no reward dependencies.

### Differentiable bucket selection (new)

`--hierarchical` replaces the Hamming walk and the uniform draw with **one bucket per
state, scored by a differentiable similarity**. For hyperplane scores `u = Wq` and a bucket
code `c` in `{-1,+1}^d`, the similarity is `<c, u>`, and:

- `argmax` over **all** `2^d` codes is `sign(u)` — exactly the hash code [verified].
- `argmax` over **occupied** codes equals the hash bucket whenever it is non-empty (100% of
  such queries), and otherwise falls back to the **margin-weighted** nearest occupied
  bucket rather than by unweighted Hamming distance [verified]. In a 95-of-256-occupied
  test only 35.6% of queries hashed to an occupied bucket, so the fallback is the common
  case and is strictly better informed than Hamming.
- Gradient reaches the probe through it: grad norm 5.76 with all 2,048 entries non-zero,
  scaling 1.53 / 5.76 / 18.3 as the temperature goes 4.0 / 1.0 / 0.25 [verified].

`sign` itself is piecewise constant (zero gradient a.e.) and the old path also detached the
probe before hashing, so there were two independent reasons nothing flowed.

**Pair `--hierarchical` with `--hbs-depth 8`.** At depth 16 the median bucket holds one
row, so `P(row|bucket)` is degenerate; the settings emit a warning above 12.

### Temperatures and annealing

`--temp-bucket` / `--temp-nn` set the bucket and within-bucket row softmax temperatures,
`--temp-bucket-end` / `--temp-nn-end` with `--anneal-steps` move them geometrically from
high to low over training. Both apply to HBS and CAS; for CAS `--cas-tau` is the bucket
temperature's start. The row temperature is applied on **both** sides of
`log_prob = get_action_logits - log_partition`, so numerator and denominator stay
consistent. Annealing reads the shared `StepClock`, so the sampling and loss passes of one
step agree.

### Where `log P(bucket)` enters: `--bucket-term`

The forward density is `P_F = P(bucket|q) * P(row|bucket,q)`. Since the bucket is chosen by
argmax rather than sampled, there are three defensible readings, and the flag selects one:

- `numerator` (default) — the term enters the numerator only, so `P_F` carries the pressure
  to open a *better* bucket. `P_F` then sums to `P(bucket)` over the opened rows rather
  than to 1, i.e. TB is being trained on a hierarchical model density while the behaviour
  policy is greedy. TB tolerates off-policy data, but this has **not** been validated
  against the CAS bias measurement above and is the thing to watch. **[untested]**
- `partition` — the term enters both sides, so `P_F` is exactly normalized over the opened
  set, but it then cancels within a subspace and only moves mass *between* subspaces.
- `off` — the cell-is-the-action-space reading, i.e. today's validated CAS semantics.

The three are confirmed distinct: after two identical steps from one seed the projection
weights differ by 1.5e-4 to 1.7e-4 across modes, for both HBS and CAS [verified], so the
term genuinely enters the gradient.

For a replayed action the numerator uses the bucket that *holds the observed row*, which
need not be the one the current query would open (`row_bucket` for HBS, `cell_of` for CAS,
`global_row` for CAS-FullSpace).

`partition` is rejected on **CAS-FullSpace**, which opens cells until it has `min_legal`
legal rows and so has no single opened cell to offset the partition with. The natural
generalization is `logsumexp` over the cells actually opened, but that is a choice about
what the action space is, not a default, so the settings raise rather than guess.
`numerator` and `off` both work there.

### Diagnostics

Written per measured step to `trajectory_metrics.jsonl`, and in prose to `training.log`:

- `probe_alignment` — cos(probe, fingerprint of the action actually chosen).
- `HBS probe ...` lines — bucket hit@1 against chance, recall, scanned fraction, cosine
  best/mean, and `probe norm` (1.0 when `--hbs-normalize-probe` is on; the radial component
  otherwise). `--hbs-probe-every` sets the cadence; it is a `no_grad` measurement.
- `cell_logp`, `cell_entropy` — CAS only, `-log P(opened cell)` and the whole
  distribution's entropy, against `log(nlist)` for "no preference". Diagnostic only under
  `--bucket-term off`.
- `legal_fraction`, `cells_walked`, `fallback_rate` — CAS-FullSpace only.
- Calibration: spearman(log P, R), TB slope, implied `log_Z` (`fwd/calibration.py`).

Note that the arms log under `rxnflow.gflownet.fwd.*`. `hbs/train.py` attaches handlers to
that logger as well as to `hbs`; without it those records reach no file, which silently
happened after the modules moved out of `hbs/`.

## Open questions, in the order worth attacking

1. **Does a trained bucket choice escape the geometry result?** The collision table says a
   *fixed* hash cannot resolve a neighbour at attainable cosines. It says nothing about a
   policy *learning* which of `2^8` buckets to open, which is what `--hierarchical` makes
   possible. This reframes the hash from "a similarity-preserving index" to "an arbitrary
   but fixed partition the policy learns a distribution over".
2. **Is `--bucket-term numerator` biased in practice, and by how much?** The CAS
   measurement gives the method for answering it: compare the partition against an exact
   sum over the library at several probe cosines.
3. **One vector, two jobs.** The same probe picks the bucket and ranks rows inside it.
   Separate heads would decouple them for almost nothing and would say whether the conflict
   is real. **[untested]**
4. **Does annealing recover commitment?** The CAS result that killed a sampled cell used a
   *fixed* temperature. Whether a trajectory from high to low temperature buys gradient
   early and commitment late is open; `cell_logp` and `cell_entropy` already measure it.
