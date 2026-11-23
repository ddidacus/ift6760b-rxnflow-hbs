"""RxnFlow policy that searches the whole building-block space, without subsampling.

Upstream scores a synthon action as an inner product between a query the policy forms
from the state (and reaction) and a *learned* embedding of the synthon, and keeps that
affordable by scoring K of each library's N rows drawn uniformly, correcting the
partition estimate with log(N/K). That correction is the price of not looking at the
other N-K rows.

This policy removes the subsample. The reservoir -- every row of every library -- is
hashed once at the start of a run in *fixed* feature space (``hbs/reservoir.py``:
precomputed fingerprints, L2-normalized, so the vectors never move and the table is
never rebuilt). An action is scored by hashing its query into the same space, opening
the bucket the query falls in, and taking inner products against that bucket's rows
only. There is no K and no importance correction: the bucket *is* the candidate set and
the normalization over it is exact.

Learning stays on the query side. Upstream's action heads are kept as they are, and one
``Linear(synthon_dim -> feature_dim)`` owned by the model projects their output into the
fixed feature space, so the inner product being ranked and the angle the hash estimates
live in the same space. Because only the query moves, the index stays valid for the
whole run, and nothing in the action path has to be re-encoded.

What this costs, against upstream:

* The synthon encoder leaves the action path. A logit is now linear in the fingerprint
  bits with state-dependent coefficients, where upstream's was ``query . MLP(bits)``.
  That is strictly less expressive, and it is the deliberate trade for a static index.
* Candidate sets are query-dependent and uneven. Bucket occupancy in fingerprint space
  is clustered, so one state may score a few dozen rows and another a few thousand.
  ``min_rows`` only guards the degenerate case of a bucket too small to score at all.
* Reachability is no longer guaranteed. A row outside the query's bucket cannot be
  selected at that step, where uniform subsampling gave every row probability K/N.
  ``get_action_logits`` still scores an observed action directly, so replayed and
  retrosynthesis-derived trajectories keep a finite forward logit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from rxnflow.config import Config
from rxnflow.core.types import (
    Action,
    ActionKey,
    ActionSubspace,
    ActionType,
    State,
)
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import PROPERTY_DIM
from rxnflow.envs.graph import GraphBatch
from rxnflow.gflownet.policy import (
    ActionCategorical,
    ActionLogits,
    RxnFlowPolicy,
    SubsamplingPolicy,
)
from rxnflow.gflownet.property_penalty import get_property_penalty, get_size_mask
from rxnflow.models import RxnFlowModel

from hbs.metrics import TrajectoryMetrics
from hbs.reservoir import HashedReservoir

logger = logging.getLogger(__name__)

_REACTION_TYPES = (
    ActionType.UNIRXN_TRANSFORM,
    ActionType.UNIRXN_TERMINAL,
    ActionType.BIRXN_BRICK,
    ActionType.BIRXN_LINKER,
)


@dataclass
class HBSSettings:
    """Settings for bucket-based candidate selection over the fixed reservoir."""

    # Hyperplanes, so 2^depth cells: the resolution of the search, one value for every
    # library. Deep codes make buckets small -- at 16 the median bucket holds a single
    # row -- so depth sets how finely the space is cut and min_rows sets how much of it
    # a step actually scores.
    depth: int = 16
    # Candidates to collect before stopping: the query's own bucket first, then its
    # Hamming neighbours until this many rows are in hand. Buckets are always opened
    # nearest-first, so a small bucket beside the query is never skipped for being
    # small -- this floor only decides how much further out the search reaches. A large
    # floor therefore adds *more distant* blocks rather than rescuing near ones, and a
    # small floor keeps the offer tight around the query. At depth 16 over 48,739 blocks
    # the median bucket holds one row, so this value sets the candidate budget almost
    # exactly. It needs the jitter (noise_sigma) to supply coverage over time, since a
    # tight offer repeated every step is what collapsed catalog usage to 2.3%.
    min_rows: int = 16
    # Calls between retrieval-quality probes against the exact inner products; 0 off.
    probe_every: int = 200
    # Seed for the shared hyperplanes.
    seed: int = 0
    # Scale of the deterministic jitter added to the probe *before hashing only*, in
    # units of the probe's own norm. This is the one knob that gives the offered block
    # set temporal variation: at 0 the set is a deterministic function of the state and
    # changes only as the policy drifts, which is what collapsed the catalog coverage to
    # 2.3% of the largest group against the baseline's 32%. The jitter is a function of
    # (seed, step, group), never fresh randomness, so sampling and the loss within one
    # step open the same buckets and the normalization over them stays exact.
    noise_sigma: float = 0.0
    noise_seed: int = 0
    # L2-normalize the probe before the dot product, making a synthon logit a pure
    # cosine and leaving logit_scale as the only temperature. Off by default, which
    # is upstream's convention (it normalizes the synthon side, not the query), but
    # that convention assumes nothing downstream depends on the query's direction.
    # Here the hash sees direction only, so an unnormalized probe can absorb gradient
    # in its magnitude while its direction drifts out of the cone the non-negative
    # fingerprints occupy -- measured in this run at step 200, cos_best fell 0.139 ->
    # 0.059 and bucket recall 2.9% -> 1.2% together. Normalizing couples every unit of
    # gradient to the direction retrieval actually uses.
    normalize_probe: bool = False
    # Hash about the reservoir's centroid instead of the origin. Off by default:
    # the logit ranks cosine about the origin, so that is what the hash should
    # estimate. See HashedReservoir.
    center: bool = False

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("hbs.depth must be positive")
        if self.noise_sigma < 0:
            raise ValueError("hbs.noise_sigma must be non-negative (0 disables)")
        if self.min_rows < 1:
            raise ValueError("hbs.min_rows must be positive")
        if self.min_rows < 64 and self.noise_sigma <= 0:
            logger.warning(
                "hbs.min_rows=%d with no jitter offers the same ~%d blocks every step; "
                "set --hbs-noise-sigma so the offer moves",
                self.min_rows, self.min_rows,
            )
        if self.probe_every < 0:
            raise ValueError("hbs.probe_every must be non-negative (0 disables)")
class StepClock:
    """Counts sampling calls, so the probe jitter is a function of the step.

    The sampling policy advances it at the start of each batch; the training policy
    reads the same value when it scores that batch, so both open the same buckets.
    """

    def __init__(self, value: int = 0):
        self.value = int(value)

    def advance(self) -> None:
        self.value += 1


class MetricsPolicyMixin:
    """Times lockstep sampling and exposes the candidate sets, changing nothing.

    A policy mixes this in and calls ``_record_candidates`` at the end of its
    ``get_action_categorical``. Every hook is inert until a collector is attached,
    and none of them touches logits, the action distribution or the trainer's RNG,
    so both arms sample exactly what they would sample without it.
    """

    metrics: TrajectoryMetrics | None = None
    _measuring: list[State] | None = None

    def attach_metrics(self, collector: TrajectoryMetrics | None) -> None:
        self.metrics = collector

    def sample_from_model(self, *args, **kwargs):
        clock = getattr(self, "clock", None)
        if clock is not None:
            clock.advance()
        if self.metrics is None:
            return super().sample_from_model(*args, **kwargs)
        self.metrics.begin_sampling()
        try:
            return super().sample_from_model(*args, **kwargs)
        finally:
            self.metrics.end_sampling()

    def sample_actions(self, states: list[State], *args, **kwargs):
        if self.metrics is None:
            return super().sample_actions(states, *args, **kwargs)
        self.metrics.begin_iteration()
        # Identity, not equality: only the categorical built for this very call is
        # a sampling-time candidate set. log_prob() builds its own for the loss.
        self._measuring = states
        try:
            return super().sample_actions(states, *args, **kwargs)
        finally:
            self._measuring = None
            self.metrics.end_iteration()

    def _record_candidates(
        self, states: list[State], categorical: ActionCategorical
    ) -> None:
        if self.metrics is not None and self._measuring is states:
            self.metrics.record_candidates(states, categorical)


class InstrumentedPolicy(MetricsPolicyMixin, RxnFlowPolicy):
    """Upstream policy with the metric hooks and no behavioural change."""

    def get_action_categorical(
        self, states: list[State], *args, **kwargs
    ) -> ActionCategorical:
        categorical = super().get_action_categorical(states, *args, **kwargs)
        self._record_candidates(states, categorical)
        return categorical


class HBSPolicy(MetricsPolicyMixin, RxnFlowPolicy):
    """Full-space action selection by hashed lookup instead of subsampling."""

    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        rng: np.random.Generator,
        settings: HBSSettings,
        reservoir: HashedReservoir,
        projection: torch.nn.Module,
        clock: StepClock | None = None,
    ):
        super().__init__(env, model, config, device, rng)
        self.clock = clock if clock is not None else StepClock()
        self._library_order = {name: i for i, name in enumerate(sorted(env.synthons))}
        self.hbs = settings
        self.reservoir = reservoir
        # Owned by the model so the optimizer and the EMA copy both see it.
        self.projection = projection
        # Every library is scored over a bucket, never over a sample, so there is no
        # inclusion probability to correct for: log(N/N) = 0 leaves log_partition and
        # the sampler's importance term untouched, and the normalization over the
        # opened bucket is exact.
        self.subsampling = {
            name: SubsamplingPolicy(len(library), 1.0, len(library), rng)
            for name, library in env.synthons.items()
        }
        self._calls = 0
        self._probe_library = max(env.synthons, key=lambda n: len(env.synthons[n]))
        # cos(projected query, fingerprint of the action actually selected), averaged
        # over the last scored batch. This is the quantity TB's numerator pulls on, so
        # it is the direct answer to "is the probe learning the reservoir's space" --
        # and unlike a probe it costs nothing and sees every action in the batch.
        self.alignment: float | None = None
        # Mean number of code bits that changed since the previous call, which is how
        # the jitter scale should actually be read.
        self._drift: list[float] = []
        self.bits_flipped: float | None = None

    def _probe_vector(self, projected: torch.Tensor) -> torch.Tensor:
        """The query as both scored and hashed, so the two cannot diverge."""
        if self.hbs.normalize_probe:
            return F.normalize(projected, dim=-1)
        return projected

    # ======== Candidate selection

    def _hash_jitter(self, library_name: str) -> NDArray[np.float32] | None:
        """Jitter for this (step, group), scaled to the probe's own magnitude."""
        if self.hbs.noise_sigma <= 0:
            return None
        rng = np.random.default_rng(
            [self.hbs.noise_seed, self.clock.value, self._library_order[library_name]]
        )
        return rng.standard_normal(self.reservoir.dim).astype(np.float32)

    def _lookup(
        self, library_name: str, probe: NDArray[np.float32]
    ) -> list[NDArray[np.int64]]:
        """Rows of the bucket each state's jittered probe falls into.

        The jitter moves which blocks are offered; the caller scores them with the
        clean probe, so the policy's preferences carry no added variance.
        """
        jitter = self._hash_jitter(library_name)
        if jitter is None:
            hashed = probe
        else:
            scale = self.hbs.noise_sigma * np.linalg.norm(probe, axis=1, keepdims=True)
            hashed = probe + scale * jitter[None, :] / np.sqrt(self.reservoir.dim)
        self._record_code_drift(library_name, probe, hashed)
        return self.reservoir.lookup(library_name, hashed, self.hbs.min_rows)

    def _record_code_drift(
        self, library_name: str, clean: NDArray[np.float32], hashed: NDArray[np.float32]
    ) -> None:
        """Bits by which the jitter moved each probe's bucket code.

        Measured within one call, between the same probes hashed with and without the
        jitter -- not between successive calls, which involve different molecules and
        would report state-to-state variation instead. At sigma 0 this is exactly 0.
        """
        if self.hbs.noise_sigma <= 0:
            self.bits_flipped = 0.0
            return
        differing = self.reservoir.codes_of(library_name, clean) ^ self.reservoir.codes_of(
            library_name, hashed
        )
        depth = self.reservoir.index[library_name].depth
        bits = sum(int(((differing >> np.int64(b)) & 1).sum()) for b in range(depth))
        self._drift.append(bits / len(differing))
        self.bits_flipped = float(np.mean(self._drift[-256:]))

    def _probe_due(self, library_name: str) -> bool:
        if not self.hbs.probe_every or library_name != self._probe_library:
            return False
        return self._calls % self.hbs.probe_every == 0

    def _probe(
        self,
        library_name: str,
        probe: NDArray[np.float32],
        candidates: list[NDArray[np.int64]],
    ) -> None:
        """Report whether the opened bucket actually holds the best rows.

        Scored against the exact inner products in the same fixed space the index is
        built in, so this measures the search and nothing else -- there is no stale
        embedding to confound it, which was the whole point of fixing the features.
        """
        scores = self.reservoir.exact_scores(library_name, probe)
        # How close the probe sits to the fingerprints it is meant to retrieve. This
        # is the question behind the whole design: the head has to learn to emit a
        # direction in fingerprint space, and the hash can only help once it does.
        best_cosine = float(scores.max(axis=1).mean())
        mean_cosine = float(scores.mean())
        total = scores.shape[1]
        hit_1 = hit_k = 0.0
        scanned = 0.0
        for row, rows_opened in enumerate(candidates):
            opened = set(rows_opened.tolist())
            scanned += len(opened)
            order = np.argsort(-scores[row])
            budget = max(1, min(len(opened), total))
            hit_1 += int(order[0]) in opened
            hit_k += len(opened & set(order[:budget].tolist())) / budget
        count = len(candidates)
        logger.info(
            "HBS probe (library %s, N=%d, %d queries): best row in bucket %.1f%% "
            "(chance %.1f%%), its own top rows recalled %.1f%%, scanned %.0f of %d "
            "rows (%.1f%%), probe-fingerprint cosine best %.4f / mean %.4f",
            library_name,
            total,
            count,
            100 * hit_1 / count,
            100 * scanned / count / total,
            100 * hit_k / count,
            scanned / count,
            total,
            100 * scanned / count / total,
            best_cosine,
            mean_cosine,
        )

    # ======== Action distribution

    def get_action_categorical(
        self,
        states: list[State],
        batch: GraphBatch,
        state_emb: torch.Tensor,
        logit_scale: torch.Tensor,
        *,
        apply_property_penalty: bool = False,
    ) -> ActionCategorical:
        self._calls += 1
        base_spaces = [self.env.get_action_space(state) for state in states]
        state_groups: dict[int | None, list[int]] = {}
        for state_i, state in enumerate(states):
            assert base_spaces[state_i], f"state {states[state_i].smiles} has no actions"
            state_groups.setdefault(state.attachment_type, []).append(state_i)

        # 1. Form every query first and project it into the reservoir's space: the
        # lookup needs the probe that the logits will be formed from.
        plans = []
        for rows in state_groups.values():
            indices = torch.tensor(rows, device=self.device)
            emb = state_emb[indices]
            scale = logit_scale[indices, None]
            subspaces = {
                subspace.name: subspace
                for state_i in rows
                for subspace in base_spaces[state_i]
            }
            spaces_by_action: dict[tuple[ActionType, str], list[ActionSubspace]] = {}
            for subspace in subspaces.values():
                key = (subspace.action_type, subspace.name[0])
                spaces_by_action.setdefault(key, []).append(subspace)

            queries: dict[tuple[ActionType, str], torch.Tensor] = {}
            for action_type in _REACTION_TYPES:
                names = [name for kind, name in spaces_by_action if kind == action_type]
                if names:
                    batched = self.model.forward_reactions(emb, names, action_type)
                    for name, query in zip(names, batched.unbind(0), strict=True):
                        queries[action_type, name] = query
            for kind, name in list(spaces_by_action):
                if kind.is_first:
                    queries[kind, name] = self.model.forward_first_synthon(emb)
            # Unimolecular heads emit a scalar logit; only synthon queries are probes.
            probes = {
                key: self._probe_vector(self.projection(query))
                for key, query in queries.items()
                if not key[0].is_unirxn
            }
            plans.append((rows, indices, scale, spaces_by_action, queries, probes))

        # 2. Open each state's bucket, on one transfer of the group's probes.
        opened: list[dict[ActionKey, NDArray[np.int64]]] = [{} for _ in states]
        union: dict[str, list[NDArray[np.int64]]] = {}
        for rows, _, _, spaces_by_action, _, probes in plans:
            keys = list(probes)
            if not keys:
                continue
            stacked = torch.cat([probes[key] for key in keys]).detach()
            on_cpu = stacked.to(torch.float32).cpu().numpy()
            per_key = dict(zip(keys, np.split(on_cpu, len(keys)), strict=True))
            for key in keys:
                probe = per_key[key]
                for subspace in spaces_by_action[key]:
                    library_name = subspace.name[1]
                    assert library_name is not None
                    found = self._lookup(library_name, probe)
                    for local, state_i in enumerate(rows):
                        opened[state_i][subspace.name] = found[local]
                        union.setdefault(library_name, []).append(found[local])
                    if self._probe_due(library_name):
                        self._probe(library_name, probe, found)

        # 3. Gather the fixed features of what this batch opened, once per library.
        rows_used = {
            name: np.unique(np.concatenate(chunks)) for name, chunks in union.items()
        }
        cache = self._gather_libraries(rows_used)
        positions = {}
        for library_name, used in rows_used.items():
            lookup = np.full(len(self.env.synthons[library_name]), -1, dtype=np.int64)
            lookup[used] = np.arange(len(used))
            positions[library_name] = lookup

        # 4. Score as upstream does -- one matrix product per action type against the
        # gathered rows -- then take each state's own columns out of that product.
        action_logits: list[list[ActionLogits]] = [[] for _ in states]
        for rows, indices, scale, spaces_by_action, queries, probes in plans:
            library_names = list(
                dict.fromkeys(
                    subspace.name[1]
                    for spaces in spaces_by_action.values()
                    for subspace in spaces
                    if subspace.name[1] is not None
                )
            )
            library_masks: dict[str, torch.Tensor] = {}
            if apply_property_penalty and library_names:
                prop = torch.cat([cache[name][1] for name in library_names])
                size = torch.cat([cache[name][2] for name in library_names])
                size_mask = get_size_mask(
                    batch.num_heavy_atoms[indices, None], size, self.env.max_atoms
                )
                property_penalty = get_property_penalty(
                    batch.graph_features[indices], prop, self.env.property_limits
                )
                widths = [len(cache[name][0]) for name in library_names]
                masks = (size_mask & property_penalty).split(widths, dim=1)
                library_masks = dict(zip(library_names, masks, strict=True))

            group_logits: dict[ActionKey, tuple[torch.Tensor, ...]] = {}
            for key, spaces in spaces_by_action.items():
                if key[0].is_unirxn:
                    group_logits[spaces[0].name] = (queries[key] * scale).unbind(0)
                    continue
                libraries = [subspace.name[1] for subspace in spaces]
                features = torch.cat([cache[name][0] for name in libraries])
                logits = (probes[key] @ features.T) * scale
                if apply_property_penalty:
                    mask = torch.cat([library_masks[name] for name in libraries], dim=1)
                    logits = logits.masked_fill(~mask, -torch.inf)
                widths = [len(cache[name][0]) for name in libraries]
                for subspace, values in zip(
                    spaces, logits.split(widths, dim=1), strict=True
                ):
                    library_name = subspace.name[1]
                    columns = np.stack(
                        [
                            positions[library_name][opened[state_i][subspace.name]]
                            for state_i in rows
                        ]
                    ) if len({len(opened[i][subspace.name]) for i in rows}) == 1 else None
                    if columns is not None:
                        gather = torch.from_numpy(columns).to(self.device)
                        group_logits[subspace.name] = values.gather(1, gather).unbind(0)
                    else:
                        # Buckets differ in size between states, so each state takes
                        # its own columns; there is no rectangular gather to batch.
                        group_logits[subspace.name] = tuple(
                            values[local].index_select(
                                0,
                                torch.from_numpy(
                                    positions[library_name][opened[state_i][subspace.name]]
                                ).to(self.device),
                            )
                            for local, state_i in enumerate(rows)
                        )

            for local, state_i in enumerate(rows):
                action_logits[state_i] = [
                    ActionLogits(
                        ActionSubspace(
                            subspace.name,
                            subspace.action_type,
                            subspace.num_actions,
                            opened[state_i][subspace.name],
                        )
                        if subspace.name[1] is not None
                        else subspace,
                        group_logits[subspace.name][local],
                    )
                    for subspace in base_spaces[state_i]
                ]

        # Candidate-set diversity is deliberately *not* recorded here. It answers
        # "is the policy offered library-representative variety", which is a question
        # about the baseline: a low value there means uniform subsampling handed the
        # policy a poor slice to choose from. For retrieval a low value is the
        # mechanism succeeding, not a defect, so the same number read across arms
        # would invert. Retrieval quality is measured by _probe instead, against the
        # exact inner products. Latency and route metrics are still collected for
        # both arms, by the trainer, and are comparable.
        return ActionCategorical(action_logits, self.subsampling)

    def get_action_logits(
        self, state_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score the observed edges, which need not lie in any opened bucket.

        The body is ``RxnFlowPolicy.get_action_logits`` with the synthon encoder
        replaced by the fixed features, so a replayed or retrosynthesis-derived action
        keeps a finite forward logit even when the current query would not retrieve it.
        """
        by_reaction: dict[tuple[ActionType, str], list[int]] = {}
        by_library: dict[str, list[int]] = {}
        for row, action in enumerate(actions):
            name = "" if action.action_type.is_first else action.reaction
            by_reaction.setdefault((action.action_type, name), []).append(row)
            if action.library_name is not None:
                by_library.setdefault(action.library_name, []).append(row)

        feature_rows: list[int] = []
        gathered: list[torch.Tensor] = []
        for library_name, rows in by_library.items():
            indices = np.array(
                [actions[i].synthon_index for i in rows], dtype=np.int64
            )
            feature_rows.extend(rows)
            gathered.append(self._gather_libraries({library_name: indices})[
                library_name
            ][0])
        features = state_emb.new_zeros((len(actions), self.reservoir.dim))
        if gathered:
            features = features.index_copy(
                0,
                torch.tensor(feature_rows, device=self.device),
                torch.cat(gathered),
            )

        logits = state_emb.new_zeros(len(actions))
        cosines: list[torch.Tensor] = []
        for (action_type, action_name), rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            emb = state_emb[indices]
            if action_type.is_unirxn:
                # Inherited, so the scalar head is called exactly as upstream calls it.
                values = self._unirxn_logits(action_type, action_name, emb).squeeze(1)
            else:
                if action_type.is_first:
                    query = self.model.forward_first_synthon(emb)
                else:
                    query = self.model.forward_birxn(emb, action_name, action_type)
                probe = self._probe_vector(self.projection(query))
                values = (probe * features[indices]).sum(1)
                with torch.no_grad():
                    # features are already unit-norm, so this is the cosine.
                    cosines.append(
                        (F.normalize(probe, dim=-1) * features[indices]).sum(1)
                    )
            logits = logits.index_copy(0, indices, values)
        if cosines:
            self.alignment = float(torch.cat(cosines).mean())
        return logits * logit_scale

    # ======== Fixed features

    def _gather_libraries(
        self, samples: dict[str, NDArray[np.int64]]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fixed features, properties and sizes for the given rows of each library.

        The body is ``RxnFlowPolicy._get_synthon_cache`` with one line changed: where
        upstream encodes the rows with the learned synthon encoder, this normalizes the
        precomputed fingerprints. Nothing here carries gradients -- the reservoir is a
        constant, and the only learned part of a synthon logit is the query projection.
        """
        if not samples:
            return {}
        widths = [len(indices) for indices in samples.values()]
        count = sum(widths)
        dim = self.reservoir.dim
        fp_arr = np.empty((count, dim), dtype=np.uint8)
        prop_arr = np.empty((count, PROPERTY_DIM), dtype=np.float32)
        size_arr = np.empty(count, dtype=np.uint8)
        offset = 0
        for library_name, indices in samples.items():
            library = self.env.synthons[library_name]
            rows = slice(offset, offset + len(indices))
            np.take(library.fingerprints, indices, axis=0, out=fp_arr[rows], mode="clip")
            np.take(library.properties, indices, axis=0, out=prop_arr[rows], mode="clip")
            np.take(library.heavy_atoms, indices, out=size_arr[rows], mode="clip")
            offset += len(indices)

        fp = torch.from_numpy(fp_arr).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prop = torch.from_numpy(prop_arr).to(self.device, non_blocking=True)
        size = torch.from_numpy(size_arr).to(self.device, non_blocking=True)
        features = F.normalize(fp, dim=-1)
        return {
            library_name: (feature_split, prop_split, size_split)
            for library_name, feature_split, prop_split, size_split in zip(
                samples,
                features.split(widths),
                prop.split(widths),
                size.split(widths),
                strict=True,
            )
        }


