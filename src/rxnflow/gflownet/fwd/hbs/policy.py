"""Action selection by hashed lookup over the whole catalog, instead of subsampling.

Candidates are the bucket the projected query hashes into plus a uniform draw, each
row carrying its own importance weight. Only the query moves, so the index built at the
start of a run stays valid for all of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
from rxnflow.gflownet.fwd.base import MetricsPolicyMixin, StepClock, Temperature
from rxnflow.gflownet.fwd.hbs.reservoir import HashedReservoir
from rxnflow.gflownet.fwd.sampling import (
    ComplementSampler,
    WeightedActionCategorical,
    uniform_budget,
)
from rxnflow.gflownet.policy import (
    ActionCategorical,
    ActionLogits,
    RxnFlowPolicy,
    SubsamplingPolicy,
)
from rxnflow.gflownet.property_penalty import get_property_penalty, get_size_mask
from rxnflow.models import RxnFlowModel

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

    # Hyperplanes, so 2^depth buckets: the resolution of the search. Deep codes make
    # buckets small, so depth cuts the space and min_rows sets what a step scores.
    depth: int = 16
    # Rows to collect before stopping, the query's bucket first and then its Hamming
    # neighbours. Buckets open nearest-first, so this sets how far out the offer
    # reaches, not which near blocks are kept. Pair a small value with noise_sigma.
    min_rows: int = 16
    # Bucket *plus* a uniform draw, each row importance-weighted (fwd/sampling.py).
    # Off is provably biased -- it cost ~15.5 nats of logZ on jnk3 -- and kept only to
    # reproduce the pre-fix runs.
    hybrid: bool = True
    # Rows in the uniform half. None takes upstream's K for each library, so the
    # uniform half *is* the baseline arm's estimator and logZ is comparable.
    uniform_rows: int | None = None
    # Weight applied to the row importances inside the *proposal* (not the
    # normalization). 0 proposes from the policy's raw preference over offered rows.
    explore_temp: float = 0.0
    # Calls between retrieval-quality probes against the exact inner products; 0 off.
    probe_every: int = 200
    # Seed for the shared hyperplanes.
    seed: int = 0
    # Jitter added to the probe before hashing only, in units of its own norm: the
    # one knob that moves the offered block set over time (at 0 it is a deterministic
    # function of the state). A function of (seed, step, group), never fresh
    # randomness, so sampling and the loss open the same buckets.
    noise_sigma: float = 0.0
    noise_seed: int = 0
    # L2-normalize the probe, so a logit is a pure cosine. Off is upstream's
    # convention, but the hash sees direction only: an unnormalized probe can absorb
    # gradient in its magnitude while its direction drifts out of the fingerprint cone.
    normalize_probe: bool = False
    # Hash about the reservoir centroid instead of the origin. Off: the logit ranks
    # cosine about the origin, so that is what the hash should estimate.
    center: bool = False
    # Score one bucket by <code, hyperplane scores> -- differentiable, and its argmax
    # over all codes is the hash code itself -- instead of walking a Hamming ball and
    # adding a uniform draw. min_rows, hybrid and uniform_rows are unused when on.
    hierarchical: bool = False
    # Where log P(bucket) enters the loss. "numerator" is P_F = P(bucket)P(row|bucket),
    # which is what gives the probe pressure to open a better bucket, and leaves P_F
    # summing to P(bucket) over the opened rows. "partition" adds it on both sides, so
    # P_F is exactly normalized over the opened set but the term only moves mass
    # between subspaces. "off" is the cell-is-the-action-space reading.
    bucket_term: str = "numerator"
    temp_bucket: Temperature = field(default_factory=Temperature)
    temp_nn: Temperature = field(default_factory=Temperature)

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("hbs.depth must be positive")
        if self.noise_sigma < 0:
            raise ValueError("hbs.noise_sigma must be non-negative (0 disables)")
        if self.min_rows < 1:
            raise ValueError("hbs.min_rows must be positive")
        if self.uniform_rows is not None and self.uniform_rows < 0:
            raise ValueError("hbs.uniform_rows must be non-negative or None")
        if self.explore_temp < 0:
            raise ValueError("hbs.explore_temp must be non-negative")
        if not self.hybrid:
            logger.warning(
                "hbs.hybrid is off: the partition will be estimated over the bucket "
                "only and declared as the whole library, which is biased. See "
                "fwd/sampling.py; this reproduces the pre-fix behaviour."
            )
        if self.min_rows < 64 and self.noise_sigma <= 0:
            logger.warning(
                "hbs.min_rows=%d with no jitter offers the same ~%d blocks every step; "
                "set --hbs-noise-sigma so the offer moves",
                self.min_rows, self.min_rows,
            )
        if self.probe_every < 0:
            raise ValueError("hbs.probe_every must be non-negative (0 disables)")
        if self.bucket_term not in ("numerator", "partition", "off"):
            raise ValueError("hbs.bucket_term must be numerator, partition or off")
        self.temp_bucket.validate("hbs.temp_bucket")
        self.temp_nn.validate("hbs.temp_nn")
        if self.hierarchical and self.depth > 12:
            logger.warning(
                "hbs.depth=%d with one bucket per state: the median bucket holds a "
                "single row, so P(row|bucket) is degenerate. Use depth 8-10.",
                self.depth,
            )


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
        # Left at log(N/N) = 0 and kept only for inherited code paths: with a hybrid
        # candidate set the correction is per row, and travels in
        # WeightedActionCategorical.partition_weights instead.
        self.subsampling = {
            name: SubsamplingPolicy(len(library), 1.0, len(library), rng)
            for name, library in env.synthons.items()
        }
        self._calls = 0
        # Uniform half of the candidate set, on its own generator so enabling the
        # hybrid does not shift the policy's rng stream.
        self._library_sizes = {name: len(lib) for name, lib in env.synthons.items()}
        self._complement = ComplementSampler(
            self._library_sizes, np.random.default_rng(rng.integers(2**63))
        )
        self._uniform_k = {
            name: (
                settings.uniform_rows
                if settings.uniform_rows is not None
                else uniform_budget(
                    size,
                    config.subsampling.sampling_ratio,
                    config.subsampling.min_sampling,
                )
            )
            for name, size in self._library_sizes.items()
        }
        self._probe_library = max(env.synthons, key=lambda n: len(env.synthons[n]))
        # cos(projected query, fingerprint of the selected action) over the last
        # batch: whether the probe is learning the reservoir's space.
        self._partition_cache: dict[str, tuple[torch.Tensor, ...]] = {}
        self._bucket_rows: dict[str, list[NDArray[np.int64]]] = {}
        # Mean log P(opened bucket) over the batch, as cell_logp is for CAS.
        self.bucket_logp: float | None = None
        self.alignment: float | None = None
        # Mean number of code bits that changed since the previous call, which is how
        # the jitter scale should actually be read.
        self._drift: list[float] = []
        self.bits_flipped: float | None = None

    def _row_temperature(self) -> float:
        """Temperature of the within-bucket row softmax at this step."""
        return self.hbs.temp_nn.at(self.clock.value) if self.hbs.hierarchical else 1.0

    def _partition(self, library_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """This library's hyperplanes-with-centroid and occupied codes, as tensors."""
        cached = self._partition_cache.get(library_name)
        if cached is None:
            planes, mean, signs = self.reservoir.partition_of(library_name)
            cached = (
                torch.as_tensor(planes, device=self.device),
                torch.as_tensor(mean, device=self.device),
                torch.as_tensor(signs, device=self.device),
            )
            self._partition_cache[library_name] = cached
            self._bucket_rows[library_name] = self.reservoir.rows_of_buckets(
                library_name
            )
        return cached

    def _bucket_log_probs(
        self, library_name: str, probe: torch.Tensor
    ) -> tuple[torch.Tensor, NDArray[np.int64]]:
        """log P(bucket | probe) over occupied buckets, and the argmax bucket.

        The similarity is <code, hyperplane scores>, which is differentiable in the
        probe and whose argmax over all codes is the hash code itself, so the bucket
        this opens is the one the hash would have opened.
        """
        planes, mean, signs = self._partition(library_name)
        unit = F.normalize(probe, dim=-1) - mean
        scores = (unit @ planes.T) @ signs.T
        chosen = scores.detach().argmax(dim=1)
        temperature = self.hbs.temp_bucket.at(self.clock.value)
        return F.log_softmax(scores / temperature, dim=1), chosen.cpu().numpy()

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

        Measured within one call, not between calls, which would report state-to-state
        variation instead. Exactly 0 at sigma 0.
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

        Scored against the exact inner products in the index's own space, so this
        measures the search and nothing else.
        """
        scores = self.reservoir.exact_scores(library_name, probe)
        # How close the probe sits to the fingerprints it is meant to retrieve.
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
        # Norm is 1 by construction when normalize_probe is on; when it is off this is
        # the radial component the hash cannot see, so it reads the free gradient sink.
        logger.info(
            "HBS probe (library %s, N=%d, %d queries): best row in bucket %.1f%% "
            "(chance %.1f%%), its own top rows recalled %.1f%%, scanned %.0f of %d "
            "rows (%.1f%%), probe-fingerprint cosine best %.4f / mean %.4f, "
            "probe norm %.4f",
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
            float(np.linalg.norm(probe, axis=1).mean()),
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
        row_weights: list[dict[ActionKey, NDArray[np.float32]]] = [{} for _ in states]
        union: dict[str, list[NDArray[np.int64]]] = {}
        bucket_logp: list[dict[ActionKey, torch.Tensor]] = [{} for _ in states]
        for rows, _, _, spaces_by_action, _, probes in plans:
            keys = list(probes)
            if not keys:
                continue
            if self.hbs.hierarchical:
                for key in keys:
                    for subspace in spaces_by_action[key]:
                        library_name = subspace.name[1]
                        assert library_name is not None
                        log_p, chosen = self._bucket_log_probs(
                            library_name, probes[key]
                        )
                        buckets = self._bucket_rows[library_name]
                        for local, state_i in enumerate(rows):
                            candidates = buckets[chosen[local]]
                            opened[state_i][subspace.name] = candidates
                            row_weights[state_i][subspace.name] = np.zeros(
                                len(candidates), dtype=np.float32
                            )
                            bucket_logp[state_i][subspace.name] = log_p[
                                local, chosen[local]
                            ]
                            union.setdefault(library_name, []).append(candidates)
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
                    # Bucket alone, so the probe measures the hash and not the
                    # uniform half.
                    if self._probe_due(library_name):
                        self._probe(library_name, probe, found)
                    # One draw per (library, subspace, group), as upstream also
                    # shares a draw; weights_for removes each state's own overlap.
                    drawn = (
                        self._complement.draw(
                            library_name, self._uniform_k[library_name]
                        )
                        if self.hbs.hybrid
                        else np.empty(0, dtype=np.int64)
                    )
                    for local, state_i in enumerate(rows):
                        candidates, weights = self._complement.weights_for(
                            library_name, found[local], drawn
                        )
                        opened[state_i][subspace.name] = candidates
                        row_weights[state_i][subspace.name] = weights
                        union.setdefault(library_name, []).append(candidates)

        if self.hbs.hierarchical:
            chosen_logp = [v for per_state in bucket_logp for v in per_state.values()]
            if chosen_logp:
                self.bucket_logp = -float(torch.stack(chosen_logp).mean().detach())

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
        partition_weights: list[list[torch.Tensor]] = [[] for _ in states]
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
                logits = (probes[key] @ features.T) * scale / self._row_temperature()
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
                entries: list[ActionLogits] = []
                weights: list[torch.Tensor] = []
                for subspace in base_spaces[state_i]:
                    logits = group_logits[subspace.name][local]
                    if subspace.name[1] is None:
                        # Unimolecular: one action, always offered, no correction.
                        entries.append(ActionLogits(subspace, logits))
                        weights.append(torch.zeros(len(logits), device=self.device))
                        continue
                    entries.append(
                        ActionLogits(
                            ActionSubspace(
                                subspace.name,
                                subspace.action_type,
                                subspace.num_actions,
                                opened[state_i][subspace.name],
                            ),
                            logits,
                        )
                    )
                    weight = torch.from_numpy(
                        row_weights[state_i][subspace.name]
                    ).to(self.device)
                    if self.hbs.hierarchical and self.hbs.bucket_term == "partition":
                        # Both sides carry it, so P_F stays normalized over the opened
                        # set and the term only moves mass between subspaces.
                        weight = weight + bucket_logp[state_i][subspace.name]
                    weights.append(weight)
                action_logits[state_i] = entries
                partition_weights[state_i] = weights

        # Candidate-set diversity is deliberately not recorded: for retrieval a low
        # value is the mechanism working, so the number would invert across arms.
        # _probe measures retrieval quality instead.
        return WeightedActionCategorical(
            action_logits,
            self.subsampling,
            partition_weights=partition_weights,
            explore_temp=self.hbs.explore_temp,
        )

    def get_action_logits(
        self, state_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score the observed edges, which need not lie in any opened bucket.

        Upstream's body with the synthon encoder replaced by the fixed features, so a
        replayed action keeps a finite forward logit.
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
        # Which bucket holds each observed row: under P_F = P(bucket)P(row|bucket) the
        # numerator is that bucket's log-probability, which for a replayed action need
        # not be the one this query would open.
        owning_bucket = {
            row: self.reservoir.index[actions[row].library_name].row_bucket[
                actions[row].synthon_index
            ]
            for rows in by_library.values()
            for row in rows
        } if self.hbs.hierarchical and self.hbs.bucket_term != "off" else {}
        bucket_offset = state_emb.new_zeros(len(actions))
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
                if owning_bucket:
                    # Kept out of `values`: a log-probability must not be rescaled by
                    # logit_scale or the row temperature, so it is added after those.
                    for library_name in {actions[row].library_name for row in rows}:
                        local = [
                            i for i, row in enumerate(rows)
                            if actions[row].library_name == library_name
                        ]
                        log_p, _ = self._bucket_log_probs(library_name, probe[local])
                        picked = torch.tensor(
                            [owning_bucket[rows[i]] for i in local], device=self.device
                        )
                        bucket_offset = bucket_offset.index_add(
                            0,
                            indices[local],
                            log_p.gather(1, picked[:, None]).squeeze(1),
                        )
                with torch.no_grad():
                    # features are already unit-norm, so this is the cosine.
                    cosines.append(
                        (F.normalize(probe, dim=-1) * features[indices]).sum(1)
                    )
            logits = logits.index_copy(0, indices, values)
        if cosines:
            self.alignment = float(torch.cat(cosines).mean())
        return logits * logit_scale / self._row_temperature() + bucket_offset

    # ======== Fixed features

    def _gather_libraries(
        self, samples: dict[str, NDArray[np.int64]]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fixed features, properties and sizes for the given rows of each library.

        ``_get_synthon_cache`` with the learned encoder replaced by the normalized
        fingerprints. Nothing here carries gradient; only the query projection learns.
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


