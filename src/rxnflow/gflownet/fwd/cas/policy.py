"""Clustered action sampling: P(action) = P(cell) * P(row | cell).

The cell is scored against the cell centroids by the policy itself, so unlike a hashed
bucket the choice is differentiable and the loss can move the probe to a better cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from rxnflow.config import Config
from rxnflow.core.types import Action, ActionKey, ActionSubspace, ActionType, State
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import PROPERTY_DIM
from rxnflow.envs.graph import GraphBatch
from rxnflow.gflownet.fwd.base import (
    MetricsPolicyMixin,
    StepClock,
    Temperature,
)
from rxnflow.gflownet.fwd.cas.index import ClusteredActionIndex
from rxnflow.gflownet.fwd.sampling import WeightedActionCategorical
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
class CASSettings:
    """Settings for clustered action sampling."""

    # Target cell occupancy; nlist per library is round(N / rows_per_cell).
    rows_per_cell: int = 235
    # Cells opened per state; 1 keeps the latent decomposition exact.
    nprobe: int = 1
    # Temperature of the cell softmax. Large spreads mass over cells (the probe is
    # uncommitted); small approaches the hard argmax the index is read with.
    temp_bucket: Temperature = field(default_factory=Temperature)
    # Temperature of the within-cell row softmax, on both sides of log_prob.
    temp_nn: Temperature = field(default_factory=Temperature)
    # Where log P(cell) enters the loss; see fwd/hbs/policy.py for the three readings.
    bucket_term: str = "numerator"
    normalize_probe: bool = False
    # Upstream's subsampling rule, used only to clamp cell size on small libraries
    # so CAS never scores more candidates than the baseline it replaces.
    sampling_ratio: float = 0.1
    min_sampling: int = 50
    seed: int = 0
    kmeans_iters: int = 25
    train_sample: int | None = 200_000

    def validate(self) -> None:
        if self.rows_per_cell < 1:
            raise ValueError("rows_per_cell must be at least 1")
        if self.nprobe < 1:
            raise ValueError("nprobe must be at least 1")
        self.temp_bucket.validate("cas.temp_bucket")
        self.temp_nn.validate("cas.temp_nn")
        if self.bucket_term not in ("numerator", "partition", "off"):
            raise ValueError("cas.bucket_term must be numerator, partition or off")
        if self.nprobe > 1:
            logger.warning(
                "nprobe=%d opens several cells, so the first level is a renormalized "
                "truncation rather than an exact partition; log P(cell) is approximate",
                self.nprobe,
            )


def build_indexes(
    env: SynthesisEnv, settings: CASSettings, device: torch.device
) -> dict[str, ClusteredActionIndex]:
    """One clustered index per library, each sized to ``rows_per_cell``.

    Libraries run from 12 rows to 92,813, so occupancy is fixed rather than ``nlist``.
    """
    indexes: dict[str, ClusteredActionIndex] = {}
    skipped = 0
    for name, library in env.synthons.items():
        features = np.asarray(library.fingerprints, dtype=np.float32)
        size = len(features)
        nlist = int(np.clip(round(size / settings.rows_per_cell), 1, size))
        # Clamp nlist so a cell never exceeds upstream's K: below ~350 rows nlist
        # collapses to 1 and CAS would score more rows than uniform subsampling.
        uniform_k = min(size, max(settings.min_sampling,
                                  int(size * settings.sampling_ratio)))
        if size / nlist > uniform_k:
            nlist = int(np.clip(round(size / uniform_k), 1, size))
            skipped += 1
        index = ClusteredActionIndex(
            {name: features},
            nlist,
            seed=settings.seed,
            iters=settings.kmeans_iters,
            # Dead at this size -- no library reaches 200,000 rows -- but kept so a
            # global index built from this same code path still trains in bounded time.
            train_sample=settings.train_sample,
            device=str(device),
        )
        # CASPolicy hands row ids straight to the environment as local synthon
        # indices, which is only valid while an index holds one library.
        assert index.single_library, f"index for {name} spans several libraries"
        indexes[name] = index
    logger.info("CAS-Validity: %d of %d libraries clamped to upstream's K", skipped, len(indexes))
    sizes = np.array([ix.nlist for ix in indexes.values()])
    occupancy = np.concatenate([ix.sizes for ix in indexes.values()])
    logger.info(
        "CAS-Validity: %d libraries, nlist %d-%d (total %d cells), occupancy mean %.1f "
        "median %d max %d",
        len(indexes), int(sizes.min()), int(sizes.max()), int(sizes.sum()),
        float(occupancy.mean()), int(np.median(occupancy)), int(occupancy.max()),
    )
    return indexes


class CASPolicy(MetricsPolicyMixin, RxnFlowPolicy):
    """Candidate selection by clustered lookup, with the cell choice in the policy."""

    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        rng: np.random.Generator,
        settings: CASSettings,
        indexes: dict[str, ClusteredActionIndex],
        projection: torch.nn.Module,
        clock: StepClock | None = None,
    ):
        super().__init__(env, model, config, device, rng)
        self.clock = clock if clock is not None else StepClock()
        self.cas = settings
        self.indexes = indexes
        self.projection = projection
        self.dim = next(iter(indexes.values())).dim
        # Left at log(N/N) = 0 and kept only for inherited code paths: CAS corrects
        # with -log P(cell), a live tensor in WeightedActionCategorical.
        self.subsampling = {
            name: SubsamplingPolicy(len(library), 1.0, len(library), rng)
            for name, library in env.synthons.items()
        }
        self._calls = 0
        self.alignment: float | None = None
        # Entropy of the cell distribution in nats, against log(nlist) for "no
        # preference". Diagnostic: the cell choice does not enter the loss.
        self.cell_entropy: float | None = None
        # Mean -log P(opened cell): 0 is certainty, log(nlist) is indifference.
        self.cell_logp: float | None = None
        # The opened cell's log P per (state, subspace), kept between the candidate
        # pass and the logits so the bucket term can enter the loss.
        self._cell_logp: list[dict[ActionKey, torch.Tensor]] = []

    def _probe_vector(self, projected: torch.Tensor) -> torch.Tensor:
        if self.cas.normalize_probe:
            return F.normalize(projected, dim=-1)
        return projected

    def _open_cells(
        self, library_name: str, probes: torch.Tensor
    ) -> tuple[list[NDArray[np.int64]], torch.Tensor, torch.Tensor]:
        """Rows offered to each probe, its opened cell's ``log P``, and all cells'.

        ``probes`` stays attached to the graph, so the log-probabilities carry gradient.
        """
        index = self.indexes[library_name]
        scores = index.cell_scores(probes) / self.cas.temp_bucket.at(self.clock.value)
        log_p = torch.log_softmax(scores, dim=1)
        # argmax, not a draw: the cell is the action space, so the policy commits to
        # its best one and normalizes exactly over it.
        chosen = scores.detach().topk(min(self.cas.nprobe, index.nlist), dim=1).indices
        rows = [index.rows_in_cells(row) for row in chosen.cpu().numpy()]
        if self.cas.nprobe == 1:
            picked = log_p.gather(1, chosen)[:, 0]
        else:
            picked = torch.logsumexp(log_p.gather(1, chosen), dim=1)
        return rows, picked, log_p

    def _observed_cell_log_p(
        self, library_name: str, probe: torch.Tensor, synthon_rows: NDArray[np.int64]
    ) -> torch.Tensor:
        """log P of the cell holding each observed row, under this library's index."""
        index = self.indexes[library_name]
        return self._cell_log_p_at(index, probe, synthon_rows)

    def _cell_log_p_at(
        self, index: ClusteredActionIndex, probe: torch.Tensor, rows: NDArray[np.int64]
    ) -> torch.Tensor:
        cells = index.cell_of[
            torch.as_tensor(rows, device=index.cell_of.device)
        ].to(self.device)
        log_p = torch.log_softmax(
            index.cell_scores(probe) / self.cas.temp_bucket.at(self.clock.value), dim=1
        )
        return log_p.gather(1, cells[:, None]).squeeze(1)

    def _row_temperature(self) -> float:
        """Temperature of the within-cell row softmax at this step."""
        return self.cas.temp_nn.at(self.clock.value)

    def _assign_candidates(
        self,
        states: list[State],
        plans: list[tuple],
    ) -> tuple[list[dict[ActionKey, NDArray[np.int64]]], dict[str, list[NDArray[np.int64]]]]:
        """Candidate rows per (state, subspace), and the rows each library owes.

        One cell per (state, subspace) from that subspace's own library index, so every
        offered row is legal by construction.
        """
        opened: list[dict[ActionKey, NDArray[np.int64]]] = [{} for _ in states]
        union: dict[str, list[NDArray[np.int64]]] = {}
        self._cell_logp = [{} for _ in states]
        observed: list[float] = []
        entropies: list[float] = []
        for rows, _, _, spaces_by_action, _, probes in plans:
            for key, probe in probes.items():
                for subspace in spaces_by_action[key]:
                    library_name = subspace.name[1]
                    assert library_name is not None
                    found, log_p, full_log_p = self._open_cells(library_name, probe)
                    observed.extend((-log_p).detach().cpu().tolist())
                    with torch.no_grad():
                        entropies.extend(
                            (-(full_log_p.exp() * full_log_p).sum(1)).cpu().tolist()
                        )
                    for local, state_i in enumerate(rows):
                        opened[state_i][subspace.name] = found[local]
                        self._cell_logp[state_i][subspace.name] = log_p[local]
                        union.setdefault(library_name, []).append(found[local])
        if observed:
            self.cell_logp = float(np.mean(observed))
        if entropies:
            self.cell_entropy = float(np.mean(entropies))
        return opened, union

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

        # 1. Form every probe first: the cell a state opens is a function of it.
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
            probes = {
                key: self._probe_vector(self.projection(query))
                for key, query in queries.items()
                if not key[0].is_unirxn
            }
            plans.append((rows, indices, scale, spaces_by_action, queries, probes))

        # 2. Open cells and turn them into candidate rows, factored out so
        # fwd/cas/fullspace.py can replace just this step.
        opened, union = self._assign_candidates(states, plans)

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

        # 4. Score as upstream does, then take each state's own columns.
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
                    if subspace.name[1] is not None and len(logits) == 0:
                        # A subspace with no rows offers no action, and an empty
                        # entry would make upstream's sample() take -log(0). Routine
                        # under fwd/cas/fullspace.py, impossible per-library.
                        continue
                    if subspace.name[1] is None:
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
                    # Zero, deliberately: the cell *is* this state's action space, so
                    # log_partition is exactly logsumexp over it. Carrying -log P(cell)
                    # would need the cell to be drawn with that probability, and
                    # _open_cells takes the argmax.
                    if self.cas.bucket_term == "partition":
                        # Both sides carry it, so P_F stays normalized over the opened
                        # cell and the term only moves mass between subspaces.
                        weights.append(
                            self._cell_logp[state_i][subspace.name].expand(len(logits))
                        )
                    else:
                        weights.append(torch.zeros(len(logits), device=self.device))
                action_logits[state_i] = entries
                partition_weights[state_i] = weights

        return WeightedActionCategorical(
            action_logits,
            self.subsampling,
            partition_weights=partition_weights,
            explore_temp=0.0,
        )

    def get_action_logits(
        self, state_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score observed edges, which need not lie in the cell the probe now opens.

        Same rule as ``get_action_categorical``, so a replayed trajectory is scored by
        the function that sampled it. The cell enters only through the partition.
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
            indices = np.array([actions[i].synthon_index for i in rows], dtype=np.int64)
            feature_rows.extend(rows)
            gathered.append(
                self._gather_libraries({library_name: indices})[library_name][0]
            )
        features = state_emb.new_zeros((len(actions), self.dim))
        if gathered:
            features = features.index_copy(
                0, torch.tensor(feature_rows, device=self.device), torch.cat(gathered)
            )

        logits = state_emb.new_zeros(len(actions))
        bucket_offset = state_emb.new_zeros(len(actions))
        cosines: list[torch.Tensor] = []
        for (action_type, action_name), rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            emb = state_emb[indices]
            if action_type.is_unirxn:
                values = self._unirxn_logits(action_type, action_name, emb).squeeze(1)
            else:
                if action_type.is_first:
                    query = self.model.forward_first_synthon(emb)
                else:
                    query = self.model.forward_birxn(emb, action_name, action_type)
                probe = self._probe_vector(self.projection(query))
                values = (probe * features[indices]).sum(1)
                if self.cas.bucket_term != "off":
                    # log P of the cell holding the observed row, which for a replayed
                    # action need not be the cell this probe would open. Added after
                    # the scaling below: a log-probability is not a logit.
                    for library_name in {actions[row].library_name for row in rows}:
                        local = [
                            i for i, row in enumerate(rows)
                            if actions[row].library_name == library_name
                        ]
                        bucket_offset = bucket_offset.index_add(
                            0,
                            indices[local],
                            self._observed_cell_log_p(
                                library_name,
                                probe[local],
                                np.array(
                                    [actions[rows[i]].synthon_index for i in local],
                                    dtype=np.int64,
                                ),
                            ),
                        )
                with torch.no_grad():
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

        ``_get_synthon_cache`` with the learned encoder replaced by the precomputed
        fingerprints, so the clusters were fitted to the vectors being scored.
        """
        if not samples:
            return {}
        widths = [len(indices) for indices in samples.values()]
        count = sum(widths)
        fp_arr = np.empty((count, self.dim), dtype=np.uint8)
        prop_arr = np.empty((count, PROPERTY_DIM), dtype=np.float32)
        size_arr = np.empty(count, dtype=np.uint8)
        offset = 0
        for library_name, indices in samples.items():
            library = self.env.synthons[library_name]
            rows = slice(offset, offset + len(indices))
            np.take(library.fingerprints, indices, axis=0, out=fp_arr[rows], mode="raise")
            np.take(library.properties, indices, axis=0, out=prop_arr[rows], mode="raise")
            np.take(library.heavy_atoms, indices, out=size_arr[rows], mode="raise")
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
