# Source: src/rxnflow/gflownet/policy.py
"""Masked action distributions, shared library subsampling, and trajectory generation."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from IFT6760.core.errors import InvalidTransition
from IFT6760.core.types import (
    Action,
    ActionKey,
    ActionSpace,
    ActionSubspace,
    ActionType,
    BackwardTrajectory,
    Condition,
    InvalidReason,
    State,
    Trajectory,
    Transition,
)
from IFT6760.envs.env import SynthesisEnv
from IFT6760.envs.features import FINGERPRINT_DIM, PROPERTY_DIM
from IFT6760.envs.graph import GraphBatch, GraphData, molecule_to_graph_data
from IFT6760.models import RxnFlowModel
from IFT6760.rewards.property_penalty import get_property_penalty, get_size_mask
from IFT6760.training.config import Config


@dataclass
class ActionLogits:
    subspace: ActionSubspace
    logits: torch.Tensor  # [sampled action] for one state.


class SubsamplingPolicy:
    """Uniform subsampling of a synthon library."""

    def __init__(
        self,
        num_actions: int,
        sampling_ratio: float,
        min_sampling: int,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.num_actions = num_actions
        self.num_sampling = min(
            num_actions, max(min_sampling, int(num_actions * sampling_ratio))
        )
        self.log_importance = math.log(num_actions / self.num_sampling)
        self.full_indices = (
            np.arange(num_actions, dtype=np.int64)
            if self.num_sampling == num_actions
            else None
        )

    def sample(self) -> NDArray[np.int64]:
        if self.full_indices is not None:
            # Full ranges consume no RNG or new allocation.
            return self.full_indices
        indices = self.rng.choice(self.num_actions, self.num_sampling, replace=False)
        # Keep catalog row order without changing which actions were sampled.
        indices.sort()
        return indices


@dataclass
class ActionCategorical:
    """Action distribution over sampled subspaces, grouped by state."""

    action_logits: list[list[ActionLogits]]
    subsampling: dict[str, SubsamplingPolicy]

    @property
    def device(self) -> torch.device:
        return self.action_logits[0][0].logits.device

    def log_partition(self) -> torch.Tensor:
        """Return log(sum_a exp(logit_a) / p_a) for each state's sampled actions.

        The inclusion probability p_a is K/N for K samples from an N-synthon
        library, and 1 for unimolecular actions.
        """
        device = self.device
        partitions = []
        for state_logits in self.action_logits:
            assert state_logits
            sizes, weights = [], []
            for entry in state_logits:
                sizes.append(len(entry.logits))
                library_name = entry.subspace.name[1]
                weights.append(
                    0.0
                    if library_name is None
                    else self.subsampling[library_name].log_importance
                )
            logits = torch.cat([entry.logits for entry in state_logits])
            counts = torch.tensor(sizes, dtype=torch.long, device=device)
            # Known output sizes avoid device synchronization during expansion.
            offsets = logits.new_tensor(weights).repeat_interleave(
                counts, output_size=logits.numel()
            )
            weighted = logits + offsets
            # Stable log-sum-exp with a finite floor when every logit is -inf.
            maximum = weighted.max().detach()
            maximum = torch.where(torch.isfinite(maximum), maximum, 0)
            total = (weighted - maximum).exp().sum()
            partitions.append(maximum + total.clamp_min(1e-38).log())
        return torch.stack(partitions)

    def sample(
        self, softmax_temperature: float, random_action_prob: float, importance: float
    ) -> list[Action]:
        """Draw one action per state using exploration scores.

        Add importance * log(N/K) to synthon logits and divide by softmax_temperature.
        random_action_prob switches selected states to subspace-balanced random scores.
        """
        device = self.device
        random_states = (
            torch.rand(len(self.action_logits), device=device) < random_action_prob
        )
        columns = []
        for state_i, state_logits in enumerate(self.action_logits):
            assert state_logits
            sizes, weights = [], []
            for entry in state_logits:
                sizes.append(len(entry.logits))
                library_name = entry.subspace.name[1]
                weight = (
                    0.0
                    if library_name is None
                    else self.subsampling[library_name].log_importance
                )
                weights.append(importance * weight)

            logits = torch.cat([entry.logits for entry in state_logits])
            counts = torch.tensor(sizes, dtype=torch.long, device=device)
            offsets = logits.new_tensor(weights).repeat_interleave(
                counts, output_size=logits.numel()
            )
            values = logits + offsets
            if random_action_prob > 0:
                # At temperature 1, each subspace has equal mass before masking,
                # with uniform scores over its sampled actions.
                random_logits = logits.new_tensor(
                    [-math.log(size) for size in sizes]
                ).repeat_interleave(counts, output_size=logits.numel())
                values = torch.where(random_states[state_i], random_logits, values)
            valid = torch.isfinite(logits)
            values = values.masked_fill(~valid, -torch.inf)
            # If all candidates are masked, relax the mask and sample uniformly.
            values = torch.where(valid.any(), values, torch.zeros_like(values))
            values = values / softmax_temperature
            # Gumbel-max samples from softmax without constructing probabilities.
            uniform = torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)
            gumbels = -(-uniform.log()).log()
            columns.append((values + gumbels).argmax())

        # Transfer indices to CPU once for the whole batch, then decode actions.
        actions: list[Action] = []
        for state_logits, column in zip(
            self.action_logits, torch.stack(columns).tolist(), strict=True
        ):
            for entry in state_logits:
                if column < len(entry.logits):
                    actions.append(entry.subspace.action_at(column))
                    break
                column -= len(entry.logits)
        return actions


class RxnFlowPolicy:
    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        rng: np.random.Generator,
    ):
        self.env, self.model, self.config = env, model, config
        self.device, self.rng = device, rng
        # Each library supplies one shared sample per forward call.
        self.subsampling = {
            name: SubsamplingPolicy(
                len(library),
                config.subsampling.sampling_ratio,
                config.subsampling.min_sampling,
                rng,
            )
            for name, library in env.synthons.items()
        }

    def _get_state_inputs(self, states: list[State]) -> GraphBatch:
        """Reuse molecular preprocessing for repeated states in a batch."""
        graphs: dict[str, GraphData] = {}
        for state in states:
            if state.smiles not in graphs:
                cached = state._cache.get("graph")
                graphs[state.smiles] = (
                    molecule_to_graph_data(state.molecule) if cached is None else cached
                )
            state._cache["graph"] = graphs[state.smiles]
        return GraphBatch.from_list([graphs[state.smiles] for state in states])

    def _get_synthon_features(
        self, library_name: str, indices: NDArray[np.int64]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        library = self.env.synthons[library_name]
        fp, prop = library.fingerprints, library.properties
        types = self.env.library_site_indices[library_name]
        # Non-blocking device/dtype conversion casts uint8 fingerprints on GPU.
        fp = torch.from_numpy(fp[indices]).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prop = torch.from_numpy(prop[indices]).to(self.device, non_blocking=True)
        types = torch.tensor(types, dtype=torch.long, device=self.device)
        types = types.expand(len(indices), 2)
        return fp, prop, types

    def _encode_synthons(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Encode synthons with unit-norm embeddings for action scoring."""
        return F.normalize(self.model.encode_synthon(*features), dim=-1)

    def forward(
        self,
        states: list[State],
        beta: torch.Tensor,
        preference: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        apply_property_penalty: bool = False,
    ) -> ActionCategorical:
        """Reference implementation of forward scoring, one state at a time.

        get_action_categorical batches the same computation for training and sampling.
        """
        assert states
        # Late conditioning keeps graph encodings independent of beta and preferences.
        batch = self._get_state_inputs(states).to(self.device)
        state_emb = self.model.encode_state(batch)
        cond_emb = self.model.encode_cond(beta, preference, cond)
        action_emb = torch.cat([state_emb, cond_emb], dim=-1)
        logit_scale = self.model.logit_scale(cond_emb)

        action_spaces = self._prepare_action_space(states)
        synthon_cache = self._get_synthon_cache(action_spaces)

        action_logits: list[list[ActionLogits]] = []
        for state_i, action_space in enumerate(action_spaces):
            emb = action_emb[state_i]
            scale = logit_scale[state_i]
            state_prop = batch.graph_features[state_i]
            state_size = batch.num_heavy_atoms[state_i]

            state_logits: list[ActionLogits] = []
            for subspace in action_space:
                action_name, library_name = subspace.name
                action_type = subspace.action_type
                if action_type.is_unirxn:
                    logits = self._unirxn_logits(action_type, action_name, emb)
                    action_mask = None
                else:
                    synthon_emb, synthon_prop, synthon_size = synthon_cache[library_name]
                    if action_type.is_first:
                        logits = self._first_synthon_logits(emb, synthon_emb)
                    else:
                        logits = self._birxn_logits(
                            action_type, action_name, emb, synthon_emb
                        )

                    if apply_property_penalty:
                        size_mask = get_size_mask(
                            state_size, synthon_size, self.env.max_atoms
                        )
                        property_mask = get_property_penalty(
                            state_prop, synthon_prop, self.env.property_limits
                        )
                        action_mask = size_mask & property_mask
                    else:
                        action_mask = None

                logits = logits * scale

                if action_mask is not None:
                    logits.masked_fill_(~action_mask, -torch.inf)

                state_logits.append(ActionLogits(subspace, logits))

            action_logits.append(state_logits)
        return ActionCategorical(action_logits, self.subsampling)

    def get_action_categorical(
        self,
        states: list[State],
        batch: GraphBatch,
        state_emb: torch.Tensor,
        logit_scale: torch.Tensor,
        *,
        apply_property_penalty: bool = False,
    ) -> ActionCategorical:
        action_spaces = self._prepare_action_space(states)
        synthon_cache = self._get_synthon_cache(action_spaces)

        state_groups: dict[int | None, list[int]] = {}
        for state_i, state in enumerate(states):
            assert action_spaces[state_i]
            state_groups.setdefault(state.attachment_type, []).append(state_i)

        action_logits: list[list[ActionLogits]] = [[] for _ in states]
        for rows in state_groups.values():
            indices = torch.tensor(rows, device=self.device)
            emb = state_emb[indices]
            scale = logit_scale[indices, None]
            subspaces = {
                subspace.name: subspace
                for state_i in rows
                for subspace in action_spaces[state_i]
            }
            spaces_by_action: dict[tuple[ActionType, str], ActionSpace] = {}
            for subspace in subspaces.values():
                key = (subspace.action_type, subspace.name[0])
                spaces_by_action.setdefault(key, []).append(subspace)

            # Batch the shared MLP across reactions, then score each library separately.
            reaction_queries: dict[tuple[ActionType, str], torch.Tensor] = {}
            for action_type in (
                ActionType.UNIRXN_TRANSFORM,
                ActionType.UNIRXN_TERMINAL,
                ActionType.BIRXN_BRICK,
                ActionType.BIRXN_LINKER,
            ):
                names = [name for kind, name in spaces_by_action if kind == action_type]
                if names:
                    queries = self.model.forward_reactions(emb, names, action_type)
                    for name, query in zip(names, queries.unbind(0), strict=True):
                        reaction_queries[action_type, name] = query

            # Apply size/property masks during sampling; TB uses unmasked logits and
            # terminal rewards.
            library_names = list(
                dict.fromkeys(
                    space.name[1]
                    for space in subspaces.values()
                    if space.name[1] is not None
                )
            )
            library_masks: dict[str, torch.Tensor] = {}
            if apply_property_penalty and library_names:
                prop = torch.cat([synthon_cache[name][1] for name in library_names])
                size = torch.cat([synthon_cache[name][2] for name in library_names])
                size_mask = get_size_mask(
                    batch.num_heavy_atoms[indices, None], size, self.env.max_atoms
                )
                property_penalty = get_property_penalty(
                    batch.graph_features[indices], prop, self.env.property_limits
                )
                widths = [len(synthon_cache[name][0]) for name in library_names]
                masks = (size_mask & property_penalty).split(widths, dim=1)
                library_masks = dict(zip(library_names, masks, strict=True))

            group_logits: dict[ActionKey, tuple[torch.Tensor, ...]] = {}
            for (action_type, action_name), spaces in spaces_by_action.items():
                if action_type.is_unirxn:
                    logits = reaction_queries[action_type, action_name] * scale
                    group_logits[spaces[0].name] = logits.unbind(0)
                    continue

                libraries = [space.name[1] for space in spaces]
                synthons = torch.cat([synthon_cache[name][0] for name in libraries])
                if action_type.is_first:
                    logits = self._first_synthon_logits(emb, synthons) * scale
                else:
                    query = reaction_queries[action_type, action_name]
                    logits = (query @ synthons.T) * scale
                if apply_property_penalty:
                    mask = torch.cat([library_masks[name] for name in libraries], dim=1)
                    logits = logits.masked_fill(~mask, -torch.inf)
                widths = [len(synthon_cache[name][0]) for name in libraries]
                for subspace, values in zip(
                    spaces, logits.split(widths, dim=1), strict=True
                ):
                    group_logits[subspace.name] = values.unbind(0)

            # Keep each state's budget eligibility and original subspace order.
            for local_row, state_i in enumerate(rows):
                action_logits[state_i] = [
                    ActionLogits(subspace, group_logits[subspace.name][local_row])
                    for subspace in action_spaces[state_i]
                ]

        return ActionCategorical(action_logits, self.subsampling)

    def _prepare_action_space(self, states: list[State]) -> list[ActionSpace]:
        """Resolve eligible subspaces with one shared sample per library."""
        samples: dict[str, NDArray[np.int64]] = {}
        sampled_spaces: list[ActionSpace] = []
        for state in states:
            sampled_space: ActionSpace = []
            for subspace in self.env.get_action_space(state):
                library_name = subspace.name[1]
                if library_name is None:
                    sampled_space.append(subspace)
                    continue
                if library_name not in samples:
                    samples[library_name] = self.subsampling[library_name].sample()
                indices = samples[library_name]
                sampled_space.append(
                    ActionSubspace(
                        subspace.name, subspace.action_type, subspace.num_actions, indices
                    )
                )
            assert sampled_space, f"state {state.smiles} has no valid actions"
            sampled_spaces.append(sampled_space)
        return sampled_spaces

    def _get_synthon_cache(
        self, action_spaces: list[ActionSpace]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode all sampled libraries together, then share their embedding views."""
        samples: dict[str, NDArray[np.int64]] = {}
        for space in action_spaces:
            for subspace in space:
                library_name = subspace.name[1]
                if library_name is not None:
                    samples.setdefault(library_name, subspace.sample_indices)
        if not samples:
            return {}

        widths = [len(indices) for indices in samples.values()]
        count = sum(widths)
        fp_arr = np.empty((count, FINGERPRINT_DIM), dtype=np.uint8)
        prop_arr = np.empty((count, PROPERTY_DIM), dtype=np.float32)
        type_arr = np.empty((count, 2), dtype=np.int64)
        size_arr = np.empty(count, dtype=np.uint8)
        offset = 0
        for library_name, indices in samples.items():
            library = self.env.synthons[library_name]
            rows = slice(offset, offset + len(indices))
            # Indices are valid; clip mode avoids the temporary buffer used by raise mode.
            np.take(library.fingerprints, indices, axis=0, out=fp_arr[rows], mode="clip")
            np.take(library.properties, indices, axis=0, out=prop_arr[rows], mode="clip")
            np.take(library.heavy_atoms, indices, out=size_arr[rows], mode="clip")
            type_arr[rows] = self.env.library_site_indices[library_name]
            offset += len(indices)

        fp = torch.from_numpy(fp_arr).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prop = torch.from_numpy(prop_arr).to(self.device, non_blocking=True)
        types = torch.from_numpy(type_arr).to(
            self.device, dtype=torch.long, non_blocking=True
        )
        size = torch.from_numpy(size_arr).to(self.device, non_blocking=True)
        embeddings = self._encode_synthons((fp, prop, types))
        return {
            library_name: (emb, prop, size)
            for library_name, emb, prop, size in zip(
                samples,
                embeddings.split(widths),
                prop.split(widths),
                size.split(widths),
                strict=True,
            )
        }

    def _first_synthon_logits(
        self,
        state_emb: torch.Tensor,
        synthons: torch.Tensor,
    ) -> torch.Tensor:
        """Score initial synthons with a state-only query."""
        query = self.model.forward_first_synthon(state_emb)
        return query @ synthons.T

    def _unirxn_logits(
        self,
        action_type: ActionType,
        action_name: str,
        state_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score the single action of a unimolecular reaction."""
        return self.model.forward_unirxn(state_emb, action_name, action_type)

    def _birxn_logits(
        self,
        action_type: ActionType,
        action_name: str,
        state_emb: torch.Tensor,
        synthons: torch.Tensor,
    ) -> torch.Tensor:
        """Score incoming synthons with a state/reaction query."""
        query = self.model.forward_birxn(state_emb, action_name, action_type)
        return query @ synthons.T

    def get_action_logits(
        self, state_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score numerator edges independently of the denominator subsample.

        Group heads by (action type, name) and synthon features by library.
        Observed actions are scored even if absent from the denominator sample.
        """
        by_reaction, by_library = {}, {}
        for row, action in enumerate(actions):
            name = "" if action.action_type.is_first else action.reaction
            by_reaction.setdefault((action.action_type, name), []).append(row)
            if action.library_name is not None:
                by_library.setdefault(action.library_name, []).append(row)
        synthon_rows, features = [], []
        for name, rows in by_library.items():
            indices = np.array([actions[i].synthon_index for i in rows], dtype=np.int64)
            synthon_rows.extend(rows)
            features.append(self._get_synthon_features(name, indices))
        synthons = state_emb.new_zeros((len(actions), self.config.model.synthon_dim))
        if features:
            fp, prop, types = zip(*features, strict=True)
            values = self._encode_synthons(
                (torch.cat(fp), torch.cat(prop), torch.cat(types))
            )
            synthons = synthons.index_copy(
                0, torch.tensor(synthon_rows, device=self.device), values
            )
        logits = state_emb.new_zeros(len(actions))
        for (action_type, action_name), rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            if action_type.is_unirxn:
                values = self._unirxn_logits(
                    action_type, action_name, state_emb[indices]
                ).squeeze(1)
            elif action_type.is_first:
                query = self.model.forward_first_synthon(state_emb[indices])
                values = (query * synthons[indices]).sum(1)
            else:
                query = self.model.forward_birxn(
                    state_emb[indices], action_name, action_type
                )
                values = (query * synthons[indices]).sum(1)
            logits = logits.index_copy(0, indices, values)
        return logits * logit_scale

    @torch.no_grad()
    def sample_actions(
        self,
        states: list[State],
        beta: torch.Tensor,
        preference: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        softmax_temperature: float = 1.0,
        random_action_prob: float = 0.0,
    ) -> list[Action]:
        assert states
        batch = self._get_state_inputs(states).to(self.device)
        state_emb = self.model.encode_state(batch)
        cond_emb = self.model.encode_cond(beta, preference, cond)
        state_emb = torch.cat([state_emb, cond_emb], dim=-1)
        logit_scale = self.model.logit_scale(cond_emb)
        fwd_cat = self.get_action_categorical(
            states,
            batch,
            state_emb,
            logit_scale,
            apply_property_penalty=True,
        )
        return fwd_cat.sample(
            softmax_temperature,
            random_action_prob,
            self.config.subsampling.importance_temp,
        )

    def log_prob(
        self,
        states: list[State],
        actions: list[Action],
        beta: torch.Tensor,
        preference: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TB uses unmasked logits for both the numerator and partition estimate.
        assert states
        batch = self._get_state_inputs(states).to(self.device)
        state_emb = self.model.encode_state(batch)
        cond_emb = self.model.encode_cond(beta, preference, cond)
        state_emb = torch.cat([state_emb, cond_emb], dim=-1)
        logit_scale = self.model.logit_scale(cond_emb)

        action_logits = self.get_action_logits(state_emb, actions, logit_scale)

        log_partitions = self.get_action_categorical(
            states, batch, state_emb, logit_scale
        ).log_partition()

        return (action_logits - log_partitions).clamp(max=0.0)

    def calc_bck_logprob(
        self,
        action: Action,
        trajectories: list[BackwardTrajectory],
        parent_smiles: str,
    ) -> float | None:
        """Normalize synthon-count-weighted route mass for the observed reverse edge."""
        # This is an explicit route preference, independent of catalog size.
        # TODO: Check backward consistency under trajectory limits and search
        # approximation.
        log_penalty = math.log(self.config.training.backward_synthon_penalty)
        numerator = denominator = -math.inf
        for trajectory in trajectories:
            num_synthons = sum(not edge.action_type.is_unirxn for edge, _ in trajectory)
            # Subtract the shared FirstSynthon count; extra UniReactions do not affect
            # route weight.
            log_weight = -(num_synthons - 1) * log_penalty
            denominator = np.logaddexp(denominator, log_weight)
            if trajectory[0] == (action, parent_smiles):
                numerator = np.logaddexp(numerator, log_weight)
        if numerator == -math.inf:
            return None
        return float(numerator - denominator)

    def sample_from_model(
        self,
        count: int,
        *,
        beta: torch.Tensor,
        preference: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_list: list[Condition] | None = None,
        softmax_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
    ) -> list[Trajectory]:
        """Grow trajectories using fixed conditions and optional backward analysis.

        cond contains encoded features [count, condition_dim]; cond_list retains
        the corresponding raw values for reward evaluation and replay.
        """
        if count <= 0:
            raise ValueError("sample count must be positive")
        assert beta.shape == (count,)
        assert preference.shape == (count, self.model.num_objectives)
        assert (cond is None) == (cond_list is None)
        if cond is not None:
            assert len(cond_list) == count
            assert cond.shape == (count, self.model.condition_dim)
            cond = cond.to(self.device)
        beta_values, preference_values = beta.tolist(), preference.tolist()
        beta, preference = beta.to(self.device), preference.to(self.device)
        states = [self.env.initial_state() for _ in range(count)]
        steps: list[list[Transition]] = [[] for _ in range(count)]
        reasons: list[InvalidReason | None] = [None] * count
        # Seed each reverse search by prepending the generated edge to its parent routes.
        backward_trajectories: list[list[BackwardTrajectory]] = [
            [[]] for _ in range(count)
        ]

        def collect_backward() -> None:
            for index, routes in self.env.retro_analyzer.result():
                transition = steps[index][-1]
                # Heuristic backward mass includes shorter routes with different counters.
                # Forward sampling still enforces both synthesis budgets.
                value = self.calc_bck_logprob(
                    transition.action, routes, transition.state.smiles
                )
                if value is None:
                    raise RuntimeError("backward analysis lost the generated route")
                transition.log_p_B = value
                backward_trajectories[index] = routes

        for _ in range(self.env.max_reactions + 1):
            active = [
                i
                for i, state in enumerate(states)
                if reasons[i] is None and not state.terminated
            ]
            if not active:
                break
            selected = self.sample_actions(
                [states[index] for index in active],
                beta[active],
                preference[active],
                cond=cond[active] if cond is not None else None,
                softmax_temperature=softmax_temperature,
                random_action_prob=random_action_prob,
            )
            # Overlap reverse search with forward sampling; collect parent routes before
            # submitting the next step.
            if analyze_backward:
                collect_backward()
            for index, action in zip(active, selected, strict=True):
                state = states[index]
                # Keep failed actions so TB penalizes the sampled failure as well as its
                # prefix.
                try:
                    next_state = self.env.step(state, action)
                except InvalidTransition:
                    steps[index].append(Transition(state, action, ""))
                    reasons[index] = "invalid_transition"
                    continue
                # Unexpected RDKit/model errors still surface to the caller.
                steps[index].append(Transition(state, action, next_state.smiles))
                states[index] = next_state
                if analyze_backward:
                    self.env.retro_analyzer.submit(
                        index,
                        next_state.smiles,
                        self.env.max_reactions,
                        [
                            [(action, state.smiles), *route]
                            for route in backward_trajectories[index]
                            # Alternative routes may differ in length but must respect the
                            # reaction cap.
                            if len(route) <= self.env.max_reactions
                        ],
                    )

        if analyze_backward:
            collect_backward()

        trajectories = []
        for index, state in enumerate(states):
            if not state.terminated and reasons[index] is None:
                reasons[index] = "max_reactions"
            trajectories.append(
                Trajectory(
                    steps=steps[index],
                    beta=beta_values[index],
                    preference=preference_values[index],
                    cond=deepcopy(cond_list[index]) if cond_list is not None else None,
                    final_smiles=state.smiles if reasons[index] is None else "",
                    valid=reasons[index] is None,
                    invalid_reason=reasons[index],
                )
            )
        return trajectories
