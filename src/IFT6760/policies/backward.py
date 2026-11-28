"""Learned P_B over one-step reverse candidates.

Molecule-only candidates are cached by SMILES, then filtered by state counters.
Parents may be dead ends; the backward objectives train P_B to avoid them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from IFT6760.core.errors import MissingReverseEdge
from IFT6760.core.molecule import Molecule
from IFT6760.core.types import BackwardEdge, Trajectory
from IFT6760.envs.env import SynthesisEnv
from IFT6760.envs.graph import GraphBatch, GraphData, molecule_to_graph_data
from IFT6760.models import RxnFlowModel
from IFT6760.training.config import Config
from IFT6760.utils.segments import Segments


@dataclass
class Decisions:
    """States and their reverse candidates."""

    smiles: list[str]
    edges: list[tuple[BackwardEdge, ...]]

    def __len__(self) -> int:
        return len(self.smiles)


@dataclass
class ObservedBatch:
    """Forward decisions with the observed edge index for each."""

    decisions: Decisions
    observed: torch.Tensor  # [N] index of the generated edge within its candidates
    traj_index: torch.Tensor  # [N] trajectory each decision belongs to


@dataclass
class Rollouts:
    """Backward walks sampled from P_B."""

    log_prob: torch.Tensor  # [N] sum of log P_B along each walk (keeps its gradient)
    entropy: torch.Tensor  # [M] entropy of P_B at every visited state (keeps gradient)
    success: (
        torch.Tensor
    )  # [N] bool: reached the empty state, spending exactly its counters
    routes: list[list[BackwardEdge]] = field(default_factory=list)


@dataclass
class _Walk:
    smiles: str
    reactions: int  # counters still to undo; the walk is home when both reach zero
    synthons: int
    success: bool = False
    route: list[BackwardEdge] = field(default_factory=list)

    def take(self, edge: BackwardEdge, env: SynthesisEnv) -> bool:
        """Follow edge, update the counters, and return whether the walk continues.

        Success requires reaching the empty state with both counters at zero.
        """
        action, parent = edge
        counts = env.parent_counts(action, self.reactions, self.synthons)
        assert counts is not None, "allowed() admits only edges with a legal predecessor"
        self.route.append(edge)
        self.reactions, self.synthons = counts
        if parent == "":
            self.success = True
            return False
        self.smiles = parent
        return True


class BackwardPolicy:
    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
    ):
        if model.backward_head is None:
            raise ValueError(
                "the model has no backward head (backward.mode must be learned)"
            )
        self.env, self.model, self.config, self.device = env, model, config, device
        self.cfg = config.backward
        self._rxn_names = list(model.rxn_embedding.keys())
        self._rxn_index = {name: i for i, name in enumerate(self._rxn_names)}
        self._graphs: dict[str, GraphData] = {}

    @property
    def analyzer(self):
        return self.env.retro_analyzer

    def _graph(self, smiles: str) -> GraphData:
        graph = self._graphs.get(smiles)
        if graph is None:
            graph = (
                molecule_to_graph_data(Molecule(smiles=smiles))
                if smiles
                else GraphData.empty()
            )
            if len(self._graphs) >= 100_000:
                self._graphs.clear()
            self._graphs[smiles] = graph
        return graph

    def encode(self, smiles: list[str], chunk: int = 96) -> torch.Tensor:
        """Return [len(smiles), state_dim] embeddings, encoding repeats once.

        Batch unique states by size to reduce graph padding.
        """
        unique = sorted(dict.fromkeys(smiles), key=lambda s: self._graph(s).num_nodes)
        embeddings = torch.cat(
            [
                self.model.encode_state(
                    GraphBatch.from_list(
                        [self._graph(s) for s in unique[start : start + chunk]]
                    ).to(self.device)
                )
                for start in range(0, len(unique), chunk)
            ]
        )
        row_of = {s: row for row, s in enumerate(unique)}
        rows = torch.tensor([row_of[s] for s in smiles], device=self.device)
        return embeddings[rows]

    def _reaction_embeddings(self, edges: list[BackwardEdge]) -> torch.Tensor:
        """Return reaction embeddings [E, rxn_dim], with zeros for FirstSynthon."""
        table = torch.stack([self.model.rxn_embedding[name] for name in self._rxn_names])
        table = torch.cat([table, table.new_zeros(1, table.shape[1])])
        no_reaction = len(self._rxn_names)
        rows = [self._rxn_index.get(action.reaction, no_reaction) for action, _ in edges]
        return table[torch.tensor(rows, device=self.device)]

    def _synthon_embeddings(self, edges: list[BackwardEdge]) -> torch.Tensor:
        """Return normalized synthon embeddings [E, synthon_dim].

        Unimolecular edges get zeros. Features are loaded one library at a time.
        """
        out = torch.zeros(len(edges), self.config.model.synthon_dim, device=self.device)
        rows_by_library: dict[str, list[int]] = {}
        for row, (action, _) in enumerate(edges):
            if action.library_name is not None:
                rows_by_library.setdefault(action.library_name, []).append(row)
        if not rows_by_library:
            return out

        rows, fingerprints, properties, sites = [], [], [], []
        for name, library_rows in rows_by_library.items():
            library = self.env.synthons[name]
            index = np.array(
                [edges[r][0].synthon_index for r in library_rows], dtype=np.int64
            )
            fingerprints.append(
                torch.from_numpy(library.fingerprints[index]).to(
                    self.device, dtype=torch.float32
                )
            )
            properties.append(
                torch.from_numpy(library.properties[index]).to(
                    self.device, dtype=torch.float32
                )
            )
            site = torch.tensor(
                self.env.library_site_indices[name], dtype=torch.long, device=self.device
            )
            sites.append(site.expand(len(library_rows), 2))
            rows.extend(library_rows)

        encoded = self.model.encode_synthon(
            torch.cat(fingerprints), torch.cat(properties), torch.cat(sites)
        )
        values = F.normalize(encoded, dim=-1)
        return out.index_copy(0, torch.tensor(rows, device=self.device), values)

    def logits(self, decisions: Decisions) -> tuple[torch.Tensor, Segments]:
        """One logit per candidate of every decision: ([E], layout)."""
        seg = Segments([len(e) for e in decisions.edges], self.device)
        edges = [edge for options in decisions.edges for edge in options]

        # detach_shared: P_B's loss must not move the encoders it shares with P_F.
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.cfg.detach_shared
        ):
            states = self.encode(decisions.smiles)[seg.row]
            rxn = self._reaction_embeddings(edges)
            synthon = self._synthon_embeddings(edges)

        action_type = torch.tensor(
            [int(action.action_type) for action, _ in edges], device=self.device
        )
        return self.model.backward_head(states, rxn, action_type, synthon), seg

    def log_probs(self, decisions: Decisions) -> tuple[torch.Tensor, Segments]:
        """Dense [N, W] log P_B(e | s'), with -inf on padding."""
        scores, seg = self.logits(decisions)
        return seg.log_softmax(scores), seg

    def observed(self, trajectories: list[Trajectory]) -> ObservedBatch:
        """Locate each generated edge among candidates allowed by the state counters.

        Raise MissingReverseEdge if candidate generation fails to recover it.
        """
        steps = [
            (traj_index, step)
            for traj_index, trajectory in enumerate(trajectories)
            for step in trajectory.steps
            if step.product_smiles
        ]
        candidates = self.analyzer.one_step_many(
            [step.product_smiles for _, step in steps]
        )

        smiles, edges, observed, traj_index = [], [], [], []
        for (traj, step), options in zip(steps, candidates, strict=True):
            cost_r, cost_s = step.action.action_type.cost
            counts = (step.state.num_reactions + cost_r, step.state.num_synthons + cost_s)
            options = tuple(
                e for e in options if self.env.parent_counts(e[0], *counts) is not None
            )
            taken = (step.action, step.state.smiles)
            if taken not in options:
                raise MissingReverseEdge(step.product_smiles, *taken)
            smiles.append(step.product_smiles)
            edges.append(options)
            observed.append(options.index(taken))
            traj_index.append(traj)

        return ObservedBatch(
            Decisions(smiles, edges),
            torch.tensor(observed, dtype=torch.long, device=self.device),
            torch.tensor(traj_index, dtype=torch.long, device=self.device),
        )

    def allowed(
        self, edges: tuple[BackwardEdge, ...], reactions: int, synthons: int
    ) -> list[BackwardEdge]:
        """Keep edges that leave valid parent counters.

        This is a local check; a parent may still be a dead end.
        """
        return [
            edge
            for edge in edges
            if self.env.parent_counts(edge[0], reactions, synthons) is not None
        ]

    def rollout(self, start: list[tuple[str, int, int]]) -> Rollouts:
        """Walk backward from recorded (smiles, reactions, synthons) states.

        Advance active walks in batches, for at most max_reactions + 1 steps.
        Log-probabilities retain gradients; sampled choices do not.
        """
        env = self.env
        walks = [_Walk(smiles, n_r, n_s) for smiles, n_r, n_s in start]
        log_prob = torch.zeros(len(walks), device=self.device)
        entropies: list[torch.Tensor] = []
        active = list(range(len(walks)))

        for _ in range(env.max_reactions + 1):
            options = self.analyzer.one_step_many([walks[i].smiles for i in active])
            live = [
                (i, allowed)
                for i, edges in zip(active, options, strict=True)
                if (allowed := self.allowed(edges, walks[i].reactions, walks[i].synthons))
            ]
            if not live:
                break

            decisions = Decisions(
                [walks[i].smiles for i, _ in live], [edges for _, edges in live]
            )
            logp, seg = self.log_probs(decisions)
            choice = seg.sample(logp)
            rows = torch.tensor([i for i, _ in live], device=self.device)
            log_prob = log_prob.index_add(0, rows, seg.pick(logp, choice))
            entropies.append(seg.entropy(logp))

            active = [
                i
                for (i, edges), pick in zip(live, choice.tolist(), strict=True)
                if walks[i].take(edges[pick], env)
            ]

        return Rollouts(
            log_prob=log_prob,
            entropy=torch.cat(entropies) if entropies else log_prob.new_zeros(0),
            success=torch.tensor([w.success for w in walks], device=self.device),
            routes=[w.route for w in walks],
        )
