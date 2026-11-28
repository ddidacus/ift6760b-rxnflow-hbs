"""CAS-FullSpace: one k-means index over the whole catalog, not one per library.

The probe may retrieve any building block, so the library filter becomes a measurement
(``legal_fraction``) rather than a constraint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from rxnflow.core.types import ActionKey, State
from rxnflow.envs.env import SynthesisEnv
from rxnflow.gflownet.fwd.base import PROJECTION_ATTRIBUTE
from rxnflow.gflownet.fwd.cas.index import ClusteredActionIndex
from rxnflow.gflownet.fwd.cas.policy import CASPolicy, CASSettings
from rxnflow.gflownet.fwd.cas.trainer import CASRxnFlowTrainer
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel

logger = logging.getLogger(__name__)


@dataclass
class GlobalCASSettings(CASSettings):
    """``CASSettings`` plus the knobs only a global index needs."""

    # Cells in the global index; 2048 over 482,070 rows matches the per-library
    # variant's occupancy, so the two differ only in what may be retrieved.
    nlist: int = 2048
    # Legal rows a state wants before it stops opening cells.
    min_legal: int = 32
    # Hard cap on cells opened per (state, query), so a pathological state cannot
    # scan the catalog. At the measured densities the median state needs one.
    max_cells: int = 16
    # Rows drawn uniformly from an allowed library when the walk finds nothing at all.
    fallback_rows: int = 64

    def validate(self) -> None:
        super().validate()
        if self.bucket_term == "partition":
            # This variant opens cells until it has min_legal legal rows, so there is
            # no single opened cell whose log P could offset the partition. The
            # generalization -- logsumexp over the cells actually opened -- is a
            # semantic choice about what the action space is, not a default.
            raise ValueError(
                "cas.bucket_term='partition' is not defined for CAS-FullSpace, which "
                "opens several cells per state; use 'numerator' or 'off'"
            )


def build_global_index(
    env: SynthesisEnv, settings: GlobalCASSettings, device: torch.device
) -> ClusteredActionIndex:
    """One k-means over every synthon row in the catalog, fitted once."""
    features = {
        name: np.asarray(library.fingerprints, dtype=np.float32)
        for name, library in env.synthons.items()
    }
    total = sum(len(v) for v in features.values())
    nlist = int(np.clip(settings.nlist, 1, total))
    logger.info(
        "CAS-FullSpace: fitting %d centroids over %d rows from %d libraries",
        nlist, total, len(features),
    )
    index = ClusteredActionIndex(
        features,
        nlist,
        seed=settings.seed,
        iters=settings.kmeans_iters,
        train_sample=settings.train_sample,
        device=str(device),
    )
    assert not index.single_library, "global index must span every library"
    logger.info("CAS-FullSpace: %s", index.stats())
    return index


class GlobalCASPolicy(CASPolicy):
    """CAS over one global index; the library filter becomes a measurement."""

    def __init__(self, *args, global_index: ClusteredActionIndex, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = global_index
        self._lib_index = {n: i for i, n in enumerate(self.index.library_names)}
        # Share of retrieved rows executable for the state that retrieved them;
        # 1.0 by construction under per-library CAS.
        self.legal_fraction: float | None = None
        self.cells_walked: float | None = None
        self.fallback_rate: float | None = None

    def _observed_cell_log_p(self, library_name, probe, synthon_rows):
        """As CASPolicy, but the one global index addresses rows globally."""
        return self._cell_log_p_at(
            self.index, probe, self.index.global_row(library_name, synthon_rows)
        )

    def _assign_candidates(self, states: list[State], plans: list[tuple]):
        index = self.index
        cas: GlobalCASSettings = self.cas
        opened: list[dict[ActionKey, NDArray[np.int64]]] = [{} for _ in states]
        union: dict[str, list[NDArray[np.int64]]] = {}
        retrieved = legal = 0
        walked: list[int] = []
        fallbacks = 0
        observed: list[float] = []
        entropies: list[float] = []

        for rows, _, _, spaces_by_action, _, probes in plans:
            for key, probe in probes.items():
                subs = spaces_by_action[key]
                # library index -> the subspace that consumes rows from it
                route = {
                    self._lib_index[s.name[1]]: s.name
                    for s in subs
                    if s.name[1] is not None and s.name[1] in self._lib_index
                }
                if not route:
                    continue
                allowed = np.array(sorted(route), dtype=np.int32)
                scores = index.cell_scores(probe) / cas.temp_bucket.at(self.clock.value)
                with torch.no_grad():
                    log_p = torch.log_softmax(scores, dim=1)
                    observed.extend((-log_p.max(dim=1).values).cpu().tolist())
                    entropies.extend((-(log_p.exp() * log_p).sum(1)).cpu().tolist())
                    ranked = scores.argsort(dim=1, descending=True)
                    ranked = ranked[:, : cas.max_cells].cpu().numpy()

                for local, state_i in enumerate(rows):
                    found: list[NDArray[np.int64]] = []
                    count = opened_cells = 0
                    for cell in ranked[local]:
                        got = index.rows_in_cells(np.array([cell]))
                        opened_cells += 1
                        retrieved += len(got)
                        keep = got[np.isin(index.library_of[got], allowed)]
                        if len(keep):
                            found.append(keep)
                            count += len(keep)
                        if count >= cas.min_legal:
                            break
                    walked.append(opened_cells)
                    rows_found = (
                        np.concatenate(found) if found
                        else np.empty(0, dtype=np.int64)
                    )
                    if rows_found.size == 0:
                        # Nothing legal anywhere in the walk: keep the trajectory
                        # alive with a uniform draw, and count it.
                        fallbacks += 1
                        lib_i = int(allowed[self.rng.integers(len(allowed))])
                        name = index.library_names[lib_i]
                        size = len(self.env.synthons[name])
                        local_rows = np.unique(
                            self.rng.integers(size, size=min(cas.fallback_rows, size))
                        )
                        rows_found = index.global_row(name, local_rows)
                    legal += len(rows_found)
                    libs = index.library_of[rows_found]
                    locals_ = index.local_row_of[rows_found]
                    for lib_i in np.unique(libs):
                        name_key = route.get(int(lib_i))
                        if name_key is None:
                            continue
                        mine = np.sort(locals_[libs == lib_i])
                        prev = opened[state_i].get(name_key)
                        merged = (
                            np.union1d(prev, mine) if prev is not None else mine
                        )
                        opened[state_i][name_key] = merged
                        union.setdefault(name_key[1], []).append(merged)
                    # Subspaces this query reaches but that got nothing still need an
                    # entry, or step 4 raises on a missing key.
                    for s in subs:
                        opened[state_i].setdefault(
                            s.name, np.empty(0, dtype=np.int64)
                        )
                        union.setdefault(s.name[1], []).append(
                            np.empty(0, dtype=np.int64)
                        )

        if observed:
            self.cell_logp = float(np.mean(observed))
        if entropies:
            self.cell_entropy = float(np.mean(entropies))
        if retrieved:
            self.legal_fraction = legal / retrieved
        if walked:
            self.cells_walked = float(np.mean(walked))
            self.fallback_rate = fallbacks / len(walked)
        return opened, union


class GlobalCASRxnFlowTrainer(CASRxnFlowTrainer):
    """CAS-FullSpace: one index for the whole catalog instead of one per library."""

    def _prepare_models(self) -> None:
        from time import perf_counter

        from torch import nn

        started = perf_counter()
        self.global_index = build_global_index(self.env, self.cas_settings, self.device)
        # CASPolicy reads `dim` off this map and nothing else here uses it:
        # `_assign_candidates` is overridden and retrieves from the global index.
        self.indexes = {"__global__": self.global_index}
        dim = self.global_index.dim
        logger.info(
            "CAS-FullSpace index fitted in %.2fs", perf_counter() - started
        )
        for model in (self.model, self.sampling_model):
            projection = nn.Linear(self.config.model.synthon_dim, dim, bias=False).to(
                self.device
            )
            nn.init.xavier_uniform_(projection.weight)
            model.add_module(PROJECTION_ATTRIBUTE, projection)
        with torch.no_grad():
            getattr(self.sampling_model, PROJECTION_ATTRIBUTE).weight.copy_(
                getattr(self.model, PROJECTION_ATTRIBUTE).weight
            )

    def _build_policy(self, model: RxnFlowModel) -> RxnFlowPolicy:
        return GlobalCASPolicy(
            self.env,
            model,
            self.config,
            self.device,
            self.rng,
            self.cas_settings,
            self.indexes,
            getattr(model, PROJECTION_ATTRIBUTE),
            self.step_clock,
            global_index=self.global_index,
        )
