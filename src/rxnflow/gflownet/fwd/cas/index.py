"""Spherical k-means over a fixed feature matrix: cells, assignments and lookup.

Cells partition the catalog, so a query reaches a row through the cell holding it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def spherical_kmeans(
    data: torch.Tensor,
    nlist: int,
    *,
    iters: int = 25,
    seed: int = 0,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Lloyd's algorithm on the unit sphere, so the metric matches the policy's logit.

    Centroids are renormalized each step, or the assignment stops agreeing with cosine.
    Empty clusters are reseeded from the points furthest from their own centroid.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = len(data)
    nlist = min(nlist, count)
    picks = torch.randperm(count, generator=generator)[:nlist].to(data.device)
    centroids = data[picks].clone()

    previous = None
    for step in range(iters):
        assignment, best = _assign_chunked(data, centroids)
        total = torch.zeros_like(centroids)
        total.index_add_(0, assignment, data)
        counts = torch.bincount(assignment, minlength=nlist)

        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel():
            # Furthest-point reseeding: the worst-fit rows become new centres, which
            # both fills the empty cells and splits the overfull ones.
            worst = torch.argsort(best)[: empty.numel()]
            total[empty] = data[worst]
            counts[empty] = 1

        centroids = total / counts.clamp_min(1).unsqueeze(1).to(total.dtype)
        centroids = torch.nn.functional.normalize(centroids, dim=1)

        inertia = float(best.mean())
        if previous is not None and abs(previous - inertia) < tol:
            logger.info("k-means converged at iteration %d (inertia %.5f)", step, inertia)
            break
        previous = inertia
    return centroids


def _assign_chunked(
    data: torch.Tensor, centroids: torch.Tensor, chunk: int = 65_536
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest centroid per row, without materializing [N, nlist]."""
    assignment = torch.empty(len(data), dtype=torch.long, device=data.device)
    similarity = torch.empty(len(data), dtype=data.dtype, device=data.device)
    for start in range(0, len(data), chunk):
        stop = min(start + chunk, len(data))
        scores = data[start:stop] @ centroids.T
        top, which = scores.max(dim=1)
        assignment[start:stop] = which
        similarity[start:stop] = top
    # Cosine similarity to the assigned centre; larger is a better fit.
    return assignment, similarity


@dataclass
class ClusterStats:
    nlist: int
    rows: int
    occupied: int
    mean: float
    median: float
    p90: float
    maximum: int

    def __str__(self) -> str:
        return (
            f"nlist={self.nlist:,} rows={self.rows:,} occupied={self.occupied:,} "
            f"({self.occupied / self.nlist:.1%}) occupancy mean={self.mean:.1f} "
            f"median={self.median:.0f} p90={self.p90:.0f} max={self.maximum:,}"
        )


class ClusteredActionIndex:
    """Inverted lists over one L2-normalized bank of synthon features.

    The bank never changes, so the index stays valid for a whole run; only the probe
    moves. Unlike the hash, the partition was fitted to these vectors.
    """

    def __init__(
        self,
        features: dict[str, NDArray],
        nlist: int,
        *,
        seed: int = 0,
        iters: int = 25,
        train_sample: int | None = 200_000,
        device: str = "cpu",
    ):
        self.nlist = int(nlist)
        self.device = torch.device(device)
        self.library_names = list(features)

        counts = [len(features[name]) for name in self.library_names]
        self.rows = int(sum(counts))
        dim = int(features[self.library_names[0]].shape[1])
        self.dim = dim

        # Origin of every global row, so a retrieved row is still an executable action.
        self.library_of = np.repeat(
            np.arange(len(self.library_names), dtype=np.int32), counts
        )
        self.local_row_of = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in counts]
        )
        self._library_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._index_of_library = {n: i for i, n in enumerate(self.library_names)}

        bank = torch.empty((self.rows, dim), dtype=torch.float32, device=self.device)
        at = 0
        for name in self.library_names:
            block = torch.from_numpy(np.ascontiguousarray(features[name])).to(
                self.device, torch.float32
            )
            bank[at : at + len(block)] = torch.nn.functional.normalize(block, dim=1)
            at += len(block)
        self.bank = bank

        generator = torch.Generator(device="cpu").manual_seed(seed)
        if train_sample is not None and train_sample < self.rows:
            picks = torch.randperm(self.rows, generator=generator)[:train_sample]
            training = bank[picks.to(self.device)]
        else:
            training = bank
        logger.info(
            "fitting %d centroids on %d of %d rows", self.nlist, len(training), self.rows
        )
        self.centroids = spherical_kmeans(
            training, self.nlist, iters=iters, seed=seed
        )

        assignment, _ = _assign_chunked(bank, self.centroids)
        self.cell_of = assignment
        # CSR inverted lists: rows sorted by cell, with per-cell offsets.
        order = torch.argsort(assignment, stable=True)
        self.order = order.cpu().numpy()
        sizes = torch.bincount(assignment, minlength=self.nlist).cpu().numpy()
        self.sizes = sizes
        self.start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        logger.info("%s", self.stats())

    # ---- geometry / sizing

    def stats(self) -> ClusterStats:
        nonzero = self.sizes[self.sizes > 0]
        return ClusterStats(
            nlist=self.nlist,
            rows=self.rows,
            occupied=int(len(nonzero)),
            mean=float(self.sizes.mean()),
            median=float(np.median(nonzero)) if len(nonzero) else 0.0,
            p90=float(np.percentile(nonzero, 90)) if len(nonzero) else 0.0,
            maximum=int(self.sizes.max()),
        )

    # ---- queries

    def cell_scores(self, probes: torch.Tensor) -> torch.Tensor:
        """``probes @ centroids.T`` -- the differentiable soft-assignment logits.

        The gradient path the hash never had: ``softmax(scores / tau)`` is a
        distribution over cells the loss can move.
        """
        return probes.to(self.centroids.dtype) @ self.centroids.T

    def nearest_cells(self, probes: torch.Tensor, nprobe: int = 1) -> torch.Tensor:
        nprobe = max(1, min(int(nprobe), self.nlist))
        return self.cell_scores(probes).topk(nprobe, dim=1).indices

    @property
    def single_library(self) -> bool:
        """True when ``row == local synthon index``, which the policy assumes.

        Holds while an index covers one library. Once it spans several, callers must go
        through ``restrict`` / ``global_row`` instead.
        """
        return len(self.library_names) == 1

    def rows_in_cells(self, cells: NDArray[np.int64]) -> NDArray[np.int64]:
        """Row ids held by these cells, local synthon indices only if ``single_library``."""
        pieces = [self.order[self.start[c] : self.start[c + 1]] for c in np.atleast_1d(cells)]
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def lookup(
        self, probes: torch.Tensor | NDArray[np.float32], nprobe: int = 1
    ) -> list[NDArray[np.int64]]:
        """Candidate global rows per probe: the members of its ``nprobe`` nearest cells."""
        if isinstance(probes, np.ndarray):
            probes = torch.from_numpy(np.ascontiguousarray(probes))
        cells = self.nearest_cells(probes.to(self.device), nprobe).cpu().numpy()
        return [self.rows_in_cells(row) for row in cells]

    # ---- mapping back to executable actions

    def restrict(
        self, global_rows: NDArray[np.int64], library_name: str
    ) -> NDArray[np.int64]:
        """Local row ids of the retrieved rows that belong to this library.

        The share of ``global_rows`` this drops is the action-level validity rate.
        """
        which = self._index_of_library[library_name]
        keep = global_rows[self.library_of[global_rows] == which]
        return self.local_row_of[keep]

    def global_row(
        self, library_name: str, local_rows: NDArray[np.int64] | int
    ) -> NDArray[np.int64]:
        """Global row ids for local rows of one library -- the inverse of ``restrict``."""
        which = self._index_of_library[library_name]
        return np.asarray(local_rows, dtype=np.int64) + self._library_start[which]

    def global_rows_of(self, library_name: str) -> tuple[int, int]:
        """[start, stop) of this library's block in the global bank."""
        which = self._index_of_library[library_name]
        return int(self._library_start[which]), int(self._library_start[which + 1])
