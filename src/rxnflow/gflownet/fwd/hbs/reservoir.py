"""Locality-sensitive hash over the fixed synthon features, built once per run.

Random hyperplanes give every row a code. A query opens its own bucket first, then
Hamming neighbours, until ``min_rows`` are in hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ortho_group

logger = logging.getLogger(__name__)

# int64 codes: one bit per hyperplane, so a code holds at most 62 bits with room to
# spare for the sign bit. Depth never needs to approach this in practice.
MAX_DEPTH = 62


def probe_masks(depth: int, max_radius: int, limit: int) -> list[NDArray[np.int64]]:
    """Code offsets grouped by Hamming distance: [[0], single flips, double flips, ...].

    XOR-ing a code with these visits buckets nearest-first, at C(depth, r) lookups for
    radius r rather than a distance to every occupied bucket.
    """
    levels: list[NDArray[np.int64]] = [np.zeros(1, dtype=np.int64)]
    total = 1
    for radius in range(1, max_radius + 1):
        combos = list(combinations(range(depth), radius))
        if not combos or total + len(combos) > limit:
            break
        masks = np.zeros(len(combos), dtype=np.int64)
        for i, bits in enumerate(combos):
            value = np.int64(0)
            for bit in bits:
                value |= np.int64(1) << np.int64(bit)
            masks[i] = value
        levels.append(masks)
        total += len(masks)
    return levels


def normalize_rows(features: NDArray) -> NDArray[np.float32]:
    """L2-normalize, mirroring upstream's F.normalize on synthon embeddings."""
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


@dataclass
class LibraryIndex:
    """One library's hash table over its fixed features."""

    name: str
    num_rows: int
    depth: int
    mean: NDArray[np.float32]
    # Rows ordered by bucket code, and where each distinct code starts in that order.
    order: NDArray[np.int64]
    bucket_codes: NDArray[np.int64]
    bucket_start: NDArray[np.int64]
    # The partition as the policy scores it: each occupied code as +-1 per bit, and
    # the bucket every row belongs to. <code, hyperplane scores> is a differentiable
    # similarity whose argmax over all codes is the hash code itself.
    code_signs: NDArray[np.float32]
    row_bucket: NDArray[np.int64]

    @property
    def num_buckets(self) -> int:
        return len(self.bucket_codes)

    @property
    def sizes(self) -> NDArray[np.int64]:
        return np.diff(self.bucket_start)

    def rows_in(self, bucket: int) -> NDArray[np.int64]:
        return self.order[self.bucket_start[bucket] : self.bucket_start[bucket + 1]]


class HashedReservoir:
    """Fixed-feature hash tables for every library, built once.

    ``lookup`` returns the rows of the bucket a query falls in, widening to the nearest
    buckets only when it holds too few. There is no sample size.
    """

    def __init__(
        self,
        features: dict[str, NDArray],
        depth: int,
        *,
        seed: int = 0,
        center: bool = False,
        max_radius: int = 6,
        probe_limit: int = 8192,
        dense_buckets: int = 4096,
        chunk: int = 8192,
    ):
        if not features:
            raise ValueError("reservoir needs at least one library")
        dims = {name: value.shape[1] for name, value in features.items()}
        if len(set(dims.values())) != 1:
            raise ValueError(f"libraries must share a feature space, got {dims}")
        self.dim = next(iter(dims.values()))
        # One depth for every library: code length is the resolution of the search,
        # not a per-library budget.
        self.depth = int(np.clip(depth, 1, MAX_DEPTH))
        # Hashing about the origin estimates the angle the logit ranks. Centring on
        # the reservoir mean estimates it about the centroid instead, which only
        # matches when queries are distributed like the reservoir.
        self.center = center
        rng = np.random.default_rng(seed)
        # Orthogonal rather than Gaussian hyperplanes (Super-Bit LSH): lower variance
        # on the angle estimate, defined only for depth <= dim.
        if self.depth > self.dim:
            raise ValueError(
                f"depth {self.depth} exceeds the feature dimension {self.dim}: "
                "orthogonal hyperplanes do not exist beyond it"
            )
        self.hyperplanes = ortho_group.rvs(self.dim, random_state=rng).astype(np.float32)
        self.masks = probe_masks(self.depth, max_radius, probe_limit)
        # Two exact ways to visit buckets in Hamming order, dispatched on bucket
        # count: multi-probe enumerates nearby codes (cheap when codes are dense),
        # the exact scan distances every occupied bucket (cheap when there are few).
        self.dense_buckets = dense_buckets
        # References, not copies: the environment already holds these arrays.
        self.features = features
        self.index: dict[str, LibraryIndex] = {}
        for name, values in features.items():
            self.index[name] = self._build(name, values, chunk)

    def _build(self, name: str, values: NDArray, chunk: int) -> LibraryIndex:
        rows = len(values)
        depth = self.depth
        planes = self.hyperplanes[:depth]
        # Mean of the normalized vectors: the hash splits the data around its own
        # centroid, so bits stay informative for features sitting in a cone.
        if self.center:
            total = np.zeros(self.dim, dtype=np.float64)
            for start in range(0, rows, chunk):
                total += normalize_rows(values[start : start + chunk]).sum(axis=0)
            mean = (total / max(rows, 1)).astype(np.float32)
        else:
            mean = np.zeros(self.dim, dtype=np.float32)

        codes = np.empty(rows, dtype=np.int64)
        weights = (np.int64(1) << np.arange(depth, dtype=np.int64))[None, :]
        for start in range(0, rows, chunk):
            block = normalize_rows(values[start : start + chunk]) - mean
            bits = (block @ planes.T) > 0
            codes[start : start + len(block)] = (bits * weights).sum(axis=1)

        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
        start_positions = np.concatenate(
            ([0], boundaries, [rows])
        ).astype(np.int64)
        bucket_codes = sorted_codes[start_positions[:-1]]
        bits = (bucket_codes[:, None] >> np.arange(depth, dtype=np.int64)) & 1
        code_signs = (bits * 2 - 1).astype(np.float32)
        row_bucket = np.empty(rows, dtype=np.int64)
        row_bucket[order] = np.repeat(
            np.arange(len(bucket_codes), dtype=np.int64), np.diff(start_positions)
        )
        return LibraryIndex(
            name=name,
            num_rows=rows,
            depth=depth,
            mean=mean,
            order=order,
            bucket_codes=bucket_codes,
            bucket_start=start_positions,
            code_signs=code_signs,
            row_bucket=row_bucket,
        )

    # ---- queries

    def hash_queries(
        self, library_name: str, queries: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Bit pattern of each query in this library's partition, as 0/1 floats."""
        entry = self.index[library_name]
        centred = normalize_rows(queries) - entry.mean
        return ((centred @ self.hyperplanes[: entry.depth].T) > 0).astype(np.float32)

    def lookup(
        self, library_name: str, queries: NDArray[np.float32], min_rows: int = 256
    ) -> list[NDArray[np.int64]]:
        """Rows to score for each query, nearest bucket first.

        Stops once ``min_rows`` rows are in hand. Nothing is drawn at random, but the
        set cannot collapse to the few rows one deep bucket holds.
        """
        entry = self.index[library_name]
        codes = self.codes_of(library_name, queries)
        need = min(max(min_rows, 1), entry.num_rows)
        if entry.num_buckets <= self.dense_buckets:
            return self._lookup_dense(entry, codes, need)
        opened, short = self._lookup_probe(entry, codes, need)
        if short:
            # Multi-probe exhausted its radius; the exact scan always fills, and a
            # query needing it sits in a sparse part of the code space.
            exact = self._lookup_dense(entry, codes[short], need)
            for position, rows in zip(short, exact, strict=True):
                opened[position] = rows
        return opened

    def _lookup_dense(
        self, entry: LibraryIndex, codes: NDArray[np.int64], need: int
    ) -> list[NDArray[np.int64]]:
        """Exact: order every occupied bucket by Hamming distance and open until full."""
        distance = self._hamming_to_buckets(entry, codes)
        sizes = entry.sizes
        results: list[NDArray[np.int64]] = []
        for row in distance:
            ranked = np.argsort(row, kind="stable")
            collected = sizes[ranked].cumsum()
            opened = min(int(np.searchsorted(collected, need) + 1), len(ranked))
            chosen = ranked[:opened]
            if len(chosen) == 1:
                results.append(entry.rows_in(int(chosen[0])))
            else:
                results.append(np.concatenate([entry.rows_in(int(b)) for b in chosen]))
        return results

    def _lookup_probe(
        self, entry: LibraryIndex, codes: NDArray[np.int64], need: int
    ) -> tuple[list[NDArray[np.int64]], list[int]]:
        """Multi-probe: XOR the query code with offsets grouped by Hamming distance."""
        count = len(codes)
        collected: list[list[int]] = [[] for _ in range(count)]
        found = np.zeros(count, dtype=np.int64)
        pending = np.arange(count)
        sizes = entry.sizes
        for masks in self.masks:
            if pending.size == 0:
                break
            probes = codes[pending, None] ^ masks[None, :]
            slots = np.searchsorted(entry.bucket_codes, probes)
            np.clip(slots, 0, entry.num_buckets - 1, out=slots)
            hit = entry.bucket_codes[slots] == probes
            matched_sizes = np.where(hit, sizes[slots], 0)
            for local, query in enumerate(pending):
                matched = slots[local][hit[local]]
                if matched.size:
                    # Sorted within the level, so equidistant buckets open in bucket
                    # order -- the exact scan's tie-break, so both paths agree.
                    collected[int(query)].extend(np.sort(matched).tolist())
            found[pending] += matched_sizes.sum(axis=1)
            pending = pending[found[pending] < need]

        results: list[NDArray[np.int64]] = []
        short: list[int] = []
        for query, buckets in enumerate(collected):
            if found[query] < need:
                short.append(query)
                results.append(np.empty(0, dtype=np.int64))
                continue
            # Stop at the first bucket that meets the floor, or nearer-coded queries
            # would get larger candidate sets than the exact scan gives.
            total = 0
            opened = []
            for bucket in buckets:
                opened.append(bucket)
                total += int(sizes[bucket])
                if total >= need:
                    break
            results.append(np.concatenate([entry.rows_in(b) for b in opened]))
        return results, short

    def _hamming_to_buckets(
        self, entry: LibraryIndex, codes: NDArray[np.int64]
    ) -> NDArray[np.int16]:
        """[queries, buckets] Hamming distances between packed codes."""
        differing = entry.bucket_codes[None, :] ^ codes[:, None]
        distance = np.zeros(differing.shape, dtype=np.int16)
        for bit in range(entry.depth):
            distance += ((differing >> np.int64(bit)) & 1).astype(np.int16)
        return distance

    def partition_of(
        self, library_name: str
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Hyperplanes, centroid and occupied codes, for scoring buckets in torch."""
        entry = self.index[library_name]
        return self.hyperplanes[: entry.depth], entry.mean, entry.code_signs

    def rows_of_buckets(self, library_name: str) -> list[NDArray[np.int64]]:
        """Rows held by each occupied bucket, indexed as ``code_signs`` is."""
        entry = self.index[library_name]
        return [entry.rows_in(b) for b in range(entry.num_buckets)]

    def codes_of(
        self, library_name: str, queries: NDArray[np.float32]
    ) -> NDArray[np.int64]:
        """Packed bit code of each query in this library's partition."""
        entry = self.index[library_name]
        bits = self.hash_queries(library_name, queries) > 0
        weights = np.int64(1) << np.arange(entry.depth, dtype=np.int64)
        return (bits * weights[None, :]).sum(axis=1)

    def exact_scores(
        self, library_name: str, queries: NDArray[np.float32], chunk: int = 8192
    ) -> NDArray[np.float32]:
        """[queries, rows] exact inner products, for probes.

        Chunked over rows: normalizing the largest library in one piece costs 250MB.
        """
        values = self.features[library_name]
        scores = np.empty((len(queries), len(values)), dtype=np.float32)
        unit = normalize_rows(queries)
        for start in range(0, len(values), chunk):
            block = normalize_rows(values[start : start + chunk])
            scores[:, start : start + len(block)] = unit @ block.T
        return scores

    def summary(self) -> str:
        depths = [e.depth for e in self.index.values()]
        sizes = np.concatenate([e.sizes for e in self.index.values()])
        occupancy = [e.num_buckets for e in self.index.values()]
        return (
            f"{len(self.index)} libraries, "
            f"{sum(e.num_rows for e in self.index.values()):,} rows, dim {self.dim}; "
            f"depth {min(depths)}; probe radius {len(self.masks) - 1} "
            f"({sum(len(m) for m in self.masks)} buckets reachable); "
            f"buckets per library {min(occupancy)}-{max(occupancy)}; "
            f"bucket size min {sizes.min()} / median {int(np.median(sizes))} / "
            f"mean {sizes.mean():.0f} / max {sizes.max()}"
        )


if __name__ == "__main__":
    # Recall and cost against brute force, and agreement with an exact bucket scan.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(0)
    dim, rows = 678, 20_000
    reservoir_bits = (rng.random((rows, dim)) < 0.08).astype(np.float32)
    queries = normalize_rows(
        reservoir_bits[rng.integers(0, rows, 200)] + 0.02 * rng.random((200, dim))
    )
    truth = normalize_rows(reservoir_bits) @ queries.T
    for depth in (8, 12, 16):
        index = HashedReservoir({"lib": reservoir_bits}, depth, seed=0)
        print(f"\ndepth {depth}: {index.summary()}")
        for floor in (1, 256):
            found = index.lookup("lib", queries, min_rows=floor)
            hits = sum(
                int(np.argmax(truth[:, i])) in set(c.tolist())
                for i, c in enumerate(found)
            )
            scanned = float(np.mean([len(c) for c in found]))
            print(
                f"  min_rows {floor:>4}: top-1 found {100 * hits / len(queries):5.1f}% "
                f"| scanned {scanned:6.0f} of {rows} ({100 * scanned / rows:4.1f}%)"
            )
