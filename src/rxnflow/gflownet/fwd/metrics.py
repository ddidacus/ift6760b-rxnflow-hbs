"""Per-trajectory diversity, candidate-set and latency metrics, identical in all arms.

One JSONL record per measured step. ``MetricsSettings`` sets the cadence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

if TYPE_CHECKING:
    from rxnflow.core.types import State, Trajectory
    from rxnflow.envs.env import SynthesisEnv
    from rxnflow.gflownet.policy import ActionCategorical

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048
PACKED_BYTES = FP_BITS // 8

# Byte popcount table: Tanimoto over packed fingerprints needs |a & b| and |a|, |b|.
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1
).astype(np.int32)


# ======== Fingerprints and the two chemical primitives


def morgan_packed(smiles: str) -> NDArray[np.uint8] | None:
    """Bit-packed Morgan fingerprint, or None when RDKit cannot read the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bits = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS), bits
    )
    return np.packbits(bits)


def _popcount(packed: NDArray[np.uint8]) -> NDArray[np.int32]:
    return _POPCOUNT[packed].sum(axis=-1)


def mean_pairwise_similarity(
    packed: NDArray[np.uint8],
    max_pairs: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Mean Tanimoto similarity over pairs of packed fingerprints.

    Exact when the set has at most max_pairs pairs (or max_pairs is None); otherwise
    estimated from max_pairs random distinct pairs. Returns (mean, pairs used); the
    mean is nan for fewer than two fingerprints.
    """
    count = len(packed)
    if count < 2:
        return math.nan, 0
    total_pairs = count * (count - 1) // 2
    sizes = _popcount(packed)

    if max_pairs is None or total_pairs <= max_pairs:
        # Row at a time: exact without materializing an n x n x 256 intersection.
        total = 0.0
        for i in range(count - 1):
            others = packed[i + 1 :]
            shared = _popcount(np.bitwise_and(packed[i], others))
            union = sizes[i] + sizes[i + 1 :] - shared
            total += float(np.where(union > 0, shared / np.maximum(union, 1), 1.0).sum())
        return total / total_pairs, total_pairs

    generator = rng if rng is not None else np.random.default_rng(0)
    # Oversample to absorb the i == j draws, then keep exactly max_pairs.
    draw = int(max_pairs * 1.3) + 16
    left = generator.integers(0, count, draw)
    right = generator.integers(0, count, draw)
    keep = left != right
    left, right = left[keep][:max_pairs], right[keep][:max_pairs]
    shared = _popcount(np.bitwise_and(packed[left], packed[right]))
    union = sizes[left] + sizes[right] - shared
    similarity = np.where(union > 0, shared / np.maximum(union, 1), 1.0)
    return float(similarity.mean()), int(len(left))


def tanimoto_diversity_packed(
    packed: NDArray[np.uint8],
    max_pairs: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """1 - mean pairwise Tanimoto similarity, with the pair count behind it."""
    similarity, pairs = mean_pairwise_similarity(packed, max_pairs, rng)
    return (math.nan if math.isnan(similarity) else 1.0 - similarity), pairs


def tanimoto_diversity(
    smiles: list[str],
    max_pairs: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Repo-compatible diversity over SMILES; 0.0 when fewer than two parse."""
    packed = [fp for fp in (morgan_packed(value) for value in smiles) if fp is not None]
    if len(packed) < 2:
        return 0.0
    value, _ = tanimoto_diversity_packed(np.stack(packed), max_pairs, rng)
    return 0.0 if math.isnan(value) else value


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None when it cannot be derived."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        # GetScaffoldForMol re-runs the valence check and raises for molecules
        # MolFromSmiles accepted; treat that as "no scaffold", not a failed run.
        return None
    return None if scaffold is None else Chem.MolToSmiles(scaffold)


def count_scaffolds(smiles: list[str]) -> int:
    return len({s for s in (murcko_scaffold(value) for value in smiles) if s})


# ======== Settings and the lazy synthon fingerprint table


@dataclass
class MetricsSettings:
    """How often and how widely to measure. every=0 disables the collector."""

    every: int = 25
    # Trajectories measured per recorded step, taken from the lowest-indexed ones
    # still active. Conditions are drawn per index, so these are an unbiased sample.
    states: int = 8
    # Pair budget per diversity estimate; a candidate set of 167 rows is exact under
    # 2000 only up to 63 rows, so library-level numbers are sampled estimates.
    max_pairs: int = 2000
    # Rows drawn once per library to estimate its own diversity, the scale a
    # candidate set is read against. Fingerprinting is the cost, so keep it small.
    reference_rows: int = 256
    seed: int = 0

    def validate(self) -> None:
        if self.every < 0:
            raise ValueError("metrics.every must be non-negative (0 disables)")
        if self.every and self.states < 1:
            raise ValueError("metrics.states must be positive")
        if self.every and self.max_pairs < 1:
            raise ValueError("metrics.max_pairs must be positive")


class SynthonFingerprints:
    """Per-library packed fingerprints, computed once per row on first use."""

    def __init__(self, env: SynthesisEnv):
        self.env = env
        self._table: dict[str, NDArray[np.uint8]] = {}
        self._known: dict[str, NDArray[np.bool_]] = {}
        self._reference: dict[str, float] = {}
        self.failures = 0

    def reference(self, library_name: str, sample: int, rng, max_pairs: int) -> float:
        """The library's own diversity: the scale a candidate set is read against.

        A uniform candidate set should land near this, a retrieved one below it.
        Estimated once per library from ``sample`` random rows.
        """
        cached = self._reference.get(library_name)
        if cached is not None:
            return cached
        count = len(self.env.synthons[library_name])
        draw = min(sample, count)
        chosen = rng.choice(count, draw, replace=False) if draw < count else np.arange(count)
        value, _ = tanimoto_diversity_packed(
            self.rows(library_name, np.sort(chosen.astype(np.int64))), max_pairs, rng
        )
        self._reference[library_name] = float(value)
        return self._reference[library_name]

    def rows(self, library_name: str, rows: NDArray[np.int64]) -> NDArray[np.uint8]:
        table = self._table.get(library_name)
        if table is None:
            count = len(self.env.synthons[library_name])
            table = np.zeros((count, PACKED_BYTES), dtype=np.uint8)
            self._table[library_name] = table
            self._known[library_name] = np.zeros(count, dtype=bool)
        known = self._known[library_name]
        missing = rows[~known[rows]]
        if missing.size:
            smiles = self.env.synthons[library_name].smiles
            for row in np.unique(missing):
                packed = morgan_packed(smiles[row])
                if packed is None:
                    self.failures += 1  # leaves an all-zero row
                else:
                    table[row] = packed
                known[row] = True
        return table[rows]


# ======== Collector


class TrajectoryMetrics:
    """Times a batch's sampling and measures the trajectories it produced.

    The policy calls begin_sampling/begin_iteration/record_candidates/end_iteration/
    end_sampling; the trainer calls flush once the batch is complete.
    """

    def __init__(self, env: SynthesisEnv, settings: MetricsSettings, path: Path):
        self.env = env
        self.settings = settings
        self.path = Path(path)
        self.fingerprints = SynthonFingerprints(env)
        self.rng = np.random.default_rng(settings.seed)
        self._calls = 0
        self._active = False
        self._recording = False
        self._start: float | None = None
        self._iteration_start: list[float] = []
        self._policy_ms: list[float] = []
        self._candidates: list[dict[int, tuple[float, float, int, int]]] = []
        self._total_ms: float | None = None
        self._mark: float = 0.0

    # ---- policy side

    def begin_sampling(self) -> None:
        self._calls += 1
        self._recording = bool(
            self.settings.every and self._calls % self.settings.every == 0
        )
        self._active = True
        self._start = perf_counter()
        self._iteration_start = []
        self._policy_ms = []
        self._candidates = []
        self._total_ms = None

    def begin_iteration(self) -> None:
        if not self._active:
            return
        self._iteration_start.append((perf_counter() - self._start) * 1000.0)
        self._candidates.append({})
        self._mark = perf_counter()

    def end_iteration(self) -> None:
        if not self._active:
            return
        self._policy_ms.append((perf_counter() - self._mark) * 1000.0)

    def end_sampling(self) -> None:
        if not self._active:
            return
        self._total_ms = (perf_counter() - self._start) * 1000.0
        self._active = False

    @property
    def recording(self) -> bool:
        return self._active and self._recording

    def record_candidates(
        self, states: list[State], categorical: ActionCategorical
    ) -> None:
        """Measure the candidate sets of the first settings.states states."""
        if not self.recording or not self._candidates:
            return
        entry = self._candidates[-1]
        for position in range(min(self.settings.states, len(states))):
            per_library: list[float] = []
            reference: list[float] = []
            pooled: list[NDArray[np.uint8]] = []
            scored = 0
            for action_logits in categorical.action_logits[position]:
                subspace = action_logits.subspace
                library_name = subspace.name[1]
                if library_name is None:
                    continue  # unimolecular action: one row, nothing to compare
                indices = subspace.sample_indices
                if indices is None or len(indices) >= subspace.num_actions:
                    continue  # scored whole, so no subset was selected
                rows = np.asarray(indices, dtype=np.int64)
                packed = self.fingerprints.rows(library_name, rows)
                value, _ = tanimoto_diversity_packed(
                    packed, self.settings.max_pairs, self.rng
                )
                if not math.isnan(value):
                    per_library.append(value)
                    reference.append(
                        self.fingerprints.reference(
                            library_name,
                            self.settings.reference_rows,
                            self.rng,
                            self.settings.max_pairs,
                        )
                    )
                pooled.append(packed)
                scored += len(rows)
            if not per_library:
                continue
            union, _ = tanimoto_diversity_packed(
                np.concatenate(pooled), self.settings.max_pairs, self.rng
            )
            entry[position] = (
                float(np.mean(per_library)),
                float(union),
                len(per_library),
                scored,
                float(np.mean(reference)) if reference else math.nan,
            )

    # ---- trainer side

    def _route_molecules(self, trajectory: Trajectory) -> tuple[list[str], list[str]]:
        """Selected synthons and obtained products of one trajectory."""
        synthons: list[str] = []
        products: list[str] = []
        for transition in trajectory.steps:
            action = transition.action
            if action.library_name is not None and action.synthon_index is not None:
                library = self.env.synthons[action.library_name]
                synthons.append(library.smiles[action.synthon_index])
            if transition.product_smiles:
                products.append(transition.product_smiles)
        return synthons, products

    def flush(
        self,
        step: int,
        trajectories: list[Trajectory],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Write one line for this batch and return its summary, or None if idle."""
        if self._total_ms is None or not trajectories:
            return None

        lengths = [len(value.steps) for value in trajectories]
        iterations = len(self._iteration_start)

        # A trajectory of L steps was advanced in iterations 0..L-1. Positions within
        # an iteration follow the sampler's own index order, so they map back exactly.
        latency_ms: list[float] = []
        policy_ms: list[float] = []
        for length in lengths:
            latency_ms.append(
                self._iteration_start[length]
                if length < iterations
                else self._total_ms
            )
            policy_ms.append(float(sum(self._policy_ms[:length])))

        record: dict[str, Any] = {
            "step": step,
            "num_trajectories": len(trajectories),
            "iterations": iterations,
            "sampling_ms": round(self._total_ms, 3),
            "traj_len": lengths,
            "latency_ms": [round(value, 3) for value in latency_ms],
            "policy_ms": [round(value, 3) for value in policy_ms],
        }
        for key, value in (extra or {}).items():
            if value is not None:
                record[key] = value
        summary: dict[str, Any] = {
            "mean_latency_ms": float(np.mean(latency_ms)),
            "p90_latency_ms": float(np.percentile(latency_ms, 90)),
            "mean_policy_ms": float(np.mean(policy_ms)),
        }
        if record.get("probe_alignment") is not None:
            summary["probe_alignment"] = record["probe_alignment"]

        if self._recording:
            # Only the baseline policy records candidate sets, so these fields are
            # absent for an arm that retrieves rather than samples.
            by_trajectory: dict[int, list[tuple[float, float, int, int, float]]] = {}
            for iteration, entry in enumerate(self._candidates):
                active = [i for i, length in enumerate(lengths) if length > iteration]
                for position, values in entry.items():
                    if position < len(active):
                        by_trajectory.setdefault(active[position], []).append(values)

            block_library: list[float | None] = []
            block_union: list[float | None] = []
            subspaces: list[float | None] = []
            candidates: list[float | None] = []
            library_reference: list[float | None] = []
            for index in range(len(trajectories)):
                measured = by_trajectory.get(index)
                if not measured:
                    block_library.append(None)
                    block_union.append(None)
                    subspaces.append(None)
                    candidates.append(None)
                    library_reference.append(None)
                    continue
                block_library.append(round(float(np.mean([m[0] for m in measured])), 5))
                block_union.append(round(float(np.mean([m[1] for m in measured])), 5))
                subspaces.append(float(np.mean([m[2] for m in measured])))
                candidates.append(float(np.mean([m[3] for m in measured])))
                library_reference.append(
                    round(float(np.mean([m[4] for m in measured])), 5)
                )

            route_scaffolds: list[int] = []
            route_div: list[float] = []
            num_synthons: list[int] = []
            num_products: list[int] = []
            for trajectory in trajectories:
                synthons, products = self._route_molecules(trajectory)
                molecules = synthons + products
                num_synthons.append(len(synthons))
                num_products.append(len(products))
                route_scaffolds.append(count_scaffolds(molecules))
                route_div.append(round(tanimoto_diversity(molecules), 5))

            measured_any = any(v is not None for v in block_library)
            record.update(
                {
                    "route_scaffolds": route_scaffolds,
                    "route_div": route_div,
                    "num_synthons": num_synthons,
                    "num_products": num_products,
                    "max_pairs": self.settings.max_pairs,
                }
            )
            candidate_fields = {
                    "block_div_library": block_library,
                    "block_div_union": block_union,
                    "num_subspaces_measured": subspaces,
                    "num_candidates_measured": candidates,
                    "library_div_reference": library_reference,
            }
            if measured_any:
                record.update(candidate_fields)
            measured_library = [v for v in block_library if v is not None]
            measured_union = [v for v in block_union if v is not None]
            measured_reference = [
                v for v in library_reference if v is not None and not math.isnan(v)
            ]
            summary.update(
                {
                    "block_div_library": (
                        float(np.mean(measured_library)) if measured_library else None
                    ),
                    "block_div_union": (
                        float(np.mean(measured_union)) if measured_union else None
                    ),
                    "library_div_reference": (
                        float(np.mean(measured_reference)) if measured_reference else None
                    ),
                    "route_scaffolds": float(np.mean(route_scaffolds)),
                    "route_div": float(np.mean(route_div)),
                    "trajectories_measured": len(measured_library),
                }
            )

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._total_ms = None
        return summary


def format_summary(summary: dict[str, Any]) -> str:
    parts = [
        f"latency {summary['mean_latency_ms']:.0f}ms"
        f" (p90 {summary['p90_latency_ms']:.0f}ms,"
        f" policy {summary['mean_policy_ms']:.0f}ms)"
    ]
    if summary.get("block_div_library") is not None:
        scale = summary.get("library_div_reference")
        against = (
            f" against library {scale:.4f} ({summary['block_div_library'] / scale:.3f}x)"
            if scale
            else ""
        )
        parts.append(
            f"candidate div {summary['block_div_library']:.4f} per library /"
            f" {summary['block_div_union']:.4f} pooled{against}"
            f" over {summary['trajectories_measured']} trajectories"
        )
    if summary.get("probe_alignment") is not None:
        parts.append(f"probe alignment {summary['probe_alignment']:+.4f}")
    if summary.get("route_div") is not None:
        parts.append(
            f"route div {summary['route_div']:.4f},"
            f" {summary['route_scaffolds']:.2f} scaffolds"
        )
    return "; ".join(parts)
