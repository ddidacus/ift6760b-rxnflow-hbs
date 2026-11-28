"""Does the hashed reservoir retrieve nearest neighbours at all?

The live runs report the policy's queries retrieving at roughly 1-2x chance
(``training.log``: "best row in bucket 0.0% (chance 0.3%)"), which is the same as
not retrieving. Two very different things could cause that, and they need opposite
fixes:

1. **The index is misconfigured.** Then even a query that *is* a data row -- the
   easiest possible query, whose nearest neighbour is itself -- fails to retrieve
   its own neighbourhood. The fix is in ``fwd/hbs/reservoir.py``: depth, centring,
   hyperplanes.
2. **The index is fine but the learned query never enters the data's region.**
   Morgan fingerprints are non-negative and L2-normalized, so they occupy a narrow
   cone; a ``Linear`` projection initialized with Xavier produces a sign-symmetric
   vector that is close to orthogonal to all of them, and SimHash bits are then
   coin flips with respect to the data. The fix is on the query side.

This script separates the two by measuring recall under queries of known quality:

``self``       a data row's own feature vector. Ceiling case: recall must be ~1.0.
``neighbour``  a random convex blend of a row and its true nearest neighbour.
``centroid``   the library's mean direction -- the easiest "real" query.
``xavier``     a Xavier-initialized ``Linear(synthon_dim, dim)`` applied to a random
               unit vector, i.e. exactly the query distribution the policy starts
               from. This is the one to compare against the live 1-2x.

For each it reports recall@budget of the true top-k by cosine, and the rank of the
single best row, against the chance baseline of ``budget / N``.

    .venv-hbs/bin/python -m hbs.diagnose_retrieval --env-dir data/envs/enamine_290k \\
        --depths 8 10 12 16 --budget 256
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from rxnflow.gflownet.fwd.hbs.reservoir import HashedReservoir, normalize_rows

logger = logging.getLogger("hbs.diagnose_retrieval")


def true_top_k(features: NDArray, queries: NDArray, k: int) -> NDArray[np.int64]:
    """Exact top-k rows by cosine, the ground truth the hash is judged against."""
    scores = normalize_rows(queries) @ normalize_rows(features).T
    return np.argsort(-scores, axis=1)[:, :k]


def make_queries(
    kind: str,
    features: NDArray,
    rng: np.random.Generator,
    num: int,
    synthon_dim: int,
) -> NDArray[np.float32]:
    unit = normalize_rows(features)
    if kind == "self":
        rows = rng.choice(len(features), size=num, replace=False)
        return unit[rows].copy()
    if kind == "neighbour":
        rows = rng.choice(len(features), size=num, replace=False)
        top = true_top_k(features, unit[rows], 2)[:, 1]
        blend = rng.uniform(0.5, 0.9, size=(num, 1)).astype(np.float32)
        return normalize_rows(blend * unit[rows] + (1 - blend) * unit[top])
    if kind == "centroid":
        mean = unit.mean(axis=0, keepdims=True)
        jitter = rng.normal(0, 0.05, size=(num, unit.shape[1])).astype(np.float32)
        return normalize_rows(np.repeat(mean, num, axis=0) + jitter)
    if kind == "residual":
        # Candidate fix 1: probe = centroid + a learned residual. Simulated with a
        # residual that points at one specific row, which is the best case a learned
        # residual could reach, to test whether residual-space retrieval works at all.
        rows = rng.choice(len(features), size=num, replace=False)
        mean = unit.mean(axis=0, keepdims=True)
        mean = mean / np.linalg.norm(mean)
        delta = unit[rows] - mean
        delta = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        return normalize_rows(mean + 0.5 * delta)
    if kind == "xavier_resid":
        # Candidate fix 1 as it would actually behave at initialization: centroid
        # plus a *Xavier* residual, i.e. an uninformative residual in the cone.
        projection = nn.Linear(synthon_dim, unit.shape[1], bias=False)
        nn.init.xavier_uniform_(projection.weight)
        source = torch.randn(num, synthon_dim)
        source = source / source.norm(dim=1, keepdim=True)
        with torch.no_grad():
            delta = projection(source).numpy()
        delta = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
        mean = unit.mean(axis=0, keepdims=True)
        mean = mean / np.linalg.norm(mean)
        return normalize_rows(mean + 0.5 * delta)
    if kind == "xavier":
        # The policy's query at initialization: Xavier Linear(synthon_dim -> dim)
        # applied to a unit-norm state/reaction embedding, as fwd/trainer.py builds it.
        projection = nn.Linear(synthon_dim, unit.shape[1], bias=False)
        nn.init.xavier_uniform_(projection.weight)
        source = torch.randn(num, synthon_dim)
        source = source / source.norm(dim=1, keepdim=True)
        with torch.no_grad():
            out = projection(source).numpy()
        return normalize_rows(out)
    raise ValueError(f"unknown query kind: {kind}")


def evaluate(
    reservoir: HashedReservoir,
    library: str,
    features: NDArray,
    queries: NDArray,
    budget: int,
    k: int,
) -> dict[str, float]:
    truth = true_top_k(features, queries, k)
    opened = reservoir.lookup(library, queries.astype(np.float32), budget)
    recalls, best_hits, scanned = [], [], []
    for row_truth, rows in zip(truth, opened, strict=True):
        got = set(rows.tolist())
        recalls.append(len(got & set(row_truth.tolist())) / len(row_truth))
        best_hits.append(1.0 if row_truth[0] in got else 0.0)
        scanned.append(len(got))
    mean_scanned = float(np.mean(scanned))
    chance = mean_scanned / len(features)
    return {
        "recall@k": float(np.mean(recalls)),
        "best_found": float(np.mean(best_hits)),
        "scanned": mean_scanned,
        "chance": chance,
        # The only number that matters: how many times better than scanning the same
        # count at random. 1.0 means the hash contributed nothing.
        "gain": float(np.mean(recalls)) / chance if chance > 0 else float("nan"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=Path("data/envs/enamine_290k"))
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[8, 10, 12, 16],
        help="code lengths to compare; the runs used 16",
    )
    parser.add_argument("--budget", type=int, default=256, help="rows per lookup (min_rows)")
    parser.add_argument("--top-k", type=int, default=10, help="true neighbours to recall")
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--synthon-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--library", help="library to test; default is the largest, as the live probe uses"
    )
    parser.add_argument(
        "--kinds", nargs="+",
        default=["self", "neighbour", "centroid", "xavier"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rxnflow.envs.env import SynthesisEnv

    env = SynthesisEnv(str(args.env_dir))
    features = {name: lib.fingerprints for name, lib in env.synthons.items()}
    library = args.library or max(features, key=lambda n: len(features[n]))
    values = np.asarray(features[library])
    print(
        f"library {library}: {len(values):,} rows, {values.shape[1]} features; "
        f"budget {args.budget} rows ({args.budget / len(values):.2%} of the library), "
        f"recall of the true top-{args.top_k} by cosine"
    )
    unit = normalize_rows(values)
    pairwise = unit[:512] @ unit[:512].T
    off = pairwise[~np.eye(len(pairwise), dtype=bool)]
    print(
        f"feature geometry: mean pairwise cosine {off.mean():+.3f} "
        f"(min {off.min():+.3f}, max {off.max():+.3f}); "
        f"mean cosine to centroid {(unit[:512] @ unit.mean(0) / np.linalg.norm(unit.mean(0))).mean():+.3f}"
    )
    print()

    rng = np.random.default_rng(args.seed)
    prepared = {
        kind: make_queries(kind, values, np.random.default_rng(args.seed), args.queries, args.synthon_dim)
        for kind in args.kinds
    }
    for kind, queries in prepared.items():
        cos = (normalize_rows(queries) @ unit.T)
        print(
            f"{kind:10s} query-to-data cosine: mean {cos.mean():+.4f}, "
            f"best {cos.max(axis=1).mean():+.4f}"
        )
    print()

    header = f"{'depth':>6s} {'center':>7s} {'query':>10s} {'recall@k':>9s} {'best':>7s} {'scanned':>8s} {'chance':>8s} {'gain':>7s}"
    print(header)
    print("-" * len(header))
    for depth in args.depths:
        for center in (False, True):
            reservoir = HashedReservoir(
                features, depth, seed=args.seed, center=center
            )
            for kind, queries in prepared.items():
                r = evaluate(
                    reservoir, library, values, queries, args.budget, args.top_k
                )
                flag = "yes" if center else "no"
                print(
                    f"{depth:6d} {flag:>7s} {kind:>10s} {r['recall@k']:9.3f} "
                    f"{r['best_found']:7.3f} {r['scanned']:8.0f} {r['chance']:8.4f} "
                    f"{r['gain']:6.1f}x"
                )
        print()


if __name__ == "__main__":
    main()
