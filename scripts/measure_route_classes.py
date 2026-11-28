"""How much room is there for a route *preference*?

A preference over routes (cheaper blocks, higher-yield reactions, fewer steps) can change which
route P_B favours only if one product has routes that differ in something a cost can see. Routes
that merely visit the same reactions and blocks in a different order cost the same under any
additive cost. This script groups every complete, replay-verified backward route of each product
into *classes* and reports how many classes exist per product:

  signature   multiset of (reaction, synthon) over the route's edges (order ignored)
  block set   set of synthons only (reactions ignored)
  shape       (number of synthons, number of reactions)

Usage:
  python scripts/measure_route_classes.py --samples ~/runs/ref/qed-200/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from IFT6760.envs.env import SynthesisEnv
from IFT6760.evaluation.backward import (
    build_tree,
    edge_key,
    enumerate_routes,
    load_rows,
    row_key,
)


def classes(env, route):
    keys = [edge_key(env, e) for e in route.edges]
    signature = tuple(sorted((k[1] or "", k[2] or "") for k in keys))
    blocks = tuple(sorted(k[2] for k in keys if k[2] is not None))
    reactions = tuple(sorted(k[1] for k in keys if k[1] is not None))
    return signature, blocks, (route.synthons, route.reactions), reactions


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--env-dir", type=Path, default=Path("data/envs/example_5k"))
    ap.add_argument("--steps", type=int, nargs=2, default=[1, 200])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    env = SynthesisEnv(args.env_dir, retrosynthesis_workers=0)
    rows = load_rows(args.samples, *args.steps, args.n, args.seed)
    keys = [row_key(r) for r in rows]
    products = [k[0] for k in keys]
    tree = build_tree(env, env.retro_analyzer, keys)
    cache: dict = {}

    per_product = []
    for smiles, row in zip(products, rows, strict=True):
        routes = [
            r
            for r in enumerate_routes(env, tree, smiles, cache)
            if r.count_ok and r.replay_ok
        ]
        if not routes:
            continue
        sig, blk, shp, rxn = zip(*(classes(env, r) for r in routes), strict=True)
        per_product.append(
            {
                "smiles": smiles,
                "n_steps": len(row["traj"]),
                "routes": len(routes),
                "signatures": len(set(sig)),
                "block_sets": len(set(blk)),
                "reaction_sets": len(set(rxn)),
                "shapes": len(set(shp)),
                "synthons": sorted({s[0] for s in shp}),
                "reactions": sorted({s[1] for s in shp}),
            }
        )
    env.close()

    n = len(per_product)

    def share(key, threshold=1):
        return sum(p[key] > threshold for p in per_product) / n

    print(f"{n} products with at least one replay-verified route (of {len(products)})")
    print(
        f"routes per product: mean {np.mean([p['routes'] for p in per_product]):.2f}, "
        f"max {max(p['routes'] for p in per_product)}"
    )
    for key, label in (
        ("signatures", "route signatures (reactions+blocks, order ignored)"),
        ("block_sets", "block sets (reactions ignored)"),
        ("reaction_sets", "reaction multisets (blocks ignored)"),
        ("shapes", "shapes (#synthons, #reactions)"),
    ):
        counts = Counter(p[key] for p in per_product)
        print(
            f"{label}: >1 for {100 * share(key):.1f}% of products; distribution {dict(sorted(counts.items()))}"
        )
    differ_syn = sum(len(p["synthons"]) > 1 for p in per_product) / n
    differ_rxn = sum(len(p["reactions"]) > 1 for p in per_product) / n
    print(
        f"routes of one product differ in synthon count: {100 * differ_syn:.1f}%; in reaction count: {100 * differ_rxn:.1f}%"
    )
    by_len = Counter((p["n_steps"], p["signatures"] > 1) for p in per_product)
    print(
        "by generated trajectory length (steps, has >1 signature): ",
        dict(sorted(by_len.items())),
    )
    if args.out:
        args.out.write_text(json.dumps(per_product, indent=1))


if __name__ == "__main__":
    main()
