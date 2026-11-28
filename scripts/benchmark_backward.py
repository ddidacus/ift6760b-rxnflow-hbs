"""backward benchmark: native RxnFlow search(heuristic) vs. uniform / MLE / REINFORCE P_B.


Rows
  native    blind `RetroSynthesisAnalyzer.run` (no known route); solved if any returned route
            replays
  others    one-step candidates with a uniform or trained P_B. solve@k is exact:
            mean over products of 1 - (1 - p)^k, p = replay-verified route probability.

`--time` adds a wall-clock pass:seconds per solved product

Usage:
  python scripts/benchmark_backward.py --samples ~/runs/ref/qed-200/sample.jsonl [--time]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from IFT6760.envs.env import SynthesisEnv
from IFT6760.evaluation.backward import (
    build_tree,
    enumerate_routes,
    evaluate_native,
    evaluate_probs,
    load_rows,
    nanmean,
    observed_decisions,
    policy_probs,
    sample_uniform_routes,
    solved_by_replay,
    timed,
    train_head,
    uniform_probs,
)
from IFT6760.models import RxnFlowModel
from IFT6760.policies.backward import BackwardPolicy
from IFT6760.training.config import BackwardConfig, Config

ARMS = {
    "uniform": "none",
    "mle": "mle",
    "reinforce": "reinforce",
}


def make_policy(env, env_dir, objective, seed):
    torch.manual_seed(seed)
    backward = BackwardConfig(
        mode="learned",
        objective=objective,
        entropy_weight=1.0,
        detach_shared=True,
    )
    config = Config(env_dir=str(env_dir), backward=backward)
    model = RxnFlowModel(env, config.model, 1, None, backward)
    return BackwardPolicy(env, model, config, torch.device("cpu")), backward


def reset_caches(env, policy=None):
    env.retro_analyzer._one_step_cache.clear()
    if policy is not None:
        policy._graphs.clear()


def time_pass(env, products, attempts, seed, method, policy=None):
    """just a wall clock timer"""
    reset_caches(env, policy)
    starts = [p for p in products for _ in range(attempts)]
    analyzer = env.retro_analyzer
    if method == "native":
        _, seconds, cpu = timed(lambda: [analyzer.run(p) for p in products])
        return {"seconds": seconds, "cpu_seconds": cpu, "solved_products": None}
    if method == "uniform":
        rng = random.Random(seed)
        routes, seconds, cpu = timed(
            lambda: sample_uniform_routes(env, analyzer, starts, rng)
        )
    else:
        torch.manual_seed(seed)
        with torch.no_grad():
            rollouts, seconds, cpu = timed(lambda: policy.rollout(starts))
        routes = rollouts.routes
    solved = solved_by_replay(env, starts, routes)
    per_product = np.array(solved).reshape(len(products), attempts)
    return {
        "seconds": seconds,
        "cpu_seconds": cpu,
        "attempt_success": float(per_product.mean()),
        "solved_products": int(per_product.any(axis=1).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--env-dir", type=Path, default=Path("data/envs/example_5k"))
    ap.add_argument("--arms", default="uniform,mle,reinforce")
    ap.add_argument("--train", type=int, default=256, help="training molecules")
    ap.add_argument("--panel", type=int, default=300, help="held-out products")
    ap.add_argument("--train-steps", type=int, nargs=2, default=[150, 200])
    ap.add_argument("--panel-steps", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument(
        "--time", action="store_true", help="add the wall-clock pass (idle machine!)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    env = SynthesisEnv(args.env_dir, retrosynthesis_workers=0)
    analyzer = env.retro_analyzer
    train_rows = load_rows(args.samples, *args.train_steps, args.train, args.seed)
    train_finals = [r["final_smiles"] for r in train_rows]
    panel_rows = load_rows(
        args.samples, *args.panel_steps, args.panel, args.seed, exclude=train_finals
    )
    panel = [r["final_smiles"] for r in panel_rows]
    print(
        f"train {len(train_finals)} molecules, held-out panel {len(panel)} products",
        flush=True,
    )

    # Candidate graph and every complete route, replayed once (shared by all policies).
    everything = panel + train_finals
    tree = build_tree(env, analyzer, everything)
    cache: dict = {}
    routes = {k: enumerate_routes(env, tree, k, cache) for k in dict.fromkeys(everything)}
    flat = [r for k in panel for r in routes[k]]
    fails = Counter(r.reason for r in flat if r.count_ok and not r.replay_ok)
    print(
        f"routes reaching the empty state per panel product: {len(flat) / len(panel):.1f}; "
        f"within the count limits {sum(r.count_ok for r in flat)} / {len(flat)}; "
        f"replay ok {sum(r.replay_ok for r in flat)}; replay failures among count-ok routes: {dict(fails)}",
        flush=True,
    )

    rows = {}
    native = evaluate_native(env, panel)
    rows["native"] = {
        "solve@1": float(np.mean([r["solved"] for r in native])),
        "solve@8": float(np.mean([r["solved"] for r in native])),
        "count_solve@1": float(np.mean([r["n_routes"] > 0 for r in native])),
        "routes_replay_ok": float(
            sum(r["n_replay_ok"] for r in native)
            / max(sum(r["n_routes"] for r in native), 1)
        ),
        "per_product": native,
    }
    print(
        f"native: solved {rows['native']['solve@1']:.3f}, returned routes that replay "
        f"{rows['native']['routes_replay_ok']:.3f}",
        flush=True,
    )

    policies = {}
    for name in args.arms.split(","):
        policy, backward = make_policy(env, args.env_dir, ARMS[name], args.seed)
        train_seconds = 0.0
        if name != "uniform":
            decisions, observed = observed_decisions(env, train_rows)
            train_seconds = train_head(
                policy,
                env,
                backward,
                decisions,
                observed,
                train_finals,
                iters=args.iters,
                lr=args.lr,
                seed=args.seed,
            )
        policies[name] = policy
        probs = uniform_probs(tree) if name == "uniform" else policy_probs(policy, tree)
        held = evaluate_probs({s: routes[s] for s in panel}, probs)
        seen = evaluate_probs({s: routes[s] for s in train_finals}, probs)
        rows[name] = {
            "solve@1": float(np.mean([r["p_replay"] for r in held])),
            "solve@8": float(
                np.mean([1 - (1 - r["p_replay"]) ** args.attempts for r in held])
            ),
            "count_solve@1": float(np.mean([r["p_count"] for r in held])),
            "train_solve@1": float(np.mean([r["p_replay"] for r in seen])),
            "train_seconds": train_seconds,
            "per_product": held,
        }
        print(
            f"{name}: solve@1 {rows[name]['solve@1']:.3f} solve@8 {rows[name]['solve@8']:.3f} "
            f"(count-limit solve@1 {rows[name]['count_solve@1']:.3f}), train {train_seconds:.0f}s",
            flush=True,
        )

    # Route properties on the products every method solves.
    common = [
        i
        for i in range(len(panel))
        if rows["native"]["per_product"][i]["solved"]
        and all(rows[a]["per_product"][i]["p_replay"] > 0 for a in args.arms.split(","))
    ]
    for row in rows.values():
        per = row["per_product"]
        row["common_synthons"] = nanmean([per[i]["synthons"] for i in common])
        row["common_reactions"] = nanmean([per[i]["reactions"] for i in common])
    print(
        f"route properties over the {len(common)} products solved by every method",
        flush=True,
    )

    if args.time:
        for name in ["native", *args.arms.split(",")]:
            policy = policies.get(name)
            row = rows[name]
            for attempts in (1,) if name == "native" else (1, args.attempts):
                t = time_pass(
                    env,
                    panel,
                    attempts,
                    args.seed,
                    name if name in ("native", "uniform") else "policy",
                    policy,
                )
                solved = (
                    t["solved_products"]
                    if t["solved_products"] is not None
                    else round(row["solve@1"] * len(panel))
                )
                row[f"seconds_{attempts}"] = t["seconds"]
                row[f"sec_per_solved_{attempts}"] = t["seconds"] / max(solved, 1)
                row[f"cpu_sec_per_solved_{attempts}"] = t["cpu_seconds"] / max(solved, 1)
                row[f"empirical_solved_products_{attempts}"] = t["solved_products"]

    env.close()
    columns = [
        ("solve@1", "{:.3f}"),
        ("solve@8", "{:.3f}"),
        ("count_solve@1", "{:.3f}"),
        ("train_solve@1", "{:.3f}"),
        ("common_synthons", "{:.2f}"),
        ("common_reactions", "{:.2f}"),
        ("train_seconds", "{:.0f}"),
    ]
    if args.time:
        columns += [
            ("sec_per_solved_1", "{:.4f}"),
            (f"sec_per_solved_{args.attempts}", "{:.4f}"),
            ("cpu_sec_per_solved_1", "{:.4f}"),
            (f"cpu_sec_per_solved_{args.attempts}", "{:.4f}"),
        ]
    print("\n| method | " + " | ".join(c for c, _ in columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name, row in rows.items():
        cells = [fmt.format(row[c]) if c in row else "-" for c, fmt in columns]
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.out:
        slim = {
            n: {k: v for k, v in r.items() if k != "per_product"} for n, r in rows.items()
        }
        args.out.write_text(
            json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "results": slim},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
