"""Compare the trained `P_B` of several runs on one shared panel of molecules.

Each arm's backward policy is loaded from its run checkpoint and scored on the *same*
held-out products, so the rows are directly comparable. Probabilities are exact: the
one-step candidate tree is enumerated rather than sampled, so there is no Monte-Carlo
noise and `solve@k = 1 - (1 - p)^k` follows in closed form.

Rows:
  native     RxnFlow's recursive search (what `backward.mode: heuristic` actually uses);
             solved when any returned route replays forward to the product
  uniform    the floor: uniform over the same one-step candidates, no network
  <arm>      that run's trained P_B

Columns:
  p_replay   probability mass on routes that replay forward to the product -- the
             per-molecule reach rate, and `1 - p_replay` is the leakage factor
  solve@k    1 - (1 - p_replay)^k, the chance of a route within k attempts
  nll        negative log-likelihood of the edges the forward policy actually generated
             (uniform's nll on the same edges is the reference)
  synthons   mean synthon count of the replay-verified routes, weighted by probability

Usage:
  python scripts/compare_backward_policies.py \
      mle=runs/mle-1200-s1 reinforce=runs/reinforce-1200-s1 \
      --panel-from runs/heuristic-1200-s1 --env-dir data/envs/enamine_290k
"""

from __future__ import annotations

import argparse
import json
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
    row_key,
    timed,
    uniform_probs,
)
from IFT6760.models import RxnFlowModel
from IFT6760.policies.backward import BackwardPolicy
from IFT6760.training.config import Config


def load_policy(run: Path, env: SynthesisEnv, device: torch.device) -> BackwardPolicy:
    """Rebuild a run's backward policy from its checkpoint."""
    path = run / "checkpoints" / "latest.ckpt"
    if not path.exists():  # fall back to the newest numbered checkpoint
        path = max((run / "checkpoints").glob("step_*.ckpt"))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = Config.from_file(run / "config.yaml")
    model = RxnFlowModel(
        env,
        config.model,
        len(checkpoint["objectives"]),
        checkpoint["condition_dim"],
        config.backward,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return BackwardPolicy(env, model, config, device)


@torch.no_grad()
def edge_nll(policy: BackwardPolicy, decisions, observed: torch.Tensor) -> float:
    """Mean -log P_B of the generated edges, over the panel's decisions."""
    logp, seg = policy.log_probs(decisions)
    return float(-seg.pick(logp, observed.to(policy.device)).mean())


def summarise(rows: list[dict], attempts: int) -> dict:
    p = np.array([r["p_replay"] for r in rows])
    return {
        "products": len(rows),
        "p_replay": float(p.mean()),
        "p_replay_min": float(p.min()),
        "below_1": int((p < 1 - 1e-9).sum()),
        "zero": int((p < 1e-9).sum()),
        f"solve@{attempts}": float((1 - (1 - p) ** attempts).mean()),
        "synthons": nanmean([r["synthons"] for r in rows]),
        "reactions": nanmean([r["reactions"] for r in rows]),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("arms", nargs="*", help="name=run_dir for each learned arm")
    ap.add_argument(
        "--panel-from", type=Path, required=True, help="run supplying products"
    )
    ap.add_argument("--env-dir", type=Path, default=Path("data/envs/enamine_290k"))
    ap.add_argument("--panel", type=int, default=200, help="held-out products")
    ap.add_argument("--panel-steps", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    env = SynthesisEnv(args.env_dir, retrosynthesis_workers=args.workers)
    device = torch.device(args.device)

    rows = load_rows(
        args.panel_from / "sample.jsonl",
        *args.panel_steps,
        args.panel,
        args.seed,
    )
    keys = [row_key(r) for r in rows]
    products = [k[0] for k in keys]
    print(
        f"panel: {len(products)} products from {args.panel_from} "
        f"steps {args.panel_steps[0]}-{args.panel_steps[1]}",
        flush=True,
    )

    # Candidates do not depend on the policy, so the tree and its routes are built once.
    tree, tree_s, _ = timed(lambda: build_tree(env, env.retro_analyzer, keys))
    cache: dict = {}
    routes = {k: enumerate_routes(env, tree, k, cache) for k in keys}
    print(f"tree: {len(tree)} states in {tree_s:.1f}s", flush=True)
    decisions, observed = observed_decisions(env, rows)

    results: dict[str, dict] = {}

    native, native_s, _ = timed(lambda: evaluate_native(env, products))
    results["native"] = {
        "products": len(native),
        "p_replay": float(np.mean([r["solved"] for r in native])),
        "p_replay_min": float(min(r["solved"] for r in native)),
        "below_1": int(sum(not r["solved"] for r in native)),
        "zero": int(sum(not r["solved"] for r in native)),
        f"solve@{args.attempts}": float(np.mean([r["solved"] for r in native])),
        "synthons": nanmean([r["synthons"] for r in native]),
        "reactions": nanmean([r["reactions"] for r in native]),
        "nll": None,
        "seconds": round(native_s, 1),
    }

    scored, uniform_s, _ = timed(lambda: evaluate_probs(routes, uniform_probs(tree)))
    counts = np.array([len(tree[k][0]) for k in tree if tree[k][0]])
    results["uniform"] = {
        **summarise(scored, args.attempts),
        "nll": float(np.log(counts).mean()),
        "seconds": round(uniform_s, 1),
    }

    for item in args.arms:
        name, _, run = item.partition("=")
        policy = load_policy(Path(run), env, device)
        scored, seconds, _ = timed(
            lambda p=policy: evaluate_probs(routes, policy_probs(p, tree))
        )
        results[name] = {
            **summarise(scored, args.attempts),
            "nll": edge_nll(policy, decisions, observed),
            "seconds": round(seconds, 1),
        }
        policy = None  # free the GPU copy before loading the next arm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns = [
        "p_replay",
        f"solve@{args.attempts}",
        "nll",
        "below_1",
        "zero",
        "synthons",
        "reactions",
        "seconds",
    ]
    print("\n| arm | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name, row in results.items():
        cells = []
        for key in columns:
            v = row.get(key)
            cells.append(
                "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
            )
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(
        f"\npanel of {len(products)} products; probabilities are exact "
        f"(enumerated, not sampled). `below_1` counts products the policy cannot "
        f"always solve; `zero` counts those it never solves."
    )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    env.close()


if __name__ == "__main__":
    main()
