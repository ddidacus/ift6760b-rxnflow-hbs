"""Train P_B on fixed molecules and report its exact backward success probability.

The forward policy plays no role here: molecules (and, for MLE, the generated edges) come
from a run's sample.jsonl. P_B is trained with the objective under test, then evaluated by
exact enumeration of the one-step candidate tree (no sampling noise):

  p_succ  = probability that a backward walk from the molecule reaches the empty state
            within the forward limits (1.0 means P_B never enters a dead end),
  nll     = negative log-likelihood of the edges the forward policy generated.

`train` molecules come from the late steps of the run, `test` molecules from earlier steps
(different molecules), the analogue of SynFlowNet's train/test "solved routes".

Usage:
  python scripts/probe_backward_policy.py --samples ~/runs/ref/qed-200/sample.jsonl \
      --objective reinforce --iters 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from IFT6760.envs.env import SynthesisEnv
from IFT6760.models import RxnFlowModel
from IFT6760.objectives.backward import reinforce_loss
from IFT6760.policies.backward import BackwardPolicy, Decisions
from IFT6760.training.config import BackwardConfig, Config


def edge_key(env, edge):
    action, parent = edge
    synthon = None
    if action.library_name is not None:
        synthon = env.synthons[action.library_name].smiles[action.synthon_index]
    return (action.action_type.name, action.reaction, synthon, parent)


def observed_key(step):
    return (
        step["action_type"],
        step["reaction"],
        step.get("synthon_smiles"),
        step["state"],
    )


def load_rows(path, lo, hi, limit, seed):
    rows, seen = [], set()
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if (
                lo <= row["step"] <= hi
                and row["valid"]
                and row["final_smiles"] not in seen
            ):
                seen.add(row["final_smiles"])
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def observed_decisions(env, rows):
    """Decisions (product state, candidates) and the index of the generated edge."""
    steps = [
        (row, step) for row in rows for step in row["traj"] if step["product_smiles"]
    ]
    candidates = env.retro_analyzer.one_step_many([s["product_smiles"] for _, s in steps])
    smiles, edges, index = [], [], []
    for (_, step), options in zip(steps, candidates, strict=True):
        keys = [edge_key(env, e) for e in options]
        if observed_key(step) not in keys:
            continue  # not regenerated
        smiles.append(step["product_smiles"])
        edges.append(options)
        index.append(keys.index(observed_key(step)))
    return Decisions(smiles, edges), torch.tensor(index)


@torch.no_grad()
def exact_success(policy, env, starts, chunk=400):
    """Mean probability that a P_B walk from each start reaches the empty state in the limits."""
    analyzer = policy.analyzer
    tree, level = {}, {(s, 0, 0) for s in starts}
    for _ in range(env.max_reactions + 1):
        nxt = set()
        keys = [k for k in level if k not in tree]
        for key, edges in zip(
            keys, analyzer.one_step_many([k[0] for k in keys]), strict=True
        ):
            children = []
            for action, parent in edges:
                r = key[1] + (0 if action.action_type.is_first else 1)
                k = key[2] + (0 if action.action_type.is_unirxn else 1)
                if r > env.max_reactions or k > env.max_synthons:
                    children.append(0.0)
                elif parent == "":
                    children.append(
                        float(r >= env.min_reactions and k >= env.min_synthons)
                    )
                else:
                    children.append((parent, r, k))
                    nxt.add((parent, r, k))
            tree[key] = (edges, children)
        level = nxt
    expandable = [k for k, (edges, _) in tree.items() if edges]
    prob = {}
    for start in range(0, len(expandable), chunk):
        part = expandable[start : start + chunk]
        decisions = Decisions([k[0] for k in part], [tree[k][0] for k in part])
        logp, seg = policy.log_probs(decisions)
        for key, row in zip(part, logp.exp().tolist(), strict=True):
            prob[key] = row[: len(tree[key][0])]
    memo = {}

    def success(key):
        if key in memo:
            return memo[key]
        edges, children = tree[key]
        value = 0.0
        for p, child in zip(prob.get(key, []), children, strict=False):
            value += p * (success(child) if isinstance(child, tuple) else child)
        memo[key] = value
        return value

    return float(np.mean([success((s, 0, 0)) for s in starts]))


@torch.no_grad()
def mean_nll(policy, decisions, observed):
    logp, seg = policy.log_probs(decisions)
    return float(-seg.pick(logp, observed).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--env-dir", type=Path, default=Path("data/envs/example_5k"))
    ap.add_argument(
        "--objective",
        choices=["none", "mle", "reinforce"],
        default="reinforce",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entropy-weight", type=float, default=1.0)
    ap.add_argument(
        "--train-encoders",
        action="store_true",
        help="let the P_B loss update the shared encoders",
    )
    ap.add_argument("--train", type=int, default=128, help="training molecules")
    ap.add_argument("--test", type=int, default=128, help="held-out molecules")
    ap.add_argument("--batch", type=int, default=64, help="molecules per iteration")
    ap.add_argument("--train-steps", type=int, nargs=2, default=[150, 200])
    ap.add_argument("--test-steps", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    backward = BackwardConfig(
        mode="learned",
        objective=args.objective,
        entropy_weight=args.entropy_weight,
        detach_shared=not args.train_encoders,
    )
    config = Config(env_dir=str(args.env_dir), backward=backward)
    env = SynthesisEnv(args.env_dir, retrosynthesis_workers=args.workers)
    model = RxnFlowModel(env, config.model, 1, None, backward)
    policy = BackwardPolicy(env, model, config, torch.device("cpu"))

    train_rows = load_rows(args.samples, *args.train_steps, args.train, args.seed)
    test_rows = load_rows(args.samples, *args.test_steps, args.test, args.seed)
    train_finals = [r["final_smiles"] for r in train_rows]
    test_finals = [r["final_smiles"] for r in test_rows]
    train_dec, train_obs = observed_decisions(env, train_rows)
    test_dec, test_obs = observed_decisions(env, test_rows)

    params = (
        list(model.parameters())
        if args.train_encoders
        else list(model.backward_head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = random.Random(args.seed)
    history = []

    def evaluate(it):
        row = {
            "iter": it,
            "train_p_succ": exact_success(policy, env, train_finals),
            "test_p_succ": exact_success(policy, env, test_finals),
            "train_nll": mean_nll(policy, train_dec, train_obs),
            "test_nll": mean_nll(policy, test_dec, test_obs),
        }
        history.append(row)
        print(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
            flush=True,
        )

    print(
        f"objective={args.objective} train={len(train_finals)} test={len(test_finals)} "
        f"decisions train/test={len(train_dec)}/{len(test_dec)}",
        flush=True,
    )
    evaluate(0)
    start = time.perf_counter()
    for it in range(1, args.iters + 1):
        optimizer.zero_grad()
        loss = torch.zeros(())
        if backward.uses_mle:
            pick = torch.tensor(
                rng.sample(range(len(train_dec)), min(len(train_dec), 4 * args.batch))
            )
            batch = Decisions(
                [train_dec.smiles[i] for i in pick], [train_dec.edges[i] for i in pick]
            )
            logp, seg = policy.log_probs(batch)
            loss = loss + backward.mle_weight * -seg.pick(logp, train_obs[pick]).mean()
        if backward.uses_reinforce:
            starts = rng.sample(train_finals, min(len(train_finals), args.batch))
            rf, _ = reinforce_loss(backward, policy.rollout(starts))
            loss = loss + backward.reinforce_weight * rf
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        if it % args.eval_every == 0 or it == args.iters:
            evaluate(it)
    print(f"trained {args.iters} iterations in {time.perf_counter() - start:.0f}s")
    env.close()
    if args.out:
        args.out.write_text(
            json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "history": history},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
