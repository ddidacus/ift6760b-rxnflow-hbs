"""benchmark helpers for P_B.

A backward walk "solves" a product when it reaches the empty state *and its route replays*:
applying the edges forward through `SynthesisEnv.step`, starting from the empty state, must
be accepted by the environment (action available for the state's counters, reaction succeeds)
and end in the same terminal molecule. That is stricter than checking reaction/synthon counts.

For any policy that gives a probability to every candidate of a state, the distribution over
complete routes can be enumerated exactly (candidate graphs are tiny), so solve rates need no
sampling noise: solve@k of a product is 1 - (1 - p)^k with p the replay-verified probability.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from IFT6760.core.errors import InvalidTransition
from IFT6760.core.types import ActionType, BackwardEdge
from IFT6760.envs.env import SynthesisEnv
from IFT6760.objectives.backward import reinforce_loss
from IFT6760.policies.backward import BackwardPolicy, Decisions
from IFT6760.training.config import BackwardConfig

Key = tuple[str, int, int]  # note: (state SMILES, reactions used, synthons used)


def replay(
    env: SynthesisEnv, route: tuple[BackwardEdge, ...], product: str
) -> tuple[bool, str]:
    """apply `route` (product first, FirstSynthon last) forward from the empty state.

    Returns (ok, reason). ok means every action was accepted by the environment and the
    walk ends in the terminal molecule `product`.
    """
    state = env.initial_state()
    for action, _ in reversed(route):
        try:
            state = env.step(state, action)
        except InvalidTransition as error:
            return False, str(error)
    if not state.terminated:
        return False, "not terminal"
    if state.smiles != product:
        return False, "different product"
    return True, ""


def counts_of(action_types) -> tuple[int, int]:
    """(reactions, synthons) spent by a sequence of action types."""
    costs = [t.cost for t in action_types]
    return sum(c[0] for c in costs), sum(c[1] for c in costs)


def trajectory_key(trajectory) -> Key:
    """`(final molecule, reactions, synthons)` of a generated trajectory."""
    return (
        trajectory.final_smiles,
        *counts_of(step.action.action_type for step in trajectory.steps),
    )


def row_key(row: dict) -> Key:
    """The same, from a `sample.jsonl` row."""
    return (
        row["final_smiles"],
        *counts_of(ActionType[step["action_type"]] for step in row["traj"]),
    )


def route_counts(route: tuple[BackwardEdge, ...]) -> tuple[int, int]:
    """(reactions, synthons) used by a route."""
    costs = [action.action_type.cost for action, _ in route]
    return sum(c[0] for c in costs), sum(c[1] for c in costs)


Tree = dict[Key, tuple[tuple[BackwardEdge, ...], list[tuple]]]


def build_tree(env: SynthesisEnv, analyzer, starts: list[Key]) -> Tree:
    """One-step candidate graph from each start, on the same rule the sampler uses.

    A key is `(smiles, reactions, synthons)` with the counters *still to be undone*, so
    an edge appears only when `SynthesisEnv.parent_counts` admits it -- exactly what
    `BackwardPolicy.allowed` applies. Reaching the empty state with both counters at zero
    is a complete route; there is no separate legality test and no dead branch to carry.
    """
    tree: Tree = {}
    level = set(starts)
    for _ in range(env.max_reactions + 1):
        nxt: set[Key] = set()
        keys = [k for k in level if k not in tree]
        for key, found in zip(
            keys, analyzer.one_step_many([k[0] for k in keys]), strict=True
        ):
            edges = tuple(
                e for e in found if env.parent_counts(e[0], key[1], key[2]) is not None
            )
            children = []
            for action, parent in edges:
                counts = env.parent_counts(action, key[1], key[2])
                if parent == "":
                    children.append(("root", counts == (0, 0)))
                else:
                    children.append(("node", (parent, *counts)))
                    nxt.add((parent, *counts))
            tree[key] = (edges, children)
        level = nxt
    return tree


@dataclass
class Route:
    steps: tuple[tuple[Key, int], ...]  # (node, edge index) along the walk
    edges: tuple[BackwardEdge, ...]
    count_ok: (
        bool  # reaches the empty state within the min/max reaction and synthon counts
    )
    replay_ok: bool
    reason: str
    reactions: int
    synthons: int


def enumerate_routes(
    env: SynthesisEnv, tree: Tree, start: Key, cache: dict
) -> list[Route]:
    """Every complete backward route from `start` (those that reach the empty state)."""
    found: list[Route] = []
    product = start[0]

    def walk(key: Key, path: tuple) -> None:
        edges, children = tree[key]
        for index, child in enumerate(children):
            steps = (*path, (key, index))
            if child[0] == "root":
                route = tuple(tree[k][0][i] for k, i in steps)
                if (product, route) not in cache:
                    cache[product, route] = replay(env, route, product)
                ok, reason = cache[product, route]
                reactions, synthons = route_counts(route)
                found.append(
                    Route(steps, route, child[1], ok, reason, reactions, synthons)
                )
            elif child[0] == "node" and child[1] in tree:
                walk(child[1], steps)

    walk(start, ())
    return found


def uniform_probs(tree: Tree) -> dict[Key, list[float]]:
    return {k: [1.0 / len(edges)] * len(edges) for k, (edges, _) in tree.items() if edges}


@torch.no_grad()
def policy_probs(
    policy: BackwardPolicy, tree: Tree, chunk: int = 400
) -> dict[Key, list[float]]:
    keys = [k for k, (edges, _) in tree.items() if edges]
    probs: dict[Key, list[float]] = {}
    for start in range(0, len(keys), chunk):
        part = keys[start : start + chunk]
        logp, _ = policy.log_probs(
            Decisions([k[0] for k in part], [tree[k][0] for k in part])
        )
        for key, row in zip(part, logp.exp().tolist(), strict=True):
            probs[key] = row[: len(tree[key][0])]
    return probs


def evaluate_probs(
    routes: dict[str, list[Route]], probs: dict[Key, list[float]]
) -> list[dict]:
    results = []
    for start, options in routes.items():
        p_count = p_replay = syn = rxn = 0.0
        for route in options:
            p = 1.0
            for key, index in route.steps:
                p *= probs[key][index]
            p_count += p * route.count_ok
            if route.replay_ok:
                p_replay += p
                syn += p * route.synthons
                rxn += p * route.reactions
        results.append(
            {
                "product": start,
                "p_count": p_count,
                "p_replay": p_replay,
                "synthons": syn / p_replay if p_replay > 0 else float("nan"),
                "reactions": rxn / p_replay if p_replay > 0 else float("nan"),
            }
        )
    return results


def evaluate_native(env: SynthesisEnv, products: list[str]) -> list[dict]:
    """Blind native search (no known route) with every returned route replayed."""
    analyzer = env.retro_analyzer
    results = []
    for product in products:
        routes = [tuple(r) for r in analyzer.run(product)]
        checks = [replay(env, r, product) for r in routes]
        good = [r for r, (ok, _) in zip(routes, checks, strict=True) if ok]
        best = min(
            good, key=lambda r: (route_counts(r)[1], route_counts(r)[0]), default=None
        )
        results.append(
            {
                "product": product,
                "n_routes": len(routes),
                "n_replay_ok": len(good),
                "solved": bool(good),
                "synthons": route_counts(best)[1] if best else float("nan"),
                "reactions": route_counts(best)[0] if best else float("nan"),
            }
        )
    return results


def sample_uniform_routes(
    env: SynthesisEnv, analyzer, starts: list[Key], rng: random.Random
):
    """Backward walks with a uniform P_B over the admissible candidates (no network)."""
    n = len(starts)
    routes: list[list[BackwardEdge]] = [[] for _ in range(n)]
    current = [key[0] for key in starts]
    counts = [(key[1], key[2]) for key in starts]
    active = list(range(n))
    for _ in range(env.max_reactions + 1):
        options = analyzer.one_step_many([current[i] for i in active])
        nxt = []
        for i, found in zip(active, options, strict=True):
            edges = [e for e in found if env.parent_counts(e[0], *counts[i]) is not None]
            if not edges:
                continue
            edge = rng.choice(edges)
            routes[i].append(edge)
            action, parent = edge
            counts[i] = env.parent_counts(action, *counts[i])
            if parent == "":
                continue
            current[i] = parent
            nxt.append(i)
        active = nxt
    return routes


def solved_by_replay(env: SynthesisEnv, starts: list[str], routes) -> list[bool]:
    """A walk is solved if it reached the empty state and its route replays."""
    return [
        bool(route) and route[-1][1] == "" and replay(env, tuple(route), start)[0]
        for start, route in zip(starts, routes, strict=True)
    ]


def timed(fn):
    """(value, wall seconds, process CPU seconds)."""
    wall, cpu = time.perf_counter(), time.process_time()
    value = fn()
    return value, time.perf_counter() - wall, time.process_time() - cpu


def edge_key(env: SynthesisEnv, edge: BackwardEdge) -> tuple:
    """Identify an edge without the (unlogged) library index: type, reaction, synthon, parent."""
    action, parent = edge
    synthon = None
    if action.library_name is not None:
        synthon = env.synthons[action.library_name].smiles[action.synthon_index]
    return (action.action_type.name, action.reaction, synthon, parent)


def observed_key(step: dict) -> tuple:
    return (
        step["action_type"],
        step["reaction"],
        step.get("synthon_smiles"),
        step["state"],
    )


def load_rows(path, lo: int, hi: int, limit: int, seed: int, exclude=()) -> list[dict]:
    """Distinct valid final molecules from a run's sample.jsonl within a step range."""
    rows, seen = [], set(exclude)
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


def observed_decisions(env: SynthesisEnv, rows: list[dict]):
    """Decisions (product state, candidates) and the index of the generated edge."""
    steps = [
        (row, step) for row in rows for step in row["traj"] if step["product_smiles"]
    ]
    candidates = env.retro_analyzer.one_step_many([s["product_smiles"] for _, s in steps])
    smiles, edges, index = [], [], []
    for (_, step), options in zip(steps, candidates, strict=True):
        keys = [edge_key(env, e) for e in options]
        if observed_key(step) not in keys:
            continue  # not regenerated (never happens on the reference run)
        smiles.append(step["product_smiles"])
        edges.append(options)
        index.append(keys.index(observed_key(step)))
    return Decisions(smiles, edges), torch.tensor(index)


def train_head(
    policy: BackwardPolicy,
    env: SynthesisEnv,
    cfg: BackwardConfig,
    decisions: Decisions,
    observed: torch.Tensor,
    finals: list[str],
    *,
    iters: int,
    lr: float,
    batch: int = 64,
    seed: int = 0,
    train_encoders: bool = False,
) -> float:
    """Train P_B alone on fixed molecules."""
    model = policy.model
    params = (
        list(model.parameters())
        if train_encoders
        else list(model.backward_head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        loss = torch.zeros(())
        if cfg.uses_mle:
            pick = torch.tensor(
                rng.sample(range(len(decisions)), min(len(decisions), 4 * batch))
            )
            part = Decisions(
                [decisions.smiles[i] for i in pick], [decisions.edges[i] for i in pick]
            )
            logp, seg = policy.log_probs(part)
            loss = loss + cfg.mle_weight * -seg.pick(logp, observed[pick]).mean()
        if cfg.uses_reinforce:
            starts = rng.sample(finals, min(len(finals), batch))
            rf, _ = reinforce_loss(cfg, policy.rollout(starts))
            loss = loss + cfg.reinforce_weight * rf
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def nanmean(values) -> float:
    values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(values)) if values else float("nan")
