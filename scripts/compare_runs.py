"""Compare training runs: reward, TB residual, backward-policy statistics and diversity.

For every run directory (config.yaml, training.jsonl, sample.jsonl) it reports, over the
last `--tail` fraction of steps, the mean reward, the mean |TB error| and the mean
`traj_log_p_B`, the backward statistics of learned runs (rollout success, NLL against a
uniform P_B, entropy), and SynFlowNet's mode count: unique Murcko scaffolds among valid
molecules with reward >= `--threshold`, after `--budget` evaluated molecules.

Generative quality (note 11, Tier 3) is reported as avg top-10/top-100 reward, AUC of the
running top-10, diversity (1 - mean pairwise Tanimoto) and #circles over the top molecules.
These are per-run endpoints, so give several seeds as `name=dirA,dirB,dirC`: the table then
shows the mean over seeds with a 95% percentile bootstrap interval. With two or three seeds
that interval is wide on purpose -- it is the honest width, not a formality.

Usage:
  python scripts/compare_runs.py heuristic=runs/qed mle=runs/qed-mle reinforce=runs/qed-rf
  python scripts/compare_runs.py mle=runs/mle-s1,runs/mle-s2,runs/mle-s3
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles_list: list[str]) -> list:
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            out.append(_MORGAN.GetFingerprint(mol))
    return out


def diversity(fps: list) -> float | None:
    """1 - mean pairwise Tanimoto similarity: higher means more varied."""
    if len(fps) < 2:
        return None
    sims = [
        s
        for i in range(1, len(fps))
        for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    ]
    return 1 - float(np.mean(sims))


def n_circles(fps: list, threshold: float) -> int:
    """Greedy count of molecules no two of which are more similar than `threshold`."""
    chosen: list = []
    for fp in fps:
        if not chosen or max(DataStructs.BulkTanimotoSimilarity(fp, chosen)) < threshold:
            chosen.append(fp)
    return len(chosen)


def scaffold_counts(samples: list[dict], budget: int, top_k: int) -> dict:
    """Unthresholded scaffold counts, matching how Diego reports them.

    `scaff_total` counts distinct Bemis-Murcko scaffolds over every valid molecule
    sampled; `scaff@k` counts them among the best `k`. Neither applies a reward
    threshold, unlike `mode_counts`, so both stay meaningful on a reward whose scale is
    not known in advance -- a 2D similarity reward never approaches 1.
    """
    ranked = best_unique(samples[:budget])
    scaffolds = [scaffold(s) for s, _ in ranked]
    return {
        f"scaff@{top_k}": len({s for s in scaffolds[:top_k] if s}),
        "scaff_total": len({s for s in scaffolds if s}),
    }


def best_unique(samples: list[dict]) -> list[tuple[str, float]]:
    """(smiles, reward) per distinct valid molecule, best first."""
    best: dict[str, float] = {}
    for row in samples:
        if row["valid"]:
            smiles = row["final_smiles"]
            best[smiles] = max(best.get(smiles, row["reward"]), row["reward"])
    return sorted(best.items(), key=lambda item: -item[1])


def auc_top_k(samples: list[dict], k: int) -> float | None:
    """Mean over oracle calls of the running top-k reward (PMO's AUC top-k).

    Integrates the whole run, so it has less variance than a final-step snapshot.
    """
    top: list[float] = []
    seen: set[str] = set()
    total, calls = 0.0, 0
    for row in samples:
        if not row["valid"]:
            continue
        calls += 1
        if row["final_smiles"] not in seen:
            seen.add(row["final_smiles"])
            heapq.heappush(top, row["reward"])
            if len(top) > k:
                heapq.heappop(top)
        total += sum(top) / len(top)
    return total / calls if calls else None


def route_signature(row: dict) -> tuple | None:
    """The synthetic route of one sample, as an order-sensitive tuple of its steps.

    Two molecules share a signature when they were built by the same sequence of
    (action type, reaction) choices from the same starting synthon. This is the route
    analogue of a scaffold: it measures how varied the *synthesis* is, independently of
    how varied the products are.
    """
    traj = row.get("traj")
    if not traj:
        return None
    return tuple(
        (step.get("action_type"), step.get("reaction"), step.get("synthon_smiles"))
        for step in traj
    )


def route_stats(samples: list[dict], smiles_keep: set[str]) -> dict:
    """Route diversity over the samples whose product is in `smiles_keep`.

    `route_div` is distinct routes per distinct product: 1.0 means every molecule was
    reached one way only, higher means the run found several syntheses for the same
    product. `route_uniq` is distinct routes over the sampled routes, the direct
    counterpart of `div@k` for pathways. `n_reactions` is the mean route length in
    reactions, which says whether an arm drifts to shorter or longer syntheses.
    """
    routes: set[tuple] = set()
    products: set[str] = set()
    lengths: list[int] = []
    total = 0
    for row in samples:
        if not row["valid"] or row["final_smiles"] not in smiles_keep:
            continue
        signature = route_signature(row)
        if signature is None:
            continue
        total += 1
        routes.add(signature)
        products.add(row["final_smiles"])
        lengths.append(sum(1 for step in row["traj"] if step.get("reaction")))
    if not total:
        return {}
    return {
        "route_div": len(routes) / len(products),
        "route_uniq": len(routes) / total,
        "n_reactions": float(np.mean(lengths)),
    }


def bootstrap_ci(values: list[float], reps: int = 2000, seed: int = 0):
    """95% percentile bootstrap interval of the mean; None for fewer than two seeds."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scaffold(smiles: str) -> str | None:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except Exception:
        return None


def mode_counts(samples: list[dict], threshold: float, budget: int) -> dict[str, int]:
    """Unique valid molecules and unique scaffolds with reward >= threshold, within `budget`."""
    seen, scaffolds, evaluated, unique_hi = set(), set(), 0, set()
    for row in samples:
        evaluated += 1
        if evaluated > budget:
            break
        if not row["valid"] or row["reward"] < threshold:
            continue
        unique_hi.add(row["final_smiles"])
        if row["final_smiles"] not in seen:
            seen.add(row["final_smiles"])
            found = scaffold(row["final_smiles"])
            if found is not None:
                scaffolds.add(found)
    return {
        "evaluated": min(evaluated, budget),
        "unique_high": len(unique_hi),
        "scaffolds": len(scaffolds),
    }


def summarise(
    path: Path,
    tail: float,
    threshold: float,
    budget: int | None,
    top_k: int,
    circles: float,
    scaff_k: int,
    max_steps: int | None = None,
) -> dict:
    rows = load_jsonl(path / "training.jsonl")
    samples = (
        load_jsonl(path / "sample.jsonl") if (path / "sample.jsonl").exists() else []
    )
    if max_steps is not None:
        # Matched-budget comparison: arms that ran longer are cut to the common
        # horizon, on both the step log and the per-molecule log, so every column
        # below describes the same number of optimizer steps and oracle calls.
        rows = [r for r in rows if r["step"] <= max_steps]
        samples = [s for s in samples if s["step"] <= max_steps]
        if not rows:
            raise SystemExit(f"{path}: no steps at or below --max-steps {max_steps}")
    start = int(len(rows) * (1 - tail))
    last = rows[start:]

    def mean(key):
        values = [r[key] for r in last if key in r]
        return float(np.mean(values)) if values else None

    out = {
        "steps": len(rows),
        "reward": mean("reward"),
        "abs_tb_error": float(np.mean([abs(r["tb_error"]) for r in last])),
        "traj_log_p_B": mean("traj_log_p_B"),
        "step_time_s": mean("time"),
    }
    for key in ("bck_rollout_success", "bck_nll", "bck_uniform_nll", "bck_entropy"):
        out[key] = mean(key)
    if samples:
        out.update(mode_counts(samples, threshold, budget or len(samples)))
        out.update(scaffold_counts(samples, budget or len(samples), scaff_k))
        ranked = best_unique(samples[: budget or len(samples)])
        rewards = [r for _, r in ranked]
        for k in (10, 100):
            out[f"top{k}"] = float(np.mean(rewards[:k])) if rewards else None
        out["auc_top10"] = auc_top_k(samples[: budget or len(samples)], 10)
        top_smiles = [s for s, _ in ranked[:top_k]]
        fps = fingerprints(top_smiles)
        out[f"div@{top_k}"] = diversity(fps)
        out["circles"] = n_circles(fps, circles)
        out.update(route_stats(samples, set(top_smiles)))
    return out


def aggregate(runs: list[dict]) -> dict:
    """Per-key mean over seeds, with a bootstrap interval when there is more than one."""
    out: dict[str, tuple] = {}
    for key in runs[0]:
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            out[key] = (float(np.mean(values)), bootstrap_ci(values))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    ap.add_argument(
        "--tail", type=float, default=0.25, help="fraction of final steps to average"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="reward counted as a high-reward mode",
    )
    ap.add_argument(
        "--budget", type=int, help="molecules evaluated for the mode count (default: all)"
    )
    ap.add_argument(
        "--top-k", type=int, default=100, help="molecules used for diversity and #circles"
    )
    ap.add_argument(
        "--scaff-top-k", type=int, default=1000, help="molecules for the scaff@k column"
    )
    ap.add_argument(
        "--circle-threshold",
        type=float,
        default=0.75,
        help="two molecules join the same circle above this Tanimoto similarity",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        help="truncate every run to this many steps before summarising, so arms "
        "that ran for different budgets stay comparable",
    )
    ap.add_argument("--out", type=Path, help="write the numbers as JSON")
    args = ap.parse_args(argv)

    results = {}
    for item in args.runs:
        name, _, paths = item.partition("=")
        seeds = [
            summarise(
                Path(p),
                args.tail,
                args.threshold,
                args.budget,
                args.top_k,
                args.circle_threshold,
                args.scaff_top_k,
                args.max_steps,
            )
            for p in paths.split(",")
        ]
        results[name] = aggregate(seeds)
        results[name]["seeds"] = (float(len(seeds)), None)

    columns = [
        ("seeds", "{:.0f}"),
        ("steps", "{:.0f}"),
        ("reward", "{:.3f}"),
        ("abs_tb_error", "{:.2f}"),
        ("traj_log_p_B", "{:.2f}"),
        ("bck_rollout_success", "{:.2f}"),
        ("bck_nll", "{:.3f}"),
        ("bck_uniform_nll", "{:.3f}"),
        ("bck_entropy", "{:.3f}"),
        ("step_time_s", "{:.1f}"),
        ("evaluated", "{:.0f}"),
        ("unique_high", "{:.0f}"),
        ("scaffolds", "{:.0f}"),
        (f"scaff@{args.scaff_top_k}", "{:.0f}"),
        ("scaff_total", "{:.0f}"),
        ("top10", "{:.3f}"),
        ("top100", "{:.3f}"),
        ("auc_top10", "{:.3f}"),
        (f"div@{args.top_k}", "{:.3f}"),
        ("circles", "{:.0f}"),
        ("route_div", "{:.3f}"),
        ("route_uniq", "{:.3f}"),
        ("n_reactions", "{:.2f}"),
    ]

    def cell(row, key, fmt):
        if row.get(key) is None:
            return "-"
        mean, interval = row[key]
        text = fmt.format(mean)
        if interval is not None:
            lo, hi = interval
            text += f" [{fmt.format(lo)}, {fmt.format(hi)}]"
        return text

    print("| run | " + " | ".join(name for name, _ in columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name, row in results.items():
        print(
            f"| {name} | "
            + " | ".join(cell(row, key, fmt) for key, fmt in columns)
            + " |"
        )
    print()
    print("Tier 3 columns (top-k, auc, diversity, circles) are per-run endpoints:")
    print("with 2-3 seeds they cannot support a claim; read the interval, not the mean.")
    if all(r.get("unique_high", (0, None))[0] == 0 for r in results.values()):
        print()
        print(
            f"WARNING: no molecule reached --threshold {args.threshold}, so unique_high"
            " and scaffolds are 0 for every arm. Those are SynFlowNet's thresholded mode"
            " count and assume a reward approaching 1, which a 2D similarity reward never"
            " does. Read scaff@k and scaff_total instead, or lower --threshold."
        )
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
