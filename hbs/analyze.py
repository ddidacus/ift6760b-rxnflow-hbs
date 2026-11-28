"""Post-hoc quality + diversity metrics for a hbs run.

Reads <run_dir>/sample.jsonl (one record per sampled molecule) and reports the metrics
this repo reports elsewhere, using the same definitions as molgen/utils/diversity.py:

  avg_top_k                  mean reward of the k best distinct molecules
  div@k                      1 - mean pairwise Tanimoto over the k best distinct molecules
                             (Morgan fingerprints, radius 2, 2048 bits)
  auc_top_10                 area under (oracle calls, avg top-10) / total oracle calls, PMO-style
  scaffolds                  distinct Bemis-Murcko scaffolds
  #circles(t)                greedy count of molecules pairwise further apart than t in
                             Tanimoto distance, taken best-reward-first

Oracle calls are counted as distinct valid molecules, which is what the reward function
charges for (it caches by SMILES) and what run_summary.json reports.

Usage: .venv/bin/python -m hbs.analyze runs/cdk8_uniform_s1 [runs/cdk8_hbs_s1 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rxnflow.gflownet.fwd.metrics import count_scaffolds, morgan_packed, tanimoto_diversity

TOPK = (10, 100, 1000)
CIRCLE_THRESHOLDS = (0.75, 0.5)


def load(run_dir: Path, max_step: int | None = None):
    """Return (best reward per distinct molecule, auc_top_10 curve points).

    `max_step` stops at that training step, so runs of different lengths can be
    compared on a matched window.
    """
    best: dict[str, float] = {}
    curve: list[tuple[int, float]] = []
    top10: list[float] = []
    sampled = 0
    for line in (run_dir / "sample.jsonl").open():
        rec = json.loads(line)
        if max_step is not None and int(rec.get("step", 0)) > max_step:
            break
        sampled += 1
        if not rec.get("valid"):
            continue
        smi = rec.get("final_smiles")
        if not smi:
            continue
        reward = float(rec["reward"])
        if smi in best:
            best[smi] = max(best[smi], reward)
            continue
        best[smi] = reward
        # new molecule == one oracle call
        top10 = sorted(top10 + [reward], reverse=True)[:10]
        if len(best) % 100 == 0:
            curve.append((len(best), sum(top10) / len(top10)))
    return sampled, best, curve


def num_circles(smiles: list[str], threshold: float) -> int:
    """Molecules kept when each must be > threshold Tanimoto distance from all kept ones."""
    import math

    from rxnflow.gflownet.fwd.metrics import _popcount

    kept: list[np.ndarray] = []
    for smi in smiles:
        packed = morgan_packed(smi)
        if packed is None:
            continue
        if kept:
            stack = np.stack(kept)
            shared = _popcount(np.bitwise_and(packed, stack))
            union = _popcount(packed) + _popcount(stack) - shared
            similarity = np.where(union > 0, shared / np.maximum(union, 1), 1.0)
            if similarity.max() > 1.0 - threshold:
                continue
        kept.append(packed)
    return len(kept)


def modes_found(
    records: list[tuple[str, float]],
    reward_threshold: float,
    similarity_threshold: float = 0.7,
) -> list[int]:
    """Running count of distinct high-reward modes, in discovery order.

    The standard GFlowNet sample-efficiency measure: a molecule counts as a new mode
    when it clears ``reward_threshold`` *and* is below ``similarity_threshold`` Tanimoto
    to every mode already counted. Plotted against oracle calls it answers "how fast
    does this method find *distinct* good chemotypes", which neither ``avg_top_k`` nor
    ``auc_top_10`` can: a hill-climber that collapses onto one scaffold scores well on
    both while finding a single mode. That separation is the reason a GFlowNet is being
    used at all, so it is the right thing to measure it on.

    ``records`` is (smiles, reward) in the order the oracle saw them, so the returned
    list is indexed by oracle call and its last entry is the total at that budget.
    Modes are counted greedily in discovery order rather than best-first: the question
    is when a mode was *found*, not how the final set would be summarized.
    """
    from rxnflow.gflownet.fwd.metrics import _popcount

    kept: list[np.ndarray] = []
    curve: list[int] = []
    for smiles, reward in records:
        if reward >= reward_threshold:
            packed = morgan_packed(smiles)
            if packed is not None:
                if kept:
                    stack = np.stack(kept)
                    shared = _popcount(np.bitwise_and(packed, stack))
                    union = _popcount(packed) + _popcount(stack) - shared
                    similarity = np.where(union > 0, shared / np.maximum(union, 1), 1.0)
                    if similarity.max() < similarity_threshold:
                        kept.append(packed)
                else:
                    kept.append(packed)
        curve.append(len(kept))
    return curve


def calls_to_modes(curve: list[int], target: int) -> int | None:
    """Oracle calls needed to reach ``target`` modes, or None if never reached."""
    for calls, found in enumerate(curve, start=1):
        if found >= target:
            return calls
    return None


def auc_top_10(curve: list[tuple[int, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    x = np.array([0] + [p[0] for p in curve], dtype=float)
    y = np.array([0.0] + [p[1] for p in curve], dtype=float)
    return float(np.trapezoid(y, x) / x[-1])


def metrics(run_dir: Path, max_step: int | None = None) -> dict:
    sampled, best, curve = load(run_dir, max_step)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    smiles = [s for s, _ in ranked]
    rewards = [v for _, v in ranked]
    out = {
        "run": run_dir.name,
        "sampled": sampled,
        "unique_valid": len(ranked),
        "oracle_calls": len(ranked),
        "auc_top_10": auc_top_10(curve),
        "scaffolds_all": count_scaffolds(smiles),
    }
    for k in TOPK:
        sub = smiles[:k]
        out[f"avg_top_{k}"] = float(np.mean(rewards[:k]))
        out[f"div@{k}"] = tanimoto_diversity(sub)
        out[f"scaffolds@{k}"] = count_scaffolds(sub)
    for t in CIRCLE_THRESHOLDS:
        out[f"circles@1000_t{t}"] = num_circles(smiles[:1000], t)
    out["scaffolds_per_mol"] = out["scaffolds_all"] / max(out["unique_valid"], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="only count molecules sampled at or before this step (matched comparison)",
    )
    args = ap.parse_args()

    rows = [metrics(d, args.max_step) for d in args.run_dirs]
    keys = [k for k in rows[0] if k != "run"]
    width = max(len(k) for k in keys) + 2
    header = "".ljust(width) + "".join(r["run"].rjust(20) for r in rows)
    print(header)
    for k in keys:
        cells = []
        for r in rows:
            v = r[k]
            cells.append(("%20.4f" % v) if isinstance(v, float) else ("%20d" % v))
        print(k.ljust(width) + "".join(cells))
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
