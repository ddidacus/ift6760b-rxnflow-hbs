"""Side-by-side status of running (or finished) runs, from their logs alone.

    .venv/bin/python -m hbs.monitor runs/cdk8_enamine_uniform_s1 runs/cdk8_enamine_hbs_s1

Reads training.jsonl and trajectory_metrics.jsonl, which are appended every step, so
it is safe to call while the runs are training and costs nothing: it never touches
sample.jsonl (hundreds of MB) unless --top-k asks for the best molecules so far.
HBS retrieval quality is parsed out of the arm's own log when one is present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

PROBE = re.compile(
    r"best row in bucket ([\d.]+)% \(chance ([\d.]+)%\).*?scanned [\d.]+ of \d+ "
    r"rows \(([\d.]+)%\)"
    r"(?:, probe-fingerprint cosine best ([-\d.]+) / mean ([-\d.]+))?"
)


def trend(values: list[float], window: int) -> tuple[float | None, float | None]:
    """(recent mean, recent minus earliest) over equal-sized windows.

    The delta is what answers "is this still improving"; a single mean cannot.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    span = max(1, min(window or len(clean), len(clean) // 2)) if len(clean) > 1 else 1
    recent = float(np.mean(clean[-span:]))
    if len(clean) < 2 * span:
        return recent, None
    return recent, recent - float(np.mean(clean[:span]))


def tail_json(path: Path, window: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line still being written
    return rows[-window:] if window else rows


def flat(values: list) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(v for v in value if v is not None)
        elif value is not None:
            out.append(value)
    return out


def mean(values: list) -> float | None:
    cleaned = flat(values)
    return float(np.mean(cleaned)) if cleaned else None


def summarize(
    run_dir: Path, window: int, top_k: int, max_step: int | None = None
) -> dict[str, Any]:
    training = tail_json(run_dir / "training.jsonl", 0)
    metrics = tail_json(run_dir / "trajectory_metrics.jsonl", 0)
    if max_step is not None:
        # Compare arms at the same step rather than at each one's own latest.
        training = [r for r in training if r["step"] <= max_step]
        metrics = [r for r in metrics if r["step"] <= max_step]
    config = run_dir / "run_config.json"
    arm = json.loads(config.read_text())["arm"] if config.is_file() else "?"
    recent = training[-window:]
    out: dict[str, Any] = {
        "run": run_dir.name,
        "arm": arm,
        "step": training[-1]["step"] if training else 0,
        "s/step": mean([r["time"] for r in recent]),
        "reward": mean([r["reward"] for r in recent]),
        "reward_trend": trend([r["reward"] for r in training], window)[1],
        "loss": mean([r["loss"] for r in recent]),
        "traj_len": mean([r["traj_lens"] for r in recent]),
        "unique%": (
            100 * mean([r["num_unique"] / max(r["num_valid"], 1) for r in recent])
            if recent
            else None
        ),
        "entropy": mean([r["batch_entropy"] for r in recent]),
    }
    if training:
        done = run_dir / "run_summary.json"
        out["state"] = "done" if done.is_file() else "running"
        out["hours"] = sum(r["time"] for r in training) / 3600

    chem_all = [r for r in metrics if "route_div" in r]
    chem = chem_all[-window:] if window else chem_all
    latency = metrics[-window:] if window else metrics
    if latency:
        out["latency_ms"] = mean([r["latency_ms"] for r in latency])
        out["policy_ms"] = mean([r["policy_ms"] for r in latency])
    if chem:
        if "block_div_library" in chem[-1]:
            # Baseline only: is uniform sampling still handing the policy variety?
            # Records written before the library reference existed simply lack it.
            per_step = [
                mean(r["block_div_library"])
                for r in chem_all
                if "block_div_library" in r
            ]
            out["cand_div_lib"], out["cand_div_lib_trend"] = trend(per_step, window)
            reference = mean([r.get("library_div_reference") for r in chem])
            if reference and out["cand_div_lib"]:
                out["cand_div_vs_library"] = out["cand_div_lib"] / reference
            out["cands/state"] = mean([r["num_candidates_measured"] for r in chem])
        aligned = [r["probe_alignment"] for r in metrics if "probe_alignment" in r]
        if aligned:
            out["probe_alignment"], out["probe_alignment_trend"] = trend(aligned, window)
        out["route_div"], out["route_div_trend"] = trend(
            [mean(r["route_div"]) for r in chem_all], window
        )
        out["route_scaffolds"] = mean([r["route_scaffolds"] for r in chem])
        out["metric_steps"] = len(chem_all)

    recalls, chances, cos_best, cos_mean = [], [], [], []
    for log in sorted(run_dir.glob("training.log")):
        for line in log.open(errors="ignore"):
            found = PROBE.search(line)
            if found:
                recalls.append(float(found.group(1)))
                chances.append(float(found.group(2)))
                if found.group(4) is not None:
                    cos_best.append(float(found.group(4)))
                    cos_mean.append(float(found.group(5)))
    if recalls:
        hit = float(np.mean(recalls[-window:]))
        chance = float(np.mean(chances[-window:]))
        out["probe_hit%"] = hit
        out["probe_chance%"] = chance
        out["probe_lift"] = hit / chance if chance else None
        out["probe_scanned%"] = chance  # the scanned share *is* the chance level
        out["probes"] = len(recalls)
    if cos_best:
        # The question the fixed-reservoir design turns on: is the learned probe
        # moving toward the fingerprints it has to retrieve?
        out["probe_cos_best"], out["probe_cos_best_trend"] = trend(cos_best, window)
        out["probe_cos_mean"], out["probe_cos_mean_trend"] = trend(cos_mean, window)

    if top_k:
        best: dict[str, float] = {}
        limit = max_step
        samples = run_dir / "sample.jsonl"
        if samples.is_file():
            for line in samples.open(encoding="utf-8"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if limit is not None and row.get("step", 0) > limit:
                    continue
                smiles = row.get("final_smiles")
                if row.get("valid") and smiles:
                    value = float(row["reward"])
                    if value > best.get(smiles, -1.0):
                        best[smiles] = value
            ranked = sorted(best.values(), reverse=True)
            out["unique_valid"] = len(ranked)
            out[f"top{top_k}"] = float(np.mean(ranked[:top_k])) if ranked else None
            out["top1"] = ranked[0] if ranked else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="steps averaged for the recent columns (0 = whole run)",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="truncate every run to this step; 'match' is the shortest run's last step",
    )
    parser.add_argument(
        "--match",
        action="store_true",
        help="truncate every run to the last step the shortest of them reached",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="also scan sample.jsonl for the best k distinct molecules (slow)",
    )
    args = parser.parse_args()

    max_step = args.max_step
    if args.match and max_step is None:
        reached = []
        for run_dir in args.run_dirs:
            rows = tail_json(run_dir / "training.jsonl", 0)
            if rows:
                reached.append(rows[-1]["step"])
        max_step = min(reached) if reached else None
    rows = [summarize(d, args.window, args.top_k, max_step) for d in args.run_dirs]
    if max_step is not None:
        print(f"(all runs truncated to step {max_step})")
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys and k != "run")
    width = max(len(k) for k in keys) + 2
    print("".ljust(width) + "".join(r["run"][-26:].rjust(28) for r in rows))
    for key in keys:
        cells = []
        for row in rows:
            value = row.get(key)
            if value is None:
                cells.append("-".rjust(28))
            elif isinstance(value, float):
                cells.append(f"{value:28.4f}")
            elif isinstance(value, int):
                cells.append(f"{value:28d}")
            else:
                cells.append(str(value).rjust(28))
        print(key.ljust(width) + "".join(cells))


if __name__ == "__main__":
    main()
