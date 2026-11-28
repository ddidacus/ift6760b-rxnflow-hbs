"""Tabulate the normalize_probe A/B from each run's training.log.

    python scripts/analyze_normalize_probe.py runs/normalize_probe/*

Reads the HBS probe lines, which carry no step of their own, so they are attributed
to the most recent "Step N" line above them. Reports each quantity over the first
and last fifth of the run, which is what distinguishes "the probe learned to
retrieve" from "it only rescaled itself".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROBE = re.compile(
    r"HBS probe \(library (?P<lib>\S+), N=(?P<total>\d+), \d+ queries\): "
    r"best row in bucket (?P<hit1>[\d.]+)% \(chance (?P<chance>[\d.]+)%\), "
    r"its own top rows recalled (?P<recall>[\d.]+)%, "
    r"scanned [\d.]+ of \d+ rows \([\d.]+%\), "
    r"probe-fingerprint cosine best (?P<cos_best>[-\d.]+) / mean (?P<cos_mean>[-\d.]+)"
    r"(?:, probe norm (?P<norm>[\d.]+))?"
)
STEP = re.compile(r"Step (?P<step>\d+)/\d+:")
FIELDS = ("cos_best", "cos_mean", "hit1", "recall", "chance", "norm")


def parse_log(path: Path) -> list[dict]:
    """Probe measurements, each tagged with the step it was logged under."""
    rows, step = [], 0
    for line in path.read_text(errors="replace").splitlines():
        found = STEP.search(line)
        if found:
            step = int(found["step"])
            continue
        found = PROBE.search(line)
        if found:
            row = {"step": step, "lib": found["lib"], "total": int(found["total"])}
            for field in FIELDS:
                value = found[field]
                row[field] = float(value) if value is not None else None
            rows.append(row)
    return rows


def mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def window(rows: list[dict], field: str, first: bool) -> float | None:
    """Mean of `field` over the first or last fifth of the measurements."""
    if not rows:
        return None
    edge = max(1, len(rows) // 5)
    return mean([r[field] for r in (rows[:edge] if first else rows[-edge:])])


def final_reward(run: Path) -> tuple[float | None, int | None]:
    """Best reward seen and the last step recorded in training.jsonl."""
    path = run / "training.jsonl"
    if not path.exists():
        return None, None
    best, last = None, None
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = record.get("step", last)
        value = record.get("reward")
        if value is not None:
            best = value if best is None else max(best, value)
    return best, last


def show(value: float | None, width: int = 7, places: int = 3) -> str:
    return "-".rjust(width) if value is None else f"{value:>{width}.{places}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    print(
        f"{'run':34s} {'steps':>6} {'probes':>6} "
        f"{'norm@1':>7} {'norm@5':>7} {'cos@1':>7} {'cos@5':>7} "
        f"{'hit1@5':>7} {'chance':>7} {'recall@5':>8} {'bestR':>7}"
    )
    for run in sorted(args.runs):
        log = run / "training.log"
        if not log.exists():
            print(f"{run.name:34s} (no training.log)")
            continue
        rows = parse_log(log)
        # The largest library is the one the probe is measured on; keep only it so a
        # mixed set of libraries cannot average two different retrieval problems.
        if rows:
            biggest = max({r["lib"]: r["total"] for r in rows}.items(), key=lambda i: i[1])
            rows = [r for r in rows if r["lib"] == biggest[0]]
        best, last = final_reward(run)
        print(
            f"{run.name:34s} {last if last else 0:6d} {len(rows):6d} "
            f"{show(window(rows, 'norm', True))} {show(window(rows, 'norm', False))} "
            f"{show(window(rows, 'cos_best', True), places=4)} "
            f"{show(window(rows, 'cos_best', False), places=4)} "
            f"{show(window(rows, 'hit1', False), places=2)} "
            f"{show(window(rows, 'chance', False), places=2)} "
            f"{show(window(rows, 'recall', False), width=8, places=2)} "
            f"{show(best, places=4)}"
        )
    print(
        "\n@1 = first fifth of the run, @5 = last fifth. hit1/chance/recall are "
        "percentages.\nThe hypothesis predicts: norm flat at 1.0 when on and rising "
        "when off, and\ncos/hit1/recall improving over the run only when on."
    )


if __name__ == "__main__":
    main()
