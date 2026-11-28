"""Plot the monitoring curves of several runs, and dump the curve data as CSV.

One figure, eight panels, four arms. Every panel answers one question and uses a
single y-axis; nothing here is a dual-axis chart. Each line is direct-labelled at its
right end as well as appearing in the legend, so identity never rests on colour alone.

Panels
  1 reward vs step                 the headline optimisation curve
  2 running top-10 vs oracle calls the generative-quality curve (what AUC integrates)
  3 backward NLL vs step           does P_B learn? plotted against its own uniform floor
  4 rollout reach vs step          can a sampled backward walk reach s_0? (REINFORCE only)
  5 |TB residual| vs step          training stability; NOT comparable as a loss across arms
  6 mean log P_B per trajectory    how concentrated P_B is, uniform being the floor
  7 distinct scaffolds vs calls    mode discovery
  8 seconds per step               the cost of each arm

The raw per-step series is drawn faintly behind a rolling mean, so noise stays visible
instead of being hidden by the smoother.

Usage:
  python scripts/plot_runs.py heuristic=experiments/.../heuristic-1200-s1 \
      uniform=.../uniform-1000-s1 mle=.../mle-1000-s1 reinforce=.../reinforce-1000-s1 \
      --max-steps 1000 --out experiments/2026-11-25/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Four series on crossing lines, so the all-pairs check applies: blue+yellow+magenta+
# green is the documented-palette subset that clears it (CVD dE 13.0, normal 19.6).
# Yellow and magenta fall below 3:1 on a light surface, so the relief rule applies and
# every line carries a visible end label.
PALETTE = {
    "heuristic": "#2a78d6",
    "uniform": "#eda100",
    "mle": "#e87ba4",
    "reinforce": "#008300",
}
FALLBACK = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that keeps the array length, shrinking at the edges."""
    if window <= 1 or values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    half = window // 2
    for i in range(values.size):
        lo, hi = max(0, i - half), min(values.size, i + half + 1)
        out[i] = values[lo:hi].mean()
    return out


def running_top_k(samples: list[dict], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Running mean of the best `k` distinct rewards against the oracle-call count."""
    import heapq

    top: list[float] = []
    seen: set[str] = set()
    calls, xs, ys = 0, [], []
    for row in samples:
        if not row["valid"]:
            continue
        calls += 1
        if row["final_smiles"] not in seen:
            seen.add(row["final_smiles"])
            heapq.heappush(top, row["reward"])
            if len(top) > k:
                heapq.heappop(top)
        if calls % 50 == 0:
            xs.append(calls)
            ys.append(sum(top) / len(top))
    return np.array(xs), np.array(ys)


def scaffold_curve(samples: list[dict], every: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative distinct Bemis-Murcko scaffolds against the oracle-call count."""
    scaffolds: set[str] = set()
    calls, xs, ys = 0, [], []
    for row in samples:
        if not row["valid"]:
            continue
        calls += 1
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(row["final_smiles"]))
        except Exception:
            pass
        if calls % every == 0:
            xs.append(calls)
            ys.append(len(scaffolds))
    return np.array(xs), np.array(ys)


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SOFT)
    ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def end_label(pending: list, y_value: float, text: str, color: str) -> None:
    """Queue a right-end label; `place_labels` lays them out once the panel is full."""
    if y_value is not None and np.isfinite(y_value):
        pending.append([float(y_value), text, color])


def place_labels(ax, pending: list, min_gap: float = 0.052) -> None:
    """Draw queued end labels, spread vertically so none overlaps another.

    Lines that converge -- which several of these panels do by design -- would
    otherwise stack their labels into an unreadable smudge. Positions are resolved in
    axes fractions after the limits are final, then nudged apart by `min_gap`.
    """
    if not pending:
        return
    low, high = ax.get_ylim()
    span = (high - low) or 1.0
    items = sorted(pending, key=lambda item: item[0])
    fractions = [(value - low) / span for value, _, _ in items]
    for i in range(1, len(fractions)):  # push up
        fractions[i] = max(fractions[i], fractions[i - 1] + min_gap)
    overflow = fractions[-1] - 1.0
    if overflow > 0:  # then slide the whole stack back inside the panel
        fractions = [f - overflow for f in fractions]
        for i in range(len(fractions) - 2, -1, -1):
            fractions[i] = min(fractions[i], fractions[i + 1] - min_gap)
    for fraction, (_, text, color) in zip(fractions, items):
        ax.annotate(
            text,
            xy=(1.0, min(max(fraction, 0.0), 1.0)),
            xycoords="axes fraction",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="center",
            fontweight="medium",
            annotation_clip=False,
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    ap.add_argument("--max-steps", type=int, help="truncate every run to this many steps")
    ap.add_argument("--window", type=int, default=25, help="rolling-mean window in steps")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict] = {}
    for index, item in enumerate(args.runs):
        name, _, path = item.partition("=")
        run = Path(path)
        rows = load_jsonl(run / "training.jsonl")
        samples = load_jsonl(run / "sample.jsonl") if (run / "sample.jsonl").exists() else []
        if args.max_steps is not None:
            rows = [r for r in rows if r["step"] <= args.max_steps]
            samples = [s for s in samples if s["step"] <= args.max_steps]
        arms[name] = {
            "rows": rows,
            "samples": samples,
            "color": PALETTE.get(name, FALLBACK[index % len(FALLBACK)]),
        }
        print(f"{name}: {len(rows)} steps, {len(samples)} samples", flush=True)

    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")

    def series(rows, key):
        pairs = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
        if not pairs:
            return np.array([]), np.array([])
        x, y = zip(*pairs)
        return np.array(x, dtype=float), np.array(y, dtype=float)

    def line_panel(ax, key, title, ylabel, transform=None, only=None):
        pending: list = []
        for name, arm in arms.items():
            if only and name not in only:
                continue
            x, y = series(arm["rows"], key)
            if x.size == 0:
                continue
            if transform is not None:
                y = transform(y)
            ax.plot(x, y, color=arm["color"], linewidth=0.7, alpha=0.22)
            smooth = rolling(y, args.window)
            ax.plot(x, smooth, color=arm["color"], linewidth=2.0, label=name)
            end_label(pending, smooth[-1], name, arm["color"])
        style(ax, title, "training step", ylabel)
        place_labels(ax, pending)

    line_panel(axes[0][0], "reward", "1. Mean batch reward", "reward")
    axes[0][0].legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="lower right")

    ax = axes[0][1]
    pending: list = []
    for name, arm in arms.items():
        if not arm["samples"]:
            continue
        x, y = running_top_k(arm["samples"], 10)
        ax.plot(x, y, color=arm["color"], linewidth=2.0, label=name)
        end_label(pending, y[-1] if len(y) else None, name, arm["color"])
    style(ax, "2. Running top-10 reward", "molecules evaluated", "mean top-10 reward")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="lower right")
    place_labels(ax, pending)

    ax = axes[1][0]
    pending = []
    for name, arm in arms.items():
        x, y = series(arm["rows"], "bck_nll")
        if x.size == 0:
            continue
        smooth = rolling(y, args.window)
        ax.plot(x, smooth, color=arm["color"], linewidth=2.0, label=name)
        end_label(pending, smooth[-1], name, arm["color"])
        xu, yu = series(arm["rows"], "bck_uniform_nll")
        if xu.size and name == "reinforce":
            floor = rolling(yu, args.window)
            ax.plot(
                xu,
                floor,
                color=INK_SOFT,
                linewidth=1.4,
                linestyle=(0, (4, 3)),
                label="uniform floor",
            )
            end_label(pending, floor[-1], "floor", INK_SOFT)
    style(ax, "3. Backward NLL of the generated edges", "training step", "nats per edge")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="center right")
    place_labels(ax, pending)

    line_panel(
        axes[1][1],
        "bck_rollout_success",
        "4. Backward-walk reach (single sampled walk)",
        "fraction reaching $s_0$",
        only={"reinforce"},
    )
    axes[1][1].legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="lower right")

    line_panel(
        axes[2][0],
        "tb_error",
        "5. |TB residual| (stability, not a cross-arm loss)",
        "|residual|",
        transform=np.abs,
    )
    # The initialisation spike is ~50x the steady state and would flatten everything
    # after it, so the view is clipped to the post-warmup range.
    settled = [
        abs(r["tb_error"])
        for arm in arms.values()
        for r in arm["rows"]
        if r["step"] > 50 and "tb_error" in r
    ]
    if settled:
        axes[2][0].set_ylim(0, float(np.percentile(settled, 99.5)) * 1.1)
    line_panel(
        axes[2][1],
        "traj_log_p_B",
        r"6. Mean $\log P_B$ per trajectory",
        "nats",
    )

    ax = axes[3][0]
    pending = []
    for name, arm in arms.items():
        if not arm["samples"]:
            continue
        x, y = scaffold_curve(arm["samples"])
        ax.plot(x, y, color=arm["color"], linewidth=2.0, label=name)
        end_label(pending, y[-1] if len(y) else None, name, arm["color"])
    style(ax, "7. Distinct scaffolds found", "molecules evaluated", "distinct scaffolds")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="upper left")
    place_labels(ax, pending)

    ax = axes[3][1]
    names = list(arms)
    values = [float(np.mean([r["time"] for r in arms[n]["rows"][-200:]])) for n in names]
    bars = ax.bar(
        names,
        values,
        color=[arms[n]["color"] for n in names],
        width=0.6,
    )
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK,
        )
    style(ax, "8. Seconds per step (last 200 steps)", "", "seconds")
    ax.grid(axis="x", visible=False)

    budget = args.max_steps or max(len(a["rows"]) for a in arms.values())
    fig.suptitle(
        f"Learned backward policy vs RxnFlow's heuristic -- CDK8 similarity, "
        f"Enamine-290k, {budget} steps, 1 seed",
        fontsize=12,
        color=INK,
        x=0.02,
        ha="left",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 0.975, 0.985))
    figure_path = args.out / "monitoring.png"
    fig.savefig(figure_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {figure_path}", flush=True)

    # Tidy CSV so the curves can be re-plotted without re-reading sample.jsonl.
    csv_path = args.out / "curves.csv"
    keys = (
        "reward",
        "bck_nll",
        "bck_uniform_nll",
        "bck_rollout_success",
        "tb_error",
        "traj_log_p_B",
        "time",
        "logZ",
        "grad_norm",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "step", *keys])
        for name, arm in arms.items():
            for row in arm["rows"]:
                writer.writerow([name, row["step"], *(row.get(k) for k in keys)])
    print(f"wrote {csv_path}", flush=True)

    endpoint_path = args.out / "endpoints.csv"
    with endpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "metric", "x", "y"])
        for name, arm in arms.items():
            if not arm["samples"]:
                continue
            for metric, (x, y) in (
                ("running_top10", running_top_k(arm["samples"], 10)),
                ("scaffolds", scaffold_curve(arm["samples"])),
            ):
                for xi, yi in zip(x, y):
                    writer.writerow([name, metric, xi, yi])
    print(f"wrote {endpoint_path}", flush=True)


if __name__ == "__main__":
    main()
