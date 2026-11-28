"""VM side: set up if needed, train ONE arm, pack its results for download.

Runs inside the Colab kernel via `colab exec`. One arm per invocation, on purpose:
the kernel stays busy for the whole arm (an idle kernel gets the VM reaped), and the
local driver downloads that arm's archive the moment this returns. A reaped VM
therefore costs at most the arm in flight, never the finished ones. We lost three
trained checkpoints on 2026-11-25 by packing only after all arms had finished.

Driven entirely by environment variables, because `colab exec` has no argv:

  GFN_ARM     arm name, one of heuristic uniform mle reinforce   (required)
  GFN_STEPS   training steps                                     (default 1000)
  GFN_SEED    seed                                               (default 1)
  GFN_WORKERS retrosynthesis worker processes                     (default 2)
  GFN_SETUP   "1" to extract the source tree and install first   (default auto)
  GFN_TAG     subdirectory of /content/out for the archives      (default "run")

Writes /content/out/<tag>/<arm>-<steps>-s<seed>.tar.gz containing training.jsonl,
sample.jsonl, config.yaml, training.log and checkpoints/latest.ckpt.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time

SRC = pathlib.Path("/content/src")
ARM_CONFIG = {
    "heuristic": "configs/experiments/cdk8_sim_enamine.yaml",
    "uniform": "configs/experiments/cdk8_sim_enamine_bck_uniform.yaml",
    "mle": "configs/experiments/cdk8_sim_enamine_bck_mle.yaml",
    "reinforce": "configs/experiments/cdk8_sim_enamine_bck_reinforce.yaml",
}


def shell(cmd: list[str], cwd: str | None = None, tail: int = 2500) -> None:
    """Run a command and surface its output; `colab exec` only sees what we print."""
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout[-tail:], flush=True)
    if result.returncode:
        print("STDERR:", result.stderr[-tail:], flush=True)
        sys.exit(result.returncode)


def setup() -> None:
    """Extract the uploaded source and catalogue, then install the package."""
    SRC.mkdir(exist_ok=True)
    shell(["tar", "xzf", "/content/gfn-src.tar.gz", "-C", str(SRC)])
    envs = SRC / "data" / "envs"
    envs.mkdir(parents=True, exist_ok=True)
    shell(["tar", "xzf", "/content/enamine_290k.tar.gz", "-C", str(envs)])
    shell([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=str(SRC))

    import torch  # imported after install so the VM's wheel is the one reported

    print(
        f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    catalogue = envs / "enamine_290k"
    print(f"catalogue files: {len(list(catalogue.rglob('*')))}", flush=True)


def report(log: pathlib.Path, arm: str, steps: int) -> None:
    """Print the newest row. Monitoring must never be able to kill the job."""
    try:
        rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        row = rows[-1]
        keys = ("bck_rollout_success", "bck_nll", "bck_uniform_nll", "bck_n_cand")
        extra = "".join(
            f" {k.replace('bck_', '')}={row[k]:.3f}" for k in keys if k in row
        )
        print(
            f"  {arm} step={len(rows)}/{steps} reward={row['reward']:.4f}"
            f" t={row.get('time', 0):.2f}s{extra}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - any failure here is only cosmetic
        print(f"  {arm}: progress unavailable ({type(exc).__name__})", flush=True)


def pack(run: pathlib.Path, archive: pathlib.Path) -> None:
    """Archive the arm's results, checkpoint included, ready for `colab download`."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    wanted = ["training.jsonl", "sample.jsonl", "config.yaml", "training.log"]
    with tarfile.open(archive, "w:gz") as tf:
        for name in wanted:
            path = run / name
            if path.is_file():
                tf.add(path, arcname=f"{run.name}/{name}")
        checkpoint = run / "checkpoints" / "latest.ckpt"
        if checkpoint.is_file():
            tf.add(checkpoint, arcname=f"{run.name}/checkpoints/latest.ckpt")
    print(f"PACKED {archive} {archive.stat().st_size}", flush=True)


def main() -> None:
    arm = os.environ.get("GFN_ARM", "").strip()
    if arm not in ARM_CONFIG:
        sys.exit(f"GFN_ARM must be one of {sorted(ARM_CONFIG)}, got {arm!r}")
    steps = int(os.environ.get("GFN_STEPS", "1000"))
    seed = int(os.environ.get("GFN_SEED", "1"))
    workers = int(os.environ.get("GFN_WORKERS", "2"))
    tag = os.environ.get("GFN_TAG", "run")

    want_setup = os.environ.get("GFN_SETUP")
    if want_setup == "1" or (want_setup is None and not (SRC / "pyproject.toml").exists()):
        setup()
    sys.path.insert(0, str(SRC / "src"))

    run = pathlib.Path(f"/content/runs/{arm}-{steps}-s{seed}")
    archive = pathlib.Path(f"/content/out/{tag}/{run.name}.tar.gz")
    log = run / "training.jsonl"
    if log.exists() and len(log.read_text().splitlines()) >= steps:
        print(f"SKIP {arm}: already complete", flush=True)
        pack(run, archive)
        return

    subprocess.run(["rm", "-rf", str(run)], check=False)
    print(f"START {arm} steps={steps} seed={seed}", flush=True)
    started = time.time()
    process = subprocess.Popen(
        [
            sys.executable, "scripts/train_similarity.py",
            "--config", ARM_CONFIG[arm],
            "--steps", str(steps),
            "--output-dir", str(run),
            "--seed", str(seed),
            "--num-workers", str(workers),
        ],
        cwd=str(SRC),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    last = 0.0
    while process.poll() is None:
        time.sleep(20)
        if time.time() - last >= 120:  # a progress line roughly every two minutes
            last = time.time()
            report(log, arm, steps)
    trailing = (process.stdout.read() or "")[-2500:]
    print(f"DONE {arm} rc={process.returncode} wall={time.time() - started:.0f}s", flush=True)
    if process.returncode:
        print(trailing, flush=True)
        sys.exit(process.returncode)
    pack(run, archive)


if __name__ == "__main__":
    main()
