"""Local driver: run the whole four-arm experiment on Colab, one arm at a time.

Run this in your own terminal. It drives the `colab` CLI and nothing else, printing
every command before it runs so you can copy any single step and run it by hand.

The loop is deliberately one `colab exec` per arm:

    for each arm:  colab exec train_arm.py   ->   colab download <arm>.tar.gz

The kernel is busy for the whole arm, which is what stops Colab reaping the VM, and
the arm's results (checkpoint included) are on your disk before the next arm starts.
Packing everything at the end instead is how we lost three trained checkpoints on
2026-11-25, when the VM was reclaimed between the last arm finishing and the pack.

Prerequisites (once per machine):
    uv tool install --python 3.12 google-colab-cli
    gcloud auth application-default login --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
    colab --auth=adc whoami          # should print your address and the four scopes

Typical use:
    python scripts/colab/experiment.py --steps 1000 --seeds 1 --session run1
    python scripts/colab/experiment.py --steps 1000 --seeds 1 2 3 --arms mle reinforce

What it leaves you with, under --dest (default experiments/<today>/):
    <arm>-<steps>-s<seed>/training.jsonl   per-step metrics, one JSON object per step
    <arm>-<steps>-s<seed>/sample.jsonl     per-molecule records with full routes
    <arm>-<steps>-s<seed>/config.yaml      the resolved config, for provenance
    <arm>-<steps>-s<seed>/checkpoints/latest.ckpt   trained weights (P_B included)

Then evaluate locally, no VM needed:
    python scripts/compare_runs.py heuristic=... uniform=... mle=... reinforce=... \
        --max-steps <steps> --threshold 0.5
    python scripts/plot_runs.py <same args> --out experiments/<today>/figures
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ARMS = ("heuristic", "uniform", "mle", "reinforce")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Echo a command, then run it with its output going straight to the terminal."""
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def colab(args: argparse.Namespace, *rest: str) -> list[str]:
    return [args.colab, f"--auth={args.auth}", *rest]


def make_source_archive(dest: Path) -> Path:
    """Tar the package, scripts, configs and tests -- everything except data and runs.

    `./` anchors each exclude to the archive root: a bare `experiments/*` would also
    match `configs/experiments/`, which silently dropped the arm configs once and made
    the VM's test suite fail.
    """
    archive = dest / "gfn-src.tar.gz"
    excludes = [
        "--exclude=./data",
        "--exclude=./runs",
        "--exclude=./experiments",
        "--exclude=./.git",
        "--exclude=./.venv",
        "--exclude=*.ckpt",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
    ]
    subprocess.run(
        ["tar", "czf", str(archive), *excludes, "-C", str(REPO), "."], check=True
    )
    print(f"source archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return archive


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session", default="run1", help="colab session name")
    ap.add_argument("--gpu", default="T4", help="accelerator, or 'none' for CPU")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dest", type=Path, help="local results dir (default experiments/<today>)")
    ap.add_argument(
        "--catalogue",
        type=Path,
        default=REPO / "data" / "envs" / "enamine_290k.tar.gz",
        help="prepared environment tarball to upload",
    )
    ap.add_argument("--colab", default="colab", help="path to the colab executable")
    ap.add_argument("--auth", default="adc", choices=["adc", "oauth2"])
    ap.add_argument(
        "--keep-session",
        action="store_true",
        help="leave the VM running at the end instead of stopping it",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="session already has the source and catalogue; skip upload and setup",
    )
    args = ap.parse_args(argv)

    dest = args.dest or REPO / "experiments" / dt.date.today().isoformat()
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"s{'-'.join(map(str, args.seeds))}-{args.steps}"

    # Git Bash mangles /content/... into C:/Program Files/Git/content/...; these two
    # variables switch that off for the colab CLI's absolute VM paths.
    env = {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}

    if not args.no_upload:
        if not args.catalogue.exists():
            sys.exit(
                f"catalogue not found: {args.catalogue}\n"
                "build it with scripts/prepare.py, then tar the env directory."
            )
        gpu = [] if args.gpu == "none" else ["--gpu", args.gpu]
        if run(colab(args, "new", "-s", args.session, *gpu), env=env).returncode:
            sys.exit("colab new failed: a 400 here means no entitlement for this GPU")
        with tempfile.TemporaryDirectory() as tmp:
            source = make_source_archive(Path(tmp))
            for local in (source, args.catalogue):
                if run(
                    colab(args, "upload", "-s", args.session, str(local), "/content/"),
                    env=env,
                ).returncode:
                    sys.exit(f"upload failed: {local}")

    failures: list[str] = []
    first = not args.no_upload
    for seed in args.seeds:
        for arm in args.arms:
            name = f"{arm}-{args.steps}-s{seed}"
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
            cmd = colab(
                args,
                "exec",
                "-s",
                args.session,
                "--timeout",
                "0",  # a silence timeout, not a run limit: 0 waits indefinitely
                "--env",
                f"GFN_ARM={arm}",
                "--env",
                f"GFN_STEPS={args.steps}",
                "--env",
                f"GFN_SEED={seed}",
                "--env",
                f"GFN_WORKERS={args.workers}",
                "--env",
                f"GFN_TAG={tag}",
                "--env",
                f"GFN_SETUP={'1' if first else '0'}",
                "-f",
                str(HERE / "train_arm.py"),
            )
            if run(cmd, env=env).returncode:
                print(f"!! {name} failed; continuing with the next arm", flush=True)
                failures.append(name)
                continue
            first = False

            archive = dest / f"{name}.tar.gz"
            if run(
                colab(
                    args,
                    "download",
                    "-s",
                    args.session,
                    f"/content/out/{tag}/{name}.tar.gz",
                    str(archive),
                ),
                env=env,
            ).returncode:
                print(f"!! download of {name} failed -- results are still on the VM", flush=True)
                failures.append(f"{name} (download)")
                continue
            with tarfile.open(archive) as tf:
                # `filter=` landed in 3.12 (and 3.10.12); the project declares >=3.10,
                # so fall back rather than crashing on an older interpreter.
                if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters:
                    tf.extractall(dest, filter="data")
                else:
                    tf.extractall(dest)  # noqa: S202 - archive we just produced ourselves
            archive.unlink()
            print(f"OK {name} -> {dest / name}", flush=True)

    if not args.keep_session:
        run(colab(args, "stop", "-s", args.session), env=env)

    print(f"\nresults in {dest}")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    runs = " ".join(
        f"{arm}={dest / f'{arm}-{args.steps}-s{args.seeds[0]}'}" for arm in args.arms
    )
    print("\nnext:")
    print(f"  python scripts/compare_runs.py {runs} --max-steps {args.steps} --threshold 0.5")
    print(f"  python scripts/plot_runs.py {runs} --max-steps {args.steps} --out {dest / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
