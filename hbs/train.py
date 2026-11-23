"""Train RxnFlow against a roshambo2 target, with uniform or HBS action selection.

    python -m hbs.train --config hbs/configs/cdk8.yaml \
        --steps 2500 --output-dir runs/cdk8_uniform

Adding --hbs swaps uniform action subsampling for hierarchical binary search over
the synthon embeddings (hbs/policy.py). Nothing else differs between the arms:
same config, same seed, same reward, same number of candidates scored per library.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import torch

from rxnflow import Config, RxnFlowTrainer

from hbs.metrics import MetricsSettings
from hbs.policy import HBSSettings
from hbs.reward import Roshambo2Reward
from hbs.trainer import HBSRxnFlowTrainer, MetricsRxnFlowTrainer

logger = logging.getLogger("hbs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="training updates")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", help="cpu or cuda; omitted selects CUDA if present")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="retrosynthesis processes"
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)

    group = parser.add_argument_group("per-trajectory metrics")
    group.add_argument(
        "--metrics-every",
        type=int,
        default=25,
        help="steps between candidate-set and route measurements; 0 disables "
        "(latency is recorded on every step whenever metrics are enabled)",
    )
    group.add_argument(
        "--metrics-states",
        type=int,
        default=8,
        help="trajectories measured per recorded step",
    )
    group.add_argument(
        "--metrics-pairs",
        type=int,
        default=2000,
        help="pair budget per diversity estimate; larger sets are sampled",
    )

    group = parser.add_argument_group("hierarchical binary search")
    group.add_argument(
        "--hbs",
        action="store_true",
        help="replace action subsampling with hashed lookup over the whole reservoir",
    )
    group.add_argument(
        "--hbs-depth",
        type=int,
        default=16,
        help="hyperplanes, so 2^depth buckets: the resolution of the search "
        "(default 16)",
    )
    group.add_argument(
        "--hbs-min-rows",
        type=int,
        default=16,
        help="candidates to collect per group: the query's bucket, then its Hamming "
        "neighbours until this many rows are in hand (default 16). Buckets open "
        "nearest-first, so this sets how far out the offer reaches, not which near "
        "blocks are kept; pair a small value with --hbs-noise-sigma for coverage",
    )
    group.add_argument(
        "--hbs-probe-every",
        type=int,
        default=200,
        help="calls between retrieval-quality probes; 0 disables",
    )
    group.add_argument(
        "--hbs-noise-sigma",
        type=float,
        default=0.0,
        help="deterministic jitter added to the probe before hashing only, in units of "
        "its own norm; gives the offered block set temporal variation without touching "
        "the logits. 0 disables. Read it through code_bits_flipped in the metrics",
    )
    group.add_argument(
        "--hbs-noise-seed",
        type=int,
        default=0,
        help="seed for that jitter; it is a function of (seed, step, group), never "
        "fresh randomness, so sampling and the loss open the same buckets",
    )
    group.add_argument(
        "--hbs-normalize-probe",
        action="store_true",
        help="L2-normalize the query before scoring, so a logit is a pure cosine and "
        "gradient cannot drift the direction the hash depends on",
    )
    group.add_argument(
        "--hbs-center",
        action="store_true",
        help="hash about the reservoir centroid rather than the origin",
    )
    group.add_argument(
        "--hbs-seed",
        type=int,
        default=0,
        help="seed for the shared hyperplanes",
    )
    return parser


def attach_file_logging(output_dir: Path) -> None:
    """Send this package's logs to the run's training.log, as the trainer does.

    ``rxnflow.trainer.init_logger`` configures its own logger only, and turns off
    propagation, so the reward's and the policy's messages need their own handler
    or they would never reach the file.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    to_file = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    to_file.setFormatter(formatter)
    logger.addHandler(to_file)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = Config.from_file(args.config)
    reward = Roshambo2Reward(**config.reward.settings)

    settings = HBSSettings(
        depth=args.hbs_depth,
        min_rows=args.hbs_min_rows,
        probe_every=args.hbs_probe_every,
        seed=args.hbs_seed,
        noise_sigma=args.hbs_noise_sigma,
        noise_seed=args.hbs_noise_seed,
        normalize_probe=args.hbs_normalize_probe,
        center=args.hbs_center,
    )
    metrics = MetricsSettings(
        every=args.metrics_every,
        states=args.metrics_states,
        max_pairs=args.metrics_pairs,
        seed=args.seed,
    )
    common = dict(
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=args.num_workers,
    )
    if args.hbs:
        trainer = HBSRxnFlowTrainer(
            config, reward, hbs=settings, metrics=metrics, **common
        )
    elif args.metrics_every:
        trainer = MetricsRxnFlowTrainer(config, reward, metrics=metrics, **common)
    else:
        # No metrics and no HBS: upstream's trainer, with nothing of ours in the path.
        trainer = RxnFlowTrainer(config, reward, **common)

    # The trainer creates the output directory, so this can only follow it.
    attach_file_logging(args.output_dir)
    arm = "hbs" if args.hbs else "uniform"
    logger.info(
        "Arm: %s | target: %s | steps: %d | seed: %d | device: %s",
        arm,
        reward.target,
        args.steps,
        args.seed,
        device,
    )
    summary = {
        "arm": arm,
        "config": str(args.config),
        "steps": args.steps,
        "seed": args.seed,
        "device": device,
        "target": reward.target,
        "query_sdf": reward.sdf_path,
        "hbs": vars(settings) if args.hbs else None,
        "metrics": vars(metrics) if args.metrics_every else None,
        "subsampling": {
            "sampling_ratio": config.subsampling.sampling_ratio,
            "min_sampling": config.subsampling.min_sampling,
            "importance_temp": config.subsampling.importance_temp,
        },
    }
    # Written before the first step, so a run that is killed still records which
    # arm it was and how it was configured.
    (args.output_dir / "run_config.json").write_text(json.dumps(summary, indent=2) + "\n")

    started = perf_counter()
    try:
        checkpoint = trainer.run(
            args.steps, resume_from_checkpoint=args.resume_from_checkpoint
        )
        summary["checkpoint"] = str(checkpoint)
        summary["status"] = "complete"
    except BaseException as error:
        summary["status"] = f"failed: {type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_s"] = round(perf_counter() - started, 1)
        summary["steps_done"] = trainer.step
        summary["oracle_calls"] = reward.num_oracle_calls
        # Written last so its presence marks a run that reached the end, which is
        # how the other baselines in this package mark a cell as done.
        name = "run_summary.json" if summary["status"] == "complete" else "run_failed.json"
        (args.output_dir / name).write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("Run summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
