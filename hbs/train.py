"""Train RxnFlow on a roshambo2 target or a PMO task, with uniform or HBS actions.

    python -m hbs.train --config hbs/configs/cdk8.yaml \
        --steps 2500 --output-dir runs/cdk8_uniform

    python -m hbs.train --config hbs/configs/pmo_jnk3.yaml \
        --steps 1200 --output-dir runs/pmo_jnk3_uniform_s1

Adding --hbs swaps uniform action subsampling for hierarchical binary search over
the synthon embeddings (fwd/hbs/policy.py). Nothing else differs between the arms:
same config, same seed, same reward, same number of candidates scored per library.

Which reward is used comes from the config, so a config file is self-describing:
``reward.settings.kind`` is ``roshambo2`` (the default, for back-compatibility
with the CDK8 configs), ``pmo``, or ``sim2d`` -- the 2D fingerprint similarity
of hbs/reward2d.py, which scores the same nine query ligands without conformers
or a GPU. ``--reward`` overrides it for an ad-hoc run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import logging
from pathlib import Path
from time import perf_counter

import torch

from rxnflow import Config, RewardFunction, RxnFlowTrainer
from rxnflow.gflownet.fwd.cas.fullspace import (
    GlobalCASRxnFlowTrainer,
    GlobalCASSettings,
)
from rxnflow.gflownet.fwd.base import Temperature
from rxnflow.gflownet.fwd.cas.policy import CASSettings
from rxnflow.gflownet.fwd.cas.trainer import CASRxnFlowTrainer
from rxnflow.gflownet.fwd.hbs.policy import HBSSettings
from rxnflow.gflownet.fwd.hbs.trainer import HBSRxnFlowTrainer
from rxnflow.gflownet.fwd.metrics import MetricsSettings
from rxnflow.gflownet.fwd.trainer import MetricsRxnFlowTrainer

from hbs.pmo import OracleBudgetExhausted, PMOReward
from hbs.reward2d import Mol2DSimilarityReward

logger = logging.getLogger("hbs")
# The arms log under their own package; their records go to the run's file too.
ARM_LOGGER = "rxnflow.gflownet.fwd"

REWARD_KINDS = ("roshambo2", "pmo", "sim2d")


def build_reward(config: Config, kind: str | None, pmo_budget: int | None) -> tuple:
    """Construct the reward named by the config, and a summary of it.

    ``reward.settings`` in the YAML is passed to the reward's constructor as it
    stands, minus the ``kind`` selector, which is this package's and not
    RxnFlow's. The second return value goes into ``run_config.json`` so a run
    records which objective it was actually optimizing.
    """
    settings = dict(config.reward.settings)
    kind = kind or settings.pop("kind", "roshambo2")
    settings.pop("kind", None)
    if kind not in REWARD_KINDS:
        raise ValueError(f"reward kind must be one of {REWARD_KINDS}, got {kind!r}")

    if kind == "pmo":
        if pmo_budget is not None:
            settings["max_oracle_calls"] = pmo_budget
        reward: RewardFunction = PMOReward(**settings)
        summary = {
            "kind": kind,
            "task": reward.task,
            "oracle_dir": reward.oracle_dir,
            "max_oracle_calls": reward.max_oracle_calls,
        }
    elif kind == "sim2d":
        reward = Mol2DSimilarityReward(**settings)
        summary = {
            "kind": kind,
            "target": reward.target,
            "query_sdf": reward.sdf_path,
            "target_smiles": reward.target_smiles,
        }
    else:
        # Imported here, not at module scope: hbs.reward pulls in the roshambo2
        # package, which needs a CUDA build, and the other two kinds must be
        # usable on a machine that has none.
        from hbs.reward import Roshambo2Reward

        reward = Roshambo2Reward(**settings)
        summary = {
            "kind": kind,
            "target": reward.target,
            "query_sdf": reward.sdf_path,
        }
    return reward, summary


def legacy_reward_fields(summary: dict) -> dict:
    """The reward fields earlier CDK8 runs wrote at the top level of their summary.

    ``run_config.json`` used to carry ``target`` and ``query_sdf`` directly, before
    there was more than one kind of reward. The 14 completed CDK8 runs in
    ``runs/`` have them there, so they are still written for that arm and a new
    CDK8 run's record stays comparable with an old one field for field.
    """
    if summary["kind"] != "roshambo2":
        return {}
    return {"target": summary["target"], "query_sdf": summary["query_sdf"]}


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

    group = parser.add_argument_group("reward")
    group.add_argument(
        "--reward",
        choices=REWARD_KINDS,
        help="override reward.settings.kind in the config",
    )
    group.add_argument(
        "--pmo-budget",
        type=int,
        help="oracle-call budget for a PMO run, overriding the config; PMO's "
        "published protocol is 10000. The run ends when it is spent, whichever "
        "comes first with --steps",
    )

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

    group = parser.add_argument_group("clustered action sampling (RxnFlow-CAS)")
    group.add_argument(
        "--cas",
        action="store_true",
        help="select candidates from k-means cells instead of hashed buckets, with "
        "the cell choice inside the policy: P(action) = P(cell) * P(row|cell). Unlike "
        "--hbs the cell choice is differentiable, so the loss can move the probe "
        "toward a better cell; see fwd/cas/policy.py",
    )
    group.add_argument(
        "--cas-rows-per-cell",
        type=int,
        default=235,
        help="target cell occupancy, which sets nlist per library as round(N/this). "
        "235 matches a 482k-row global index at nlist=2048, and the occupancy a 22M "
        "catalog gets at depth 16 (default 235)",
    )
    group.add_argument(
        "--cas-nprobe",
        type=int,
        default=1,
        help="cells opened per state; 1 keeps the latent decomposition exact",
    )
    group.add_argument(
        "--cas-tau",
        type=float,
        default=1.0,
        help="temperature of the cell distribution; large spreads mass over cells, "
        "small commits to one (default 1.0)",
    )
    group.add_argument(
        "--cas-normalize-probe",
        action="store_true",
        help="L2-normalize the probe, so a logit is a pure cosine",
    )
    group.add_argument(
        "--cas-fullspace",
        "--cas-global",
        dest="cas_global",
        action="store_true",
        help="CAS-FullSpace: one k-means over the WHOLE catalog instead of one per "
        "library, so the probe may retrieve any building block and the library filter "
        "becomes a measurement (legal_fraction) rather than a constraint. Without it "
        "the arm is CAS-Validity. --cas-global is the old spelling, still accepted. "
        "See fwd/cas/fullspace.py",
    )
    group.add_argument(
        "--cas-nlist", type=int, default=2048,
        help="cells in the global index (CAS-FullSpace only); 2048 over 482,070 rows "
        "is 235 rows/cell, matching the per-library variant's occupancy",
    )
    group.add_argument(
        "--cas-min-legal", type=int, default=32,
        help="legal rows a state wants before it stops opening cells (CAS-FullSpace)",
    )
    group.add_argument(
        "--cas-max-cells", type=int, default=16,
        help="cap on cells opened per state per query (CAS-FullSpace)",
    )
    group.add_argument(
        "--cas-kmeans-iters", type=int, default=25, help="Lloyd iterations (default 25)"
    )
    group.add_argument(
        "--cas-seed", type=int, default=0, help="seed for k-means initialization"
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
        "--hbs-pure-bucket",
        action="store_true",
        help="score the bucket only and declare log(N/K)=0, as the pre-fix runs did. "
        "Provably biased -- it understated logZ by ~15.5 nats on jnk3 -- and kept "
        "only to reproduce them; see fwd/sampling.py",
    )
    group.add_argument(
        "--hbs-uniform-rows",
        type=int,
        help="rows in the hybrid's uniform half; omitted uses upstream's K per "
        "library, so the uniform half is exactly the baseline arm's estimator",
    )
    group.add_argument(
        "--hbs-explore-temp",
        type=float,
        default=0.0,
        help="weight on the row importances inside the proposal (not the "
        "normalization); 0 proposes from the policy's raw preference",
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

    group = parser.add_argument_group("differentiable bucket selection")
    group.add_argument(
        "--hierarchical",
        action="store_true",
        help="open one bucket per state, scored by <code, hyperplane scores> -- a "
        "differentiable similarity whose argmax is the hash code itself -- instead of "
        "walking a Hamming ball and adding a uniform draw. Makes P_F = P(bucket) * "
        "P(row|bucket), so the probe gets gradient for which bucket it opens. Pair it "
        "with --hbs-depth 8: at 16 a bucket holds one row",
    )
    group.add_argument(
        "--bucket-term",
        choices=("numerator", "partition", "off"),
        default="numerator",
        help="where log P(bucket) enters the loss; see fwd/hbs/policy.py",
    )
    group.add_argument(
        "--temp-bucket", type=float, default=1.0,
        help="temperature of the bucket softmax, at the start of annealing",
    )
    group.add_argument(
        "--temp-bucket-end", type=float,
        help="temperature of the bucket softmax at --anneal-steps; omitted holds it",
    )
    group.add_argument(
        "--temp-nn", type=float, default=1.0,
        help="temperature of the within-bucket row softmax, at the start",
    )
    group.add_argument(
        "--temp-nn-end", type=float,
        help="temperature of the row softmax at --anneal-steps; omitted holds it",
    )
    group.add_argument(
        "--anneal-steps", type=int, default=0,
        help="steps over which both temperatures move, geometrically; 0 holds them",
    )
    return parser


def attach_file_logging(output_dir: Path) -> None:
    """Send the rewards' and the policies' logs to the run's training.log.

    ``rxnflow.trainer.init_logger`` configures its own logger only, and turns off
    propagation, so both of these need handlers of their own: ``hbs`` for the
    rewards here, and ``rxnflow.gflownet.fwd`` for the arms, whose messages (the
    reservoir summary, the HBS retrieval probe, the CAS index) are the diagnostics
    a run is read through.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    to_file = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    to_file.setFormatter(formatter)
    for target in (logger, logging.getLogger(ARM_LOGGER)):
        target.setLevel(logging.INFO)
        target.propagate = False
        for handler in target.handlers[:]:
            target.removeHandler(handler)
            handler.close()
        target.addHandler(console)
        target.addHandler(to_file)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = Config.from_file(args.config)
    reward, reward_summary = build_reward(config, args.reward, args.pmo_budget)

    temp_nn = Temperature(
        start=args.temp_nn,
        end=args.temp_nn_end if args.temp_nn_end is not None else args.temp_nn,
        steps=args.anneal_steps,
    )
    cas_cls = GlobalCASSettings if args.cas_global else CASSettings
    cas_extra = (
        dict(nlist=args.cas_nlist, min_legal=args.cas_min_legal,
             max_cells=args.cas_max_cells)
        if args.cas_global
        else {}
    )
    cas_settings = cas_cls(
        rows_per_cell=args.cas_rows_per_cell,
        nprobe=args.cas_nprobe,
        temp_bucket=Temperature(
            start=args.cas_tau,
            end=args.temp_bucket_end if args.temp_bucket_end is not None else args.cas_tau,
            steps=args.anneal_steps,
        ),
        temp_nn=temp_nn,
        bucket_term=args.bucket_term,
        normalize_probe=args.cas_normalize_probe,
        seed=args.cas_seed,
        kmeans_iters=args.cas_kmeans_iters,
        **cas_extra,
    )
    temp_bucket = Temperature(
        start=args.temp_bucket,
        end=args.temp_bucket_end if args.temp_bucket_end is not None else args.temp_bucket,
        steps=args.anneal_steps,
    )
    settings = HBSSettings(
        hierarchical=args.hierarchical,
        bucket_term=args.bucket_term,
        temp_bucket=temp_bucket,
        temp_nn=temp_nn,
        depth=args.hbs_depth,
        min_rows=args.hbs_min_rows,
        hybrid=not args.hbs_pure_bucket,
        uniform_rows=args.hbs_uniform_rows,
        explore_temp=args.hbs_explore_temp,
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
    if args.cas and args.hbs:
        raise SystemExit("--cas and --hbs select different arms; pass only one")
    if args.cas_global and not args.cas:
        raise SystemExit("--cas-global needs --cas")
    if args.cas_global:
        trainer = GlobalCASRxnFlowTrainer(
            config, reward, cas=cas_settings, metrics=metrics, **common
        )
    elif args.cas:
        trainer = CASRxnFlowTrainer(
            config, reward, cas=cas_settings, metrics=metrics, **common
        )
    elif args.hbs:
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
    # Variant names: CAS-Validity keeps the per-library division, so every offered
    # row is reaction-legal by construction; CAS-FullSpace clusters the whole catalog
    # and measures legality instead of enforcing it.
    arm = ("cas-fullspace" if args.cas_global else "cas-validity") if args.cas else (
        "hbs" if args.hbs else "uniform")
    objective = ", ".join(reward.objectives)
    logger.info(
        "Arm: %s | reward: %s (%s) | steps: %d | seed: %d | device: %s",
        arm,
        objective,
        reward_summary["kind"],
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
        "objectives": list(reward.objectives),
        "reward": reward_summary,
        **legacy_reward_fields(reward_summary),
        "hbs": asdict(settings) if args.hbs else None,
        "cas": asdict(cas_settings) if args.cas else None,
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
    except OracleBudgetExhausted as spent:
        # PMO's protocol ends a run on oracle calls, not on steps, so this is a
        # finished run and not a failure: it is recorded as complete so that
        # hbs/analyze.py and hbs/monitor.py treat the cell as done.
        summary["status"] = "complete"
        summary["ended_on"] = str(spent)
        logger.info("%s; ending the run.", spent)
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
