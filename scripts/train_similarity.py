"""Train against 2D fingerprint similarity to a reference ligand.

The cheap, target-directed reward of notes/11-experiment-design.md: Tanimoto
similarity to a reference ligand (CDK8 by default). Use it to compare backward
policies at Enamine-290k scale, where a 3D shape reward is unaffordable.

Run, from the repository root:

    python scripts/train_similarity.py \
        --config configs/experiments/cdk8_sim_enamine.yaml \
        --steps 300 --output-dir runs/cdk8-sim-heuristic --seed 1

The arms differ only in the `backward` section of the configuration, so the same
command with `cdk8_sim_enamine_bck_mle.yaml` runs the MLE arm. `--num-workers`
sets the retrosynthesis worker count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from IFT6760 import Config, RxnFlowTrainer
from IFT6760.rewards.similarity import SimilarityReward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/similarity"))
    parser.add_argument(
        "--steps", type=int, required=True, help="additional training updates"
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num-workers", type=int, default=4, help="retrosynthesis processes"
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. reward.beta=16 backward.entropy_weight=2",
    )
    args = parser.parse_args(argv)
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = Config.from_file(args.config, args.overrides)
    reward = SimilarityReward(**config.reward.settings)
    # The reference anchors every number this run produces, so record it where the
    # log will show it rather than leaving it implicit in the configuration.
    print(f"reward: Tanimoto to {reward.reference_smiles}")
    print(f"        target={reward.target} radius={reward.radius} bits={reward.fp_size}")
    trainer = RxnFlowTrainer(
        config,
        reward,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=args.num_workers,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
