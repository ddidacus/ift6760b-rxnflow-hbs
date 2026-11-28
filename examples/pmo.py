"""PMO benchmark training example (fork addition, not upstream RxnFlow).

Trains plain upstream ``RxnFlowTrainer`` against one task of the Practical
Molecular Optimization benchmark (Gao et al., NeurIPS 2022). The four tasks set
up for this project are ``gsk3b``, ``jnk3``, ``drd2`` and ``amlodipine_mpo``;
``--task`` accepts any of the 23, listed in ``hbs/pmo.py``.

    python examples/pmo.py --config hbs/configs/pmo_jnk3.yaml \
        --steps 1000 --output-dir runs/pmo_jnk3

    python examples/pmo.py --config hbs/configs/pmo_jnk3.yaml --task gsk3b \
        --steps 1000 --output-dir runs/pmo_gsk3b

This is the shortest path to a PMO run and the one to copy when wiring a new
reward. It is *not* the entry point for the experiments: ``python -m hbs.train``
is, because only that one can select the HBS arm and record the per-trajectory
diversity and latency metrics. Run it the way ``hbs/sbatch/train_pmo.sh`` does.

The oracle checkpoints for ``gsk3b``, ``jnk3`` and ``drd2`` are downloaded into
``oracle/`` on first use (~74MB in total); the other 20 tasks are pure RDKit and
need no download.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from rxnflow import Config, RxnFlowTrainer

from hbs.pmo import PMO_TASKS, OracleBudgetExhausted, PMOReward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pmo"))
    parser.add_argument(
        "--steps", type=int, required=True, help="additional training updates"
    )
    parser.add_argument(
        "--task",
        choices=PMO_TASKS,
        help="PMO task, overriding reward.settings.task in the config",
    )
    parser.add_argument(
        "--oracle-budget",
        type=int,
        help="oracle-call budget, overriding the config; PMO publishes 10000. "
        "The run ends on whichever of this and --steps comes first",
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = Config.from_file(args.config)
    settings = dict(config.reward.settings)
    # 'kind' selects the reward class in hbs/train.py; it is not a constructor
    # argument, so it is dropped here as it is there.
    settings.pop("kind", None)
    if args.task is not None:
        settings["task"] = args.task
    if args.oracle_budget is not None:
        settings["max_oracle_calls"] = args.oracle_budget

    reward = PMOReward(**settings)
    trainer = RxnFlowTrainer(
        config,
        reward,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=4,
    )
    try:
        checkpoint = trainer.run(
            args.steps, resume_from_checkpoint=args.resume_from_checkpoint
        )
        print(checkpoint)
    except OracleBudgetExhausted as spent:
        # PMO ends a run on oracle calls, not on steps, so this is the normal end.
        print(f"{spent} after {trainer.step} steps")
    print(f"oracle calls: {reward.num_oracle_calls:,}")


if __name__ == "__main__":
    main()
