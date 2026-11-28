"""QED training with full-space action selection instead of uniform subsampling.

Run: python examples/hbs_qed.py --config configs/qed.yaml --steps 10

Same reward and config as ``examples/qed.py``; the only difference is the trainer.
``--arm hbs`` scores the hashed bucket a query falls in (plus a uniform draw),
``--arm cas`` samples a k-means cell and then a row inside it, ``--arm cas-fullspace``
clusters the whole catalog instead of one library at a time, and ``--arm uniform`` is
upstream's estimator, kept here so the arms are one flag apart.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

from rxnflow import Config, RewardFunction, RxnFlowTrainer
from rxnflow.gflownet.fwd.cas.fullspace import (
    GlobalCASRxnFlowTrainer,
    GlobalCASSettings,
)
from rxnflow.gflownet.fwd.cas.policy import CASSettings
from rxnflow.gflownet.fwd.cas.trainer import CASRxnFlowTrainer
from rxnflow.gflownet.fwd.hbs.policy import HBSSettings
from rxnflow.gflownet.fwd.hbs.trainer import HBSRxnFlowTrainer
from rxnflow.gflownet.fwd.metrics import MetricsSettings
from rxnflow.gflownet.fwd.trainer import MetricsRxnFlowTrainer

ARMS = ("hbs", "cas", "cas-fullspace", "uniform")


class QEDReward(RewardFunction):
    objectives = ("qed",)

    def score(self, smiles_list: list[str]) -> np.ndarray:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        qeds = [QED.qed(mol) for mol in mols]
        return np.array(qeds, dtype=np.float32).reshape(-1, 1)


def build_trainer(arm: str, config: Config, reward: RewardFunction, **kwargs):
    """The trainer for one arm; everything else about the run is identical."""
    metrics = MetricsSettings(every=kwargs.pop("metrics_every", 0))
    if arm == "hbs":
        return HBSRxnFlowTrainer(
            config, reward, hbs=HBSSettings(), metrics=metrics, **kwargs
        )
    if arm == "cas":
        return CASRxnFlowTrainer(
            config, reward, cas=CASSettings(), metrics=metrics, **kwargs
        )
    if arm == "cas-fullspace":
        return GlobalCASRxnFlowTrainer(
            config, reward, cas=GlobalCASSettings(), metrics=metrics, **kwargs
        )
    if metrics.every:
        return MetricsRxnFlowTrainer(config, reward, metrics=metrics, **kwargs)
    return RxnFlowTrainer(config, reward, **kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hbs_qed"))
    parser.add_argument(
        "--steps", type=int, required=True, help="additional training updates"
    )
    parser.add_argument("--arm", choices=ARMS, default="hbs")
    parser.add_argument(
        "--metrics-every", type=int, default=0, help="steps between metrics; 0 disables"
    )
    parser.add_argument(
        "--device", help="device, e.g. cpu or cuda; omitted selects CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    # The trainer configures its own logger only, so the arm's messages (reservoir
    # built, index fitted, settings) need a handler of their own to be seen.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = Config.from_file(args.config)
    reward = QEDReward(**config.reward.settings)
    trainer = build_trainer(
        args.arm,
        config,
        reward,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
        num_workers=args.num_workers,
        metrics_every=args.metrics_every,
    )
    checkpoint = trainer.run(
        args.steps, resume_from_checkpoint=args.resume_from_checkpoint
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
