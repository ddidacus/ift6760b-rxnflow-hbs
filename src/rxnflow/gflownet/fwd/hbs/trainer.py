"""Trainer for the HBS arm: candidates come from a hashed reservoir, not a subsample.

Builds the reservoir once and registers the query projection on the model, so the
optimizer trains it, the EMA copy tracks it, and checkpoints carry it.
"""

from __future__ import annotations

import logging
from time import perf_counter

import torch
from torch import nn

from rxnflow.gflownet.fwd.base import PROJECTION_ATTRIBUTE, StepClock
from rxnflow.gflownet.fwd.hbs.policy import HBSPolicy, HBSSettings
from rxnflow.gflownet.fwd.hbs.reservoir import HashedReservoir
from rxnflow.gflownet.fwd.trainer import MetricsRxnFlowTrainer
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel

logger = logging.getLogger(__name__)


class HBSRxnFlowTrainer(MetricsRxnFlowTrainer):
    """Training with no action subsampling: candidates come from the hashed reservoir."""

    def __init__(self, *args, hbs: HBSSettings | None = None, **kwargs):
        self.hbs_settings = hbs or HBSSettings()
        self.hbs_settings.validate()
        self.reservoir: HashedReservoir | None = None
        # Shared by both policies so the jittered buckets a batch was sampled from are
        # the same ones its loss is computed over.
        self.noise_clock = StepClock()
        super().__init__(*args, **kwargs)

    def _prepare_models(self) -> None:
        # Built once and shared by both policies: the vectors are the environment's
        # fixed fingerprints, so there is nothing to keep in sync with training.
        started = perf_counter()
        features = {
            name: library.fingerprints for name, library in self.env.synthons.items()
        }
        self.reservoir = HashedReservoir(
            features,
            self.hbs_settings.depth,
            seed=self.hbs_settings.seed,
            center=self.hbs_settings.center,
        )
        logger.info(
            "HBS reservoir hashed once in %.2fs: %s",
            perf_counter() - started,
            self.reservoir.summary(),
        )
        # Registered on both models so the optimizer trains it, _update_ema tracks it
        # (it zips parameters, and this is appended last on both), and it is saved.
        for model in (self.model, self.sampling_model):
            projection = nn.Linear(
                self.config.model.synthon_dim, self.reservoir.dim, bias=False
            ).to(self.device)
            nn.init.xavier_uniform_(projection.weight)
            model.add_module(PROJECTION_ATTRIBUTE, projection)
        with torch.no_grad():
            getattr(self.sampling_model, PROJECTION_ATTRIBUTE).weight.copy_(
                getattr(self.model, PROJECTION_ATTRIBUTE).weight
            )
        logger.info(
            "HBS query projection: %d -> %d (%d parameters), action subsampling off",
            self.config.model.synthon_dim,
            self.reservoir.dim,
            self.config.model.synthon_dim * self.reservoir.dim,
        )

    def _build_policy(self, model: RxnFlowModel) -> RxnFlowPolicy:
        assert self.reservoir is not None
        return HBSPolicy(
            self.env,
            model,
            self.config,
            self.device,
            self.rng,
            self.hbs_settings,
            self.reservoir,
            getattr(model, PROJECTION_ATTRIBUTE),
            self.noise_clock,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        logger.info("HBS action selection: %s", self.hbs_settings)

    def load_checkpoint(self, path) -> None:
        super().load_checkpoint(path)
        # Keep the jitter sequence aligned with training progress across a resume.
        self.noise_clock.value = self.step
