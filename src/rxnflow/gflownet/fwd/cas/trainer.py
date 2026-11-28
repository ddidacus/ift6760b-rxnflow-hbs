"""Trainer for CAS-Validity: one k-means index per synthon library.

The index never crosses a library boundary, so every retrieved row is legal for the
reaction that retrieved it.
"""

from __future__ import annotations

import logging
from time import perf_counter

import torch
from torch import nn

from rxnflow.gflownet.fwd.base import PROJECTION_ATTRIBUTE, StepClock
from rxnflow.gflownet.fwd.cas.index import ClusteredActionIndex
from rxnflow.gflownet.fwd.cas.policy import CASPolicy, CASSettings, build_indexes
from rxnflow.gflownet.fwd.trainer import MetricsRxnFlowTrainer
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel

logger = logging.getLogger(__name__)


class CASRxnFlowTrainer(MetricsRxnFlowTrainer):
    """Training with clustered action sampling instead of uniform subsampling."""

    def __init__(self, *args, cas: CASSettings | None = None, **kwargs):
        self.cas_settings = cas or CASSettings()
        self.cas_settings.validate()
        self.indexes: dict[str, ClusteredActionIndex] | None = None
        self.step_clock = StepClock()
        super().__init__(*args, **kwargs)

    def _prepare_models(self) -> None:
        # Fitted once and shared by both policies: the vectors are the environment's
        # fixed fingerprints, so there is nothing to keep in sync with training.
        started = perf_counter()
        self.indexes = build_indexes(self.env, self.cas_settings, self.device)
        dim = next(iter(self.indexes.values())).dim
        logger.info(
            "CAS clusters fitted in %.2fs over %d libraries",
            perf_counter() - started,
            len(self.indexes),
        )
        for model in (self.model, self.sampling_model):
            projection = nn.Linear(self.config.model.synthon_dim, dim, bias=False).to(
                self.device
            )
            nn.init.xavier_uniform_(projection.weight)
            model.add_module(PROJECTION_ATTRIBUTE, projection)
        with torch.no_grad():
            getattr(self.sampling_model, PROJECTION_ATTRIBUTE).weight.copy_(
                getattr(self.model, PROJECTION_ATTRIBUTE).weight
            )
        logger.info(
            "CAS query projection: %d -> %d (%d parameters), action subsampling off",
            self.config.model.synthon_dim,
            dim,
            self.config.model.synthon_dim * dim,
        )

    def _build_policy(self, model: RxnFlowModel) -> RxnFlowPolicy:
        assert self.indexes is not None
        return CASPolicy(
            self.env,
            model,
            self.config,
            self.device,
            self.rng,
            self.cas_settings,
            self.indexes,
            getattr(model, PROJECTION_ATTRIBUTE),
            self.step_clock,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        logger.info("CAS action selection: %s", self.cas_settings)
