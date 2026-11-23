"""Trainers for the two arms: upstream action subsampling, and HBS full-space lookup.

``MetricsRxnFlowTrainer`` is upstream's trainer plus the per-trajectory metric
collector, with upstream's action selection untouched. ``HBSRxnFlowTrainer`` replaces
subsampling with a hashed reservoir over fixed features: it builds that reservoir once,
before the first step, and never rebuilds it. The loss, the replay, the optimizer, the
schedule and the checkpoint format are upstream's in both.

The HBS arm adds exactly one tensor to the model -- the projection that puts the action
heads' query into the reservoir's feature space. It is registered on the model, so the
optimizer trains it, the EMA copy tracks it, and checkpoints carry it.
"""

from __future__ import annotations

import logging
from time import perf_counter

import torch
from torch import nn

from rxnflow import RxnFlowTrainer
from rxnflow.core.types import Trajectory
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel

from hbs.metrics import MetricsSettings, TrajectoryMetrics, format_summary
from hbs.policy import HBSPolicy, HBSSettings, InstrumentedPolicy, StepClock
from hbs.reservoir import HashedReservoir

logger = logging.getLogger(__name__)

PROJECTION_ATTRIBUTE = "hbs_query_projection"


class MetricsRxnFlowTrainer(RxnFlowTrainer):
    """Upstream training with per-trajectory diversity and latency metrics."""

    def __init__(self, *args, metrics: MetricsSettings | None = None, **kwargs):
        self.metrics_settings = metrics or MetricsSettings()
        self.metrics_settings.validate()
        self.trajectory_metrics: TrajectoryMetrics | None = None
        super().__init__(*args, **kwargs)

    def _build_policy(self, model: RxnFlowModel) -> RxnFlowPolicy:
        """The policy this arm trains with. Subclasses override this and nothing else."""
        return InstrumentedPolicy(self.env, model, self.config, self.device, self.rng)

    def _prepare_models(self) -> None:
        """Hook for an arm that has to add parameters before the optimizer is built."""

    def _setup_model(self) -> None:
        super()._setup_model()
        # Replacing the policies after upstream builds them keeps the models, their
        # EMA copy and the parameter count identical across arms. RxnFlowPolicy's
        # constructor draws no randomness, so the RNG stream is unaffected.
        self._prepare_models()
        self.policy = self._build_policy(self.model)
        self.sampling_policy = self._build_policy(self.sampling_model)
        if self.metrics_settings.every:
            self.trajectory_metrics = TrajectoryMetrics(
                self.env,
                self.metrics_settings,
                self.output_dir / "trajectory_metrics.jsonl",
            )
            # Online trajectories come from the EMA policy; the training policy is
            # only ever called for log-probabilities, which are not a sampling path.
            self.sampling_policy.attach_metrics(self.trajectory_metrics)
            logger.info(
                "Trajectory metrics every %d steps on %d trajectories "
                "(%d pairs per diversity estimate) -> %s",
                self.metrics_settings.every,
                self.metrics_settings.states,
                self.metrics_settings.max_pairs,
                self.trajectory_metrics.path,
            )

    def _write_samples(self, trajectories: list[Trajectory]) -> None:
        # Called once per step with this step's online trajectories, after the
        # training record is assembled, so self.step is the step being logged.
        super()._write_samples(trajectories)
        if self.trajectory_metrics is None:
            return
        summary = self.trajectory_metrics.flush(
            self.step,
            trajectories,
            extra={
                "probe_alignment": getattr(self.policy, "alignment", None),
                "code_bits_flipped": getattr(self.sampling_policy, "bits_flipped", None),
            },
        )
        if summary is not None and (
            self.step % self.config.training.log_every == 0 or "route_div" in summary
        ):
            logger.info("Step %d: %s", self.step, format_summary(summary))


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
        # The reservoir is built once, here, and shared by both policies: its vectors
        # are the environment's fixed fingerprints, so there is nothing to keep in
        # sync with training and no reason to hold two copies.
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
        # One projection from the action heads' output into the reservoir's space.
        # Registered on both models so the optimizer trains it, _update_ema tracks it
        # (it zips parameters, and this is appended last on both), and it is
        # checkpointed with the rest of the policy.
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
