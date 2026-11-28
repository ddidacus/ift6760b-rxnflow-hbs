"""Upstream training plus the per-trajectory metrics both full-space arms report.

An arm subclasses this and overrides ``_build_policy``, and ``_prepare_models`` if it
has to register parameters before the optimizer is built.
"""

from __future__ import annotations

import logging

import numpy as np

from rxnflow import RxnFlowTrainer
from rxnflow.core.types import Trajectory
from rxnflow.gflownet.fwd.base import InstrumentedPolicy
from rxnflow.gflownet.fwd.calibration import (
    calibration_metrics,
    format_calibration,
    scaled_log_rewards,
    trajectory_log_p_F,
)
from rxnflow.gflownet.fwd.metrics import (
    MetricsSettings,
    TrajectoryMetrics,
    format_summary,
)
from rxnflow.gflownet.policy import RxnFlowPolicy
from rxnflow.models import RxnFlowModel

logger = logging.getLogger(__name__)


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
        # Replacing the policies after upstream builds them keeps the models and the
        # parameter count identical across arms, and draws no randomness.
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
        extra = {
            "probe_alignment": getattr(self.policy, "alignment", None),
            "code_bits_flipped": getattr(self.sampling_policy, "bits_flipped", None),
            # CAS only, and diagnostic: the cell choice does not enter the loss.
            # cell_logp is -log P(opened cell), cell_entropy the whole distribution's,
            # which separates "committed to one cell" from "nlist is small".
            "cell_logp": getattr(self.policy, "cell_logp", None),
            "cell_entropy": getattr(self.policy, "cell_entropy", None),
            # CAS-FullSpace only: the share of retrieved rows that were executable.
            # CAS-Validity cannot report it, being 1.0 by construction.
            "legal_fraction": getattr(self.policy, "legal_fraction", None),
            "cells_walked": getattr(self.policy, "cells_walked", None),
            "fallback_rate": getattr(self.policy, "fallback_rate", None),
        }
        calibration = self._calibration(trajectories)
        if calibration is not None:
            extra.update(calibration)
        summary = self.trajectory_metrics.flush(self.step, trajectories, extra=extra)
        if summary is not None and (
            self.step % self.config.training.log_every == 0 or "route_div" in summary
        ):
            logger.info("Step %d: %s", self.step, format_summary(summary))
        if calibration is not None:
            logger.info("Step %d: %s", self.step, format_calibration(calibration))

    def _calibration(self, trajectories: list[Trajectory]) -> dict | None:
        """Reward-proportionality of this step's online sample, or None if not due.

        Costs one extra forward pass, so it runs on the metric cadence. Measured after
        the optimizer step, which is well inside the batch noise.
        """
        every = self.metrics_settings.every
        if not every or self.step % every != 0 or not trajectories:
            return None
        try:
            log_p_F = trajectory_log_p_F(self.policy, trajectories, self.config)
        except Exception as error:  # noqa: BLE001 - a diagnostic must not kill a run
            logger.warning("Calibration metric failed at step %d: %s", self.step, error)
            return None
        log_p_B = np.array(
            [sum(step.log_p_B for step in t.steps) for t in trajectories],
            dtype=np.float64,
        )
        return calibration_metrics(
            log_p_F=log_p_F,
            log_p_B=log_p_B,
            scaled_log_R=scaled_log_rewards(trajectories, self.config),
            rewards=np.array([t.reward for t in trajectories], dtype=np.float64),
            smiles=[t.final_smiles if t.valid else None for t in trajectories],
            valid=np.array([bool(t.valid) for t in trajectories], dtype=bool),
        )
