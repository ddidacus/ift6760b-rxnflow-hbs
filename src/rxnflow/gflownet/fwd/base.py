"""Step clock and metric hooks, shared by both full-space arms."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rxnflow.core.types import State
from rxnflow.gflownet.fwd.metrics import TrajectoryMetrics
from rxnflow.gflownet.policy import ActionCategorical, RxnFlowPolicy

# Name of the query projection registered on the model by the full-space trainers.
PROJECTION_ATTRIBUTE = "hbs_query_projection"


@dataclass
class Temperature:
    """A softmax temperature, optionally annealed from ``start`` to ``end``.

    Geometric in the step, since a temperature acts multiplicatively on logits.
    ``steps`` of 0, or equal endpoints, holds it constant.
    """

    start: float = 1.0
    end: float = 1.0
    steps: int = 0

    def validate(self, name: str) -> None:
        if self.start <= 0 or self.end <= 0:
            raise ValueError(f"{name} temperatures must be positive")
        if self.steps < 0:
            raise ValueError(f"{name} anneal steps must be non-negative")

    @property
    def annealed(self) -> bool:
        return self.steps > 0 and self.start != self.end

    def at(self, step: int) -> float:
        if not self.annealed:
            return self.start
        fraction = min(max(step / self.steps, 0.0), 1.0)
        return math.exp(
            math.log(self.start) * (1 - fraction) + math.log(self.end) * fraction
        )

    def __str__(self) -> str:
        if not self.annealed:
            return f"{self.start:g}"
        return f"{self.start:g}->{self.end:g} over {self.steps}"


class StepClock:
    """Counts sampling calls, so the probe jitter is a function of the step.

    The sampling policy advances it per batch and the training policy reads the same
    value, so both open the same buckets.
    """

    def __init__(self, value: int = 0):
        self.value = int(value)

    def advance(self) -> None:
        self.value += 1


class MetricsPolicyMixin:
    """Times lockstep sampling and exposes the candidate sets, changing nothing.

    A policy mixes this in and calls ``_record_candidates`` at the end of its
    ``get_action_categorical``. Inert until a collector is attached.
    """

    metrics: TrajectoryMetrics | None = None
    _measuring: list[State] | None = None

    def attach_metrics(self, collector: TrajectoryMetrics | None) -> None:
        self.metrics = collector

    def sample_from_model(self, *args, **kwargs):
        clock = getattr(self, "clock", None)
        if clock is not None:
            clock.advance()
        if self.metrics is None:
            return super().sample_from_model(*args, **kwargs)
        self.metrics.begin_sampling()
        try:
            return super().sample_from_model(*args, **kwargs)
        finally:
            self.metrics.end_sampling()

    def sample_actions(self, states: list[State], *args, **kwargs):
        if self.metrics is None:
            return super().sample_actions(states, *args, **kwargs)
        self.metrics.begin_iteration()
        # Identity, not equality: only the categorical built for this very call is
        # a sampling-time candidate set. log_prob() builds its own for the loss.
        self._measuring = states
        try:
            return super().sample_actions(states, *args, **kwargs)
        finally:
            self._measuring = None
            self.metrics.end_iteration()

    def _record_candidates(
        self, states: list[State], categorical: ActionCategorical
    ) -> None:
        if self.metrics is not None and self._measuring is states:
            self.metrics.record_candidates(states, categorical)


class InstrumentedPolicy(MetricsPolicyMixin, RxnFlowPolicy):
    """Upstream policy with the metric hooks and no behavioural change."""

    def get_action_categorical(
        self, states: list[State], *args, **kwargs
    ) -> ActionCategorical:
        categorical = super().get_action_categorical(states, *args, **kwargs)
        self._record_candidates(states, categorical)
        return categorical
