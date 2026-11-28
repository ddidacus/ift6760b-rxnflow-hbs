"""Partition estimation for a query-dependent candidate set.

Inclusion is not uniform, so every offered row carries its own importance weight:
declaring the query's bucket to be the whole library understates logZ by ~15.5 nats.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from rxnflow.gflownet.policy import ActionCategorical

logger = logging.getLogger(__name__)


@dataclass
class WeightedActionCategorical(ActionCategorical):
    """``ActionCategorical`` whose importance correction is per row, not per library.

    A bucket row was certain to be offered and a uniformly drawn row was not, so one
    scalar per library will not do. ``log_partition`` adds the weights; ``sample``
    does not, leaving the proposal as the policy's own preference over offered rows.
    """

    partition_weights: list[list[torch.Tensor]] = field(default_factory=list)
    explore_temp: float = 0.0

    def log_partition(self) -> torch.Tensor:
        if not self.partition_weights:
            return super().log_partition()
        partitions = []
        for state_logits, state_weights in zip(
            self.action_logits, self.partition_weights, strict=True
        ):
            assert state_logits
            weighted = torch.cat(
                [
                    entry.logits + weight
                    for entry, weight in zip(state_logits, state_weights, strict=True)
                ]
            )
            # Stable log-sum-exp with a finite floor when every logit is -inf,
            # as upstream's does.
            maximum = weighted.max().detach()
            maximum = torch.where(torch.isfinite(maximum), maximum, 0)
            total = (weighted - maximum).exp().sum()
            partitions.append(maximum + total.clamp_min(1e-38).log())
        return torch.stack(partitions)

    def sample(
        self, softmax_temperature: float, random_action_prob: float, importance: float
    ) -> list:
        if not self.partition_weights or self.explore_temp == 0.0:
            # log_importance is 0 for every library here, so the proposal is the raw
            # policy over the offered rows.
            return super().sample(softmax_temperature, random_action_prob, 0.0)
        # Fold the row weights into the proposal, then let upstream normalize.
        original = [list(entries) for entries in self.action_logits]
        try:
            for entries, weights in zip(
                self.action_logits, self.partition_weights, strict=True
            ):
                for entry, weight in zip(entries, weights, strict=True):
                    entry.logits = entry.logits + self.explore_temp * weight
            return super().sample(softmax_temperature, random_action_prob, 0.0)
        finally:
            for index, entries in enumerate(original):
                self.action_logits[index] = entries


class ComplementSampler:
    """Uniform draws over a library, with a reusable exclusion buffer.

    The buffer is written and wiped only at the excluded positions, so a draw costs
    O(|U| + |B|) rather than O(N); this runs a few hundred times a step.
    """

    def __init__(self, sizes: dict[str, int], rng: np.random.Generator):
        self.sizes = sizes
        self.rng = rng
        self._buffers: dict[str, NDArray[np.bool_]] = {}

    def draw(self, library_name: str, count: int) -> NDArray[np.int64]:
        """``count`` rows drawn uniformly without replacement from the whole library."""
        total = self.sizes[library_name]
        count = min(max(count, 0), total)
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if count == total:
            return np.arange(total, dtype=np.int64)
        rows = self.rng.choice(total, size=count, replace=False)
        rows.sort()
        return rows

    def weights_for(
        self,
        library_name: str,
        bucket: NDArray[np.int64],
        uniform: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Combined candidate rows and their per-row log importance weights.

        Bucket rows weigh 0, having been certain. Uniform rows outside the bucket each
        stand in for ``(N - |B|) / |U \\ B|`` rows and weigh the log of that.
        """
        total = self.sizes[library_name]
        if uniform.size == 0:
            rows = np.unique(bucket)
            return rows, np.zeros(len(rows), dtype=np.float32)

        buffer = self._buffers.get(library_name)
        if buffer is None:
            buffer = np.zeros(total, dtype=bool)
            self._buffers[library_name] = buffer
        buffer[bucket] = True
        try:
            fresh = uniform[~buffer[uniform]]
        finally:
            buffer[bucket] = False

        rest = total - np.unique(bucket).size
        if fresh.size == 0 or rest <= 0:
            rows = np.unique(bucket)
            return rows, np.zeros(len(rows), dtype=np.float32)

        log_importance = math.log(rest / fresh.size)
        rows = np.concatenate([np.unique(bucket), fresh])
        weights = np.concatenate(
            [
                np.zeros(np.unique(bucket).size, dtype=np.float32),
                np.full(fresh.size, log_importance, dtype=np.float32),
            ]
        )
        order = np.argsort(rows, kind="stable")
        return rows[order], weights[order]


def uniform_budget(num_actions: int, sampling_ratio: float, min_sampling: int) -> int:
    """Upstream's K for a library of this size.

    Matching it makes the uniform half exactly the baseline's estimator, so ``logZ``
    and step cost stay comparable between the arms.
    """
    return min(num_actions, max(min_sampling, int(num_actions * sampling_ratio)))
