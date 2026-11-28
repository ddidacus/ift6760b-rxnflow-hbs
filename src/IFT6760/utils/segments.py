"""Softmax, entropy and sampling over variable-size candidate sets.

A batch of decisions is stored as one flat tensor of scores with the candidates of each
decision contiguous. Candidate sets are tiny (a handful of reverse steps per molecule), so
the simplest correct representation is a padded dense matrix [decisions, max candidates].
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class Segments:
    """Layout of a flat score tensor split into decisions of `counts[i]` candidates."""

    def __init__(
        self, counts: Sequence[int] | Tensor, device: torch.device | str = "cpu"
    ):
        counts = torch.as_tensor(counts, dtype=torch.long, device=device)
        if counts.numel() == 0 or int(counts.min()) < 1:
            raise ValueError("every decision needs at least one candidate")
        self.counts = counts
        self.num = counts.numel()
        self.width = int(counts.max())
        self.size = int(counts.sum())
        self.row = torch.repeat_interleave(
            torch.arange(self.num, device=counts.device), counts
        )
        starts = torch.cumsum(counts, 0) - counts
        self.col = torch.arange(self.size, device=counts.device) - starts[self.row]
        self.mask = (
            torch.arange(self.width, device=counts.device)[None, :] < counts[:, None]
        )

    def dense(self, values: Tensor) -> Tensor:
        """[E] -> [N, W], padded with -inf (differentiable w.r.t. `values`)."""
        empty = values.new_full((self.num, self.width), float("-inf"))
        return empty.index_put((self.row, self.col), values)

    def log_softmax(self, values: Tensor) -> Tensor:
        """Per-decision log-probabilities as a dense [N, W] matrix (-inf on padding)."""
        return torch.log_softmax(self.dense(values), dim=1)

    def flat(self, dense: Tensor) -> Tensor:
        """[N, W] -> [E]: the entries that are real candidates."""
        return dense[self.row, self.col]

    def pick(self, logp: Tensor, index: Tensor) -> Tensor:
        """Log-probability of candidate `index[i]` in decision i: [N]."""
        return logp.gather(1, index[:, None]).squeeze(1)

    def entropy(self, logp: Tensor) -> Tensor:
        """Entropy of each decision's distribution, in nats: [N]."""
        safe = logp.masked_fill(~self.mask, 0.0)
        return -(safe.exp() * safe * self.mask).sum(1)

    @torch.no_grad()
    def sample(self, logp: Tensor) -> Tensor:
        """Draw one candidate per decision from `exp(logp)` (Gumbel-max): [N]."""
        uniform = torch.rand(logp.shape, device=logp.device)
        gumbel = -(-uniform.clamp_min(torch.finfo(logp.dtype).tiny).log()).log()
        return (logp + gumbel).argmax(dim=1)
