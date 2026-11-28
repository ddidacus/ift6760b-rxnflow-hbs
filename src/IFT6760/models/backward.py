"""Backward head: scores the one-step reverse candidates of a state."""

from __future__ import annotations

import torch
from torch import nn

from IFT6760.core.types import ActionType

from .nn import mlp


class BackwardHead(nn.Module):
    """Score edges with MLP([h(state), reaction, action type, synthon]).

    Embeddings are shared with the forward policy, as in SynFlowNet.
    The caller applies softmax over each state's candidates.
    """

    def __init__(
        self,
        state_dim: int,
        rxn_dim: int,
        synthon_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(len(ActionType), rxn_dim)
        self.mlp = mlp(
            state_dim + 2 * rxn_dim + synthon_dim,
            hidden_dim,
            1,
            num_layers,
            layernorm=True,
        )

    def forward(
        self,
        state: torch.Tensor,
        rxn: torch.Tensor,
        action_type: torch.Tensor,
        synthon: torch.Tensor,
    ) -> torch.Tensor:
        """One logit per candidate edge, [E]."""
        parts = [state, rxn, self.type_emb(action_type), synthon]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)

    def zero_output(self) -> None:
        """Initialize equal logits for a uniform backward policy."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
