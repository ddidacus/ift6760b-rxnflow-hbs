"""MLE and REINFORCE for the learned P_B, following SynFlowNet.

TB consumes `traj_log_p_B` detached, so these losses are P_B's only gradient. The two are
independent objectives, never combined, and they differ in which trajectories they cover
(SynFlowNet Algorithms 1 and 2):

* MLE minimises `L_B` over the forward trajectories `{tau_F}` alone.
* REINFORCE minimises `J_B` over `{tau_B} union {tau_F}`: backward walks sampled from the
  current P_B, together with the batch's forward trajectories. A walk starts from a recorded
  terminal state (molecule *and* counters) drawn from the replay buffer and decrements them,
  so `R_B` is +1 exactly when it reaches the empty state and `illegal_reward` otherwise,
  plus an entropy bonus and no baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from IFT6760.core.types import Trajectory
from IFT6760.policies.backward import BackwardPolicy, Rollouts
from IFT6760.sampling.replay import ReplayBuffer
from IFT6760.training.config import BackwardConfig


@dataclass
class BackwardOutput:
    traj_log_p_B: torch.Tensor  # [num trajectories]
    aux_loss: torch.Tensor  # scalar, added to the TB loss
    info: dict[str, torch.Tensor] = field(default_factory=dict)


def rollout_rewards(cfg: BackwardConfig, rollouts: Rollouts) -> torch.Tensor:
    """+1 for a walk that reached the empty state, `illegal_reward` otherwise: [N]."""
    return torch.where(
        rollouts.success,
        torch.ones_like(rollouts.log_prob),
        torch.full_like(rollouts.log_prob, cfg.illegal_reward),
    )


@dataclass
class ForwardTrajectories:
    """The `{tau_F}` half of REINFORCE's union: the batch's forward trajectories.

    They reach s_0 by construction, so `R_B` is +1 for all of them and only their
    log-probability and entropy are needed here.
    """

    log_prob: torch.Tensor  # [N] sum of log P_B over each trajectory's observed edges
    entropy: torch.Tensor  # [M] entropy of P_B at each observed decision


def reinforce_loss(
    cfg: BackwardConfig,
    rollouts: Rollouts,
    forward: ForwardTrajectories | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`J_B = -mean(R_B * log P_B(tau))` over `{tau_B} union {tau_F}`, minus the entropy bonus.

    `forward` is the `{tau_F}` half and is omitted only when evaluating P_B on its own,
    where no forward trajectories exist.
    """
    rewards = rollout_rewards(cfg, rollouts)
    log_prob, entropies = rollouts.log_prob, [rollouts.entropy]
    if forward is not None:
        rewards = torch.cat([rewards, torch.ones_like(forward.log_prob)])
        log_prob = torch.cat([log_prob, forward.log_prob])
        entropies.append(forward.entropy)

    entropy = torch.cat(entropies)
    entropy = entropy.mean() if entropy.numel() else rewards.new_zeros(())
    loss = -(rewards.detach() * log_prob).mean() - cfg.entropy_weight * entropy
    info = {
        "bck_rollout_success": rollouts.success.float().mean().detach(),
        "bck_rollout_reward": rewards.mean().detach(),
        "bck_rollout_entropy": entropy.detach(),
    }
    return loss, info


class BackwardObjective:
    """Evaluates the learned P_B on a batch of trajectories and returns its training loss.

    `replay` and `rng` supply REINFORCE's `{tau_B}` start states, drawn from the replay
    buffer. One is drawn per trajectory
    in the batch, so the two halves of the union are equal.
    """

    def __init__(
        self,
        policy: BackwardPolicy,
        cfg: BackwardConfig,
        replay: ReplayBuffer | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.policy, self.cfg = policy, cfg
        self.replay, self.rng = replay, rng

    def _start_states(self, count: int) -> list[tuple[str, int, int]]:
        """Valid terminal states from the replay buffer: molecule and recorded counters.

        A walk needs the counters, not just the molecule: it decrements them, so reaching
        the empty state means the route it found spends exactly what the recorded one did.
        """
        if self.replay is None or self.rng is None:
            return []
        starts = []
        for trajectory in self.replay.sample(count, self.rng):
            if not (trajectory.valid and trajectory.final_smiles):
                continue
            costs = [step.action.action_type.cost for step in trajectory.steps]
            starts.append(
                (
                    trajectory.final_smiles,
                    sum(cost[0] for cost in costs),
                    sum(cost[1] for cost in costs),
                )
            )
        return starts

    def __call__(self, trajectories: list[Trajectory]) -> BackwardOutput:
        policy, cfg = self.policy, self.cfg
        device = policy.device
        num_trajectories = len(trajectories)
        zero = torch.zeros((), device=device)

        observed = policy.observed(trajectories)
        decisions = observed.decisions
        if len(decisions) == 0:
            return BackwardOutput(torch.zeros(num_trajectories, device=device), zero)

        counts = torch.tensor([len(e) for e in decisions.edges], device=device).float()
        if cfg.objective == "none":
            # Frozen control: uniform over the candidates, so log P_B = -log|C| and
            # the entropy is log|C|. Constant, hence no gradient and nothing to train.
            logp_observed, entropy, accuracy = -counts.log(), counts.log(), zero
        else:
            logp, seg = policy.log_probs(decisions)
            logp_observed = seg.pick(logp, observed.observed)
            entropy = seg.entropy(logp)
            accuracy = (logp.argmax(dim=1) == observed.observed).float().mean()

        traj_log_p_B = torch.zeros(num_trajectories, device=device).index_add(
            0, observed.traj_index, logp_observed
        )

        info = {
            "bck_nll": -logp_observed.mean().detach(),
            "bck_p_observed": logp_observed.exp().mean().detach(),
            "bck_uniform_nll": counts.log().mean().detach(),
            "bck_acc": accuracy.detach(),
            "bck_entropy": entropy.mean().detach(),
            "bck_n_cand": counts.mean().detach(),
        }

        terms = []
        valid = torch.tensor([t.valid for t in trajectories], device=device)
        if cfg.uses_mle and valid.any():
            # Only valid trajectories have a molecule whose route P_B could have taken.
            mle = -traj_log_p_B[valid].mean()
            terms.append(cfg.mle_weight * mle)
            info["bck_mle_loss"] = mle.detach()
        if cfg.uses_reinforce:
            # J_B covers {tau_B} union {tau_F}, as in SynFlowNet's Algorithm 2. With no
            # replay buffer there are no start states and this term is skipped.
            start = self._start_states(num_trajectories)
            if start:
                loss, rollout_info = reinforce_loss(
                    cfg,
                    policy.rollout(start),
                    ForwardTrajectories(traj_log_p_B[valid], entropy),
                )
                terms.append(cfg.reinforce_weight * loss)
                info["bck_reinforce_loss"] = loss.detach()
                info.update(rollout_info)

        aux = torch.stack(terms).sum() if terms else zero
        return BackwardOutput(traj_log_p_B, aux, info)
