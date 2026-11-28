"""Is the terminal distribution proportional to the reward?

Rank correlation and the slope of log P(x) against beta * log R(x). Neither the loss
nor ``tb_error`` reports this: both are residuals against the policy's own partition,
so a policy normalizing over the wrong action set can look converged and be mis-ranked.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from rxnflow.gflownet.trajectory_balance import scalarize_log_rewards

if TYPE_CHECKING:
    from rxnflow.config import Config
    from rxnflow.core.types import Trajectory
    from rxnflow.gflownet.policy import RxnFlowPolicy

logger = logging.getLogger(__name__)


def spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    """Tie-aware Spearman rank correlation, or None when one input is constant.

    None rather than 0.0, which would read as "disagrees" instead of "no information".
    """
    if x.size < 3:
        return None
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ols_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) of y on x, or None when x is constant."""
    if x.size < 3 or x.std() == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def scaled_log_rewards(
    trajectories: list[Trajectory], config: Config
) -> NDArray[np.float64]:
    """``beta * log R`` for each trajectory, exactly as trajectory balance forms it.

    Not taken from the loss, which also covers replay: the claim is about a fresh draw
    from the sampling policy, and off-policy replay would bias it.
    """
    floor = config.training.reward_floor
    beta = np.array([t.beta for t in trajectories], dtype=np.float64)
    preference = np.array([t.preference for t in trajectories], dtype=np.float64)
    objective_rewards = np.array(
        [t.objective_rewards for t in trajectories], dtype=np.float64
    ).clip(min=floor)
    property_rewards = np.array(
        [t.property_reward for t in trajectories], dtype=np.float64
    ).clip(min=floor)
    log_R = scalarize_log_rewards(
        objective_rewards, config.reward.moo_scalarization, preference
    )
    log_R += np.log(property_rewards)
    log_R = np.clip(log_R, a_min=math.log(floor), a_max=None)
    return beta * log_R


@torch.no_grad()
def trajectory_log_p_F(
    policy: RxnFlowPolicy,
    trajectories: list[Trajectory],
    config: Config,
    cond: torch.Tensor | None = None,
) -> NDArray[np.float64]:
    """``sum log P_F`` per trajectory, through the policy's own normalization.

    Each arm is measured under its own normalization, since the question is whether
    its induced terminal distribution tracks the reward.
    """
    device = policy.device
    beta = torch.tensor([t.beta for t in trajectories], dtype=torch.float32, device=device)
    preference = torch.tensor(
        np.array([t.preference for t in trajectories], dtype=np.float32), device=device
    )

    transitions = []
    traj_indices: list[int] = []
    for index, trajectory in enumerate(trajectories):
        for transition in trajectory.steps:
            transitions.append(transition)
            traj_indices.append(index)
    totals = torch.zeros(len(trajectories), device=device)
    if not transitions:
        return totals.cpu().numpy().astype(np.float64)

    batch_idx = torch.tensor(traj_indices, dtype=torch.long, device=device)
    log_p_F = policy.log_prob(
        [step.state for step in transitions],
        [step.action for step in transitions],
        beta[batch_idx],
        preference[batch_idx],
        cond=cond[batch_idx] if cond is not None else None,
    )
    totals.index_add_(0, batch_idx, log_p_F)
    return totals.cpu().numpy().astype(np.float64)


def calibration_metrics(
    log_p_F: NDArray[np.float64],
    log_p_B: NDArray[np.float64],
    scaled_log_R: NDArray[np.float64],
    rewards: NDArray[np.float64],
    smiles: list[str | None],
    valid: NDArray[np.bool_],
) -> dict[str, Any]:
    """Reward-proportionality diagnostics for one batch of online trajectories."""
    keep = valid & np.isfinite(log_p_F) & np.isfinite(scaled_log_R)
    out: dict[str, Any] = {"calib_n": int(keep.sum())}
    if keep.sum() < 3:
        return out

    lp, lb, slr, rew = log_p_F[keep], log_p_B[keep], scaled_log_R[keep], rewards[keep]
    kept_smiles = [s for s, k in zip(smiles, keep, strict=True) if k]

    # Fraction of the batch sharing the modal reward. On a sparse task this is the
    # fraction at the floor, and it is what limits the correlation's resolution.
    _, counts = np.unique(np.round(rew, 12), return_counts=True)
    out["calib_reward_ties"] = float(counts.max() / rew.size)

    out["logR_spearman"] = spearman(lp, rew)
    out["logR_pearson"] = pearson(lp, slr)
    fit = ols_slope(slr, lp)
    if fit is not None:
        out["logR_slope"], _ = fit
    # The exact TB identity: log P_F - log P_B against beta log R has slope 1 and
    # intercept -log Z, so the intercept is an independent read on logZ.
    fit_tb = ols_slope(slr, lp - lb)
    if fit_tb is not None:
        out["logR_slope_tb"], intercept = fit_tb
        out["logR_implied_logZ"] = -intercept

    # One row per distinct SMILES, keeping its best reward and largest log P_F:
    # closer to log P(x), and without the weighting a resampled molecule would get.
    best: dict[str, tuple[float, float]] = {}
    for smi, p, r in zip(kept_smiles, lp, rew, strict=True):
        if smi is None:
            continue
        prev = best.get(smi)
        if prev is None or r > prev[1] or (r == prev[1] and p > prev[0]):
            best[smi] = (max(p, prev[0]) if prev else p, max(r, prev[1]) if prev else r)
    if len(best) >= 3:
        u_lp = np.array([v[0] for v in best.values()], dtype=np.float64)
        u_rew = np.array([v[1] for v in best.values()], dtype=np.float64)
        out["calib_n_unique"] = len(best)
        out["logR_spearman_unique"] = spearman(u_lp, u_rew)

    out["logp_mean"] = float(lp.mean())
    out["logp_std"] = float(lp.std())
    return out


def format_calibration(metrics: dict[str, Any]) -> str:
    """One-line summary for training.log."""
    parts = []
    for key, label, fmt in (
        ("logR_spearman", "spearman(logP,R)", "{:+.3f}"),
        ("logR_spearman_unique", "uniq", "{:+.3f}"),
        ("logR_slope_tb", "TB slope", "{:.3f}"),
        ("logR_implied_logZ", "implied logZ", "{:+.2f}"),
        ("calib_reward_ties", "ties", "{:.0%}"),
    ):
        value = metrics.get(key)
        if value is not None:
            parts.append(f"{label} {fmt.format(value)}")
    return "; ".join(parts) if parts else "calibration undefined (constant reward)"
