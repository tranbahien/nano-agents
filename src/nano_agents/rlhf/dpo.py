"""Direct Preference Optimization (Rafailov et al. 2023).

DPO derives a closed-form loss for the same objective RLHF optimizes,
but without a separately-learned reward model. The key identity is that
the optimal RLHF policy

    π*(y | x) = (1/Z(x)) π_ref(y | x) exp(r(y, x) / β)

can be solved for r(y, x):

    r(y, x) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x).

Plugging this into the Bradley-Terry preference model and noting that
log Z(x) cancels between winner and loser, the preference probability
becomes

    P(y_w ≻ y_l | x) = sigmoid( β [log π*(y_w|x)/π_ref(y_w|x)
                                  - log π*(y_l|x)/π_ref(y_l|x)] ).

Setting π_θ ← π* and minimizing the negative log-likelihood of preferences
under this model gives the DPO loss:

    L_DPO(θ) = -E[ log sigmoid( β [logπθ(y_w|x)/π_ref(y_w|x)
                                  - logπθ(y_l|x)/π_ref(y_l|x)] ) ].

It is exactly a binary classification objective on preference pairs.
No PPO loop, no reward model, no rollouts — just gradient descent.

For our tabular softmax policy this loss has a closed-form gradient
that's straightforward to derive.
"""

from __future__ import annotations

import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy


def _sigmoid(x):
    return np.where(x >= 0,
                     1.0 / (1.0 + np.exp(-x)),
                     np.exp(x) / (1.0 + np.exp(x)))


def dpo_loss_for_pair(policy: SoftmaxPolicy, ref_policy: SoftmaxPolicy,
                        context: int, winner: int, loser: int,
                        beta: float) -> float:
    """Per-pair DPO loss; positive scalar."""
    log_pi_w = float(np.log(policy.probs(context)[winner] + 1e-30))
    log_pi_l = float(np.log(policy.probs(context)[loser] + 1e-30))
    log_ref_w = float(np.log(ref_policy.probs(context)[winner] + 1e-30))
    log_ref_l = float(np.log(ref_policy.probs(context)[loser] + 1e-30))
    # Implicit "reward" margin between winner and loser.
    margin = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
    # NLL of "winner preferred" under Bradley-Terry with this margin.
    # = -log sigmoid(margin) = log(1 + exp(-margin)) = softplus(-margin)
    # Numerically stable:
    if margin >= 0:
        return float(np.log1p(np.exp(-margin)))
    else:
        return float(-margin + np.log1p(np.exp(margin)))


def dpo_loss(policy: SoftmaxPolicy, ref_policy: SoftmaxPolicy,
              preferences: list[tuple], beta: float) -> float:
    return float(np.mean(
        [dpo_loss_for_pair(policy, ref_policy, c, w, l, beta)
         for (c, w, l) in preferences]))


def _grad_log_pi_softmax(policy: SoftmaxPolicy, context: int, completion: int) -> np.ndarray:
    """Gradient of log pi_theta(completion | context) w.r.t. policy.theta.

    For softmax with logits θ[context, :]:
        d log π(a | s) / d θ[s, a'] = 1[a=a'] - π(a'|s)
    Only row `context` is nonzero.
    """
    p = policy.probs(context)
    g = np.zeros_like(policy.theta)
    g[context] = -p
    g[context, completion] += 1.0
    return g


def dpo_update_step(policy: SoftmaxPolicy, ref_policy: SoftmaxPolicy,
                      preferences: list[tuple], beta: float, lr: float):
    """One full-batch gradient descent step on the DPO loss.

    Vectorized: the policy-probability terms ∇log π(a|s) = e_a − π(·|s)
    cancel between winner and loser in (∇log π(w|c) − ∇log π(l|c)), so
    we only need indicator vectors. This is much faster than looping.

    Returns the mean DPO loss before the update (for monitoring).
    """
    if len(preferences) == 0:
        return 0.0
    contexts = np.array([p[0] for p in preferences])
    winners = np.array([p[1] for p in preferences])
    losers = np.array([p[2] for p in preferences])

    # log π(w|c) − log π(l|c) for both policy and ref.
    # We compute log p row-by-row only for the visited contexts.
    log_pi_diffs = np.zeros(len(preferences))
    log_ref_diffs = np.zeros(len(preferences))
    # Group by context to amortize the softmax computation.
    unique_contexts = np.unique(contexts)
    for c in unique_contexts:
        mask = contexts == c
        log_p_policy = np.log(policy.probs(int(c)) + 1e-30)
        log_p_ref = np.log(ref_policy.probs(int(c)) + 1e-30)
        log_pi_diffs[mask] = log_p_policy[winners[mask]] - log_p_policy[losers[mask]]
        log_ref_diffs[mask] = log_p_ref[winners[mask]] - log_p_ref[losers[mask]]

    margins = beta * (log_pi_diffs - log_ref_diffs)

    # Loss = mean of softplus(-margin) for monitoring.
    loss = float(np.mean(np.where(
        margins >= 0,
        np.log1p(np.exp(-margins)),
        -margins + np.log1p(np.exp(margins))
    )))

    # ∂L/∂margin per sample = sigmoid(margin) − 1.
    dL_dmargin = _sigmoid(margins) - 1.0

    # As derived in the comments above: the gradient of margin w.r.t.
    # θ[c, a] is β * (1[a = w] − 1[a = l]) (the policy probabilities
    # cancel). So we just accumulate at the winner and loser indices.
    grad = np.zeros_like(policy.theta)
    np.add.at(grad, (contexts, winners), beta * dL_dmargin)
    np.add.at(grad, (contexts, losers), -beta * dL_dmargin)
    grad /= len(preferences)
    policy.theta -= lr * grad
    return loss


def train_dpo(policy: SoftmaxPolicy, ref_policy: SoftmaxPolicy,
               preferences: list[tuple], n_steps: int = 500,
               lr: float = 0.5, beta: float = 0.1):
    """Train policy via DPO. Returns a list of per-step DPO losses."""
    losses = []
    for _ in range(n_steps):
        losses.append(dpo_update_step(policy, ref_policy, preferences,
                                         beta=beta, lr=lr))
    return losses
