#!/usr/bin/env python3
"""
Hybrid planning demo: a learned world model + a symbolic safety filter.

A deliberately small 1-D example that makes the interfaces between
learning, planning and constraints concrete. It is NOT a deep world model,
a robot controller or a safety-certified system.

Pipeline
  1. Collect (state, action, next_state) transitions from a noisy environment.
  2. Fit a linear dynamics model by least squares and estimate its residual noise.
  3. Plan by enumerating short action sequences inside the learned model.
  4. Symbolic layer: reject imagined trajectories that leave the safe corridor,
     with a margin that grows with the model's estimated uncertainty.
  5. Execute only the first action, observe, and re-plan (receding horizon).

Run:  python world_model_hybrid_planning_demo.py
Needs: numpy
"""
from __future__ import annotations

from itertools import product
import json
import numpy as np

# ----------------------------------------------------------------------------
# Environment ("reality"): mildly nonlinear, noisy, so the learned model is
# imperfect by construction.
# ----------------------------------------------------------------------------
X_MIN, X_MAX = -1.0, 2.5      # symbolic rule: always keep x inside [X_MIN, X_MAX]
GOAL = 2.4                    # goal sits close to the upper boundary on purpose
ACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
NOISE_STD = 0.08


def true_step(x: float, u: float, rng: np.random.Generator) -> float:
    """Ground-truth dynamics, hidden from the agent."""
    return x + 0.55 * u - 0.04 * x * abs(x) + rng.normal(0.0, NOISE_STD)


# ----------------------------------------------------------------------------
# Learned world model: x' ~ a*x + b*u + c  (fit by least squares)
# ----------------------------------------------------------------------------
class LinearWorldModel:
    def __init__(self) -> None:
        self.theta = np.zeros(3)
        self.sigma = 0.0

    def fit(self, states, actions, next_states) -> "LinearWorldModel":
        features = np.column_stack([states, actions, np.ones_like(states)])
        self.theta, *_ = np.linalg.lstsq(features, next_states, rcond=None)
        residuals = next_states - features @ self.theta
        self.sigma = float(residuals.std())
        return self

    def predict(self, x: float, u: float) -> float:
        a, b, c = self.theta
        return a * x + b * u + c


def collect_experience(n: int, rng: np.random.Generator):
    """Random exploration inside the corridor."""
    xs = rng.uniform(X_MIN, X_MAX, n)
    us = rng.choice(ACTIONS, n)
    xn = np.array([true_step(x, u, rng) for x, u in zip(xs, us)])
    return xs, us, xn


# ----------------------------------------------------------------------------
# Symbolic safety rule and planners
# ----------------------------------------------------------------------------
def safe(x: float, margin: float = 0.0) -> bool:
    """Explicit rule: stay inside the corridor, shrunk by a margin."""
    return (X_MIN + margin) <= x <= (X_MAX - margin)


def plan(model, state, horizon=4, use_filter=True, k_sigma=2.0):
    """Exhaustive search over action sequences inside the learned model.

    With use_filter=True, an imagined trajectory is rejected as soon as a
    predicted state violates the rule. The margin is k_sigma * model sigma,
    so a less certain model plans more conservatively.
    """
    margin = k_sigma * model.sigma if use_filter else 0.0
    sequences = _sequences(horizon)                    # (n_sequences, horizon)
    a, b, c = model.theta
    x = np.full(len(sequences), float(state))
    cost = np.zeros(len(sequences))
    valid = np.ones(len(sequences), dtype=bool)
    for k in range(horizon):                           # roll every sequence forward
        u = sequences[:, k]
        x = a * x + b * u + c                          # imagined next state
        if use_filter:                                 # symbolic rule on each step
            valid &= (x >= X_MIN + margin) & (x <= X_MAX - margin)
        cost += (GOAL - x) ** 2 + 0.04 * u ** 2
    if not valid.any():                                # no safe plan: hold still
        return 0.0, float("inf")
    cost[~valid] = np.inf
    best = int(np.argmin(cost))
    return float(sequences[best, 0]), float(cost[best])


_SEQ_CACHE: dict[int, np.ndarray] = {}


def _sequences(horizon: int) -> np.ndarray:
    if horizon not in _SEQ_CACHE:
        _SEQ_CACHE[horizon] = np.array(list(product(ACTIONS, repeat=horizon)))
    return _SEQ_CACHE[horizon]


def reactive_policy(state, gain=1.0):
    """Baseline: no model, no planning. Move toward the goal."""
    desired = gain * (GOAL - state)
    return min(ACTIONS, key=lambda u: abs(u - desired))


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_episode(controller, rng, steps=30, x0=0.0):
    x, traj, violations = x0, [x0], 0
    for _ in range(steps):
        u = controller(x)
        x = true_step(x, u, rng)
        traj.append(x)
        violations += int(not safe(x))
    return np.array(traj), violations


def evaluate(controller_factory, n_episodes, seed):
    rng = np.random.default_rng(seed)
    controller = controller_factory()
    violated, final_err, all_traj = 0, [], []
    for _ in range(n_episodes):
        traj, v = run_episode(controller, rng)
        violated += int(v > 0)
        final_err.append(abs(GOAL - traj[-10:].mean()))
        all_traj.append(traj)
    return {
        "episodes_with_violation_pct": 100.0 * violated / n_episodes,
        "mean_goal_error_last10": float(np.mean(final_err)),
    }, np.array(all_traj)


def rollout_error_by_horizon(model, rng, horizon=12, n=400):
    """Open-loop imagined rollout vs. reality: shows compounding error."""
    errs = np.zeros(horizon)
    for _ in range(n):
        x_true = x_pred = rng.uniform(0.0, 1.5)
        seq = rng.choice(ACTIONS, horizon)
        for k, u in enumerate(seq):
            x_true = true_step(x_true, u, rng)
            x_pred = model.predict(x_pred, u)
            errs[k] += abs(x_true - x_pred)
    return (errs / n).tolist()


def main(seed: int = 0, n_episodes: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    model = LinearWorldModel().fit(*collect_experience(600, rng))
    print(f"Learned model  x' = {model.theta[0]:.3f}*x + {model.theta[1]:.3f}*u "
          f"+ {model.theta[2]:.3f}   (residual std {model.sigma:.3f})")

    controllers = {
        "Reactive baseline": lambda: reactive_policy,
        "Planner (no constraints)": lambda: (lambda x: plan(model, x, use_filter=False)[0]),
        "Hybrid planner (constraints)": lambda: (lambda x: plan(model, x, use_filter=True)[0]),
    }

    results, trajectories = {}, {}
    print(f"\n{'Agent':<30}{'Episodes w/ violation':>24}{'Goal error':>14}")
    for name, factory in controllers.items():
        metrics, trajs = evaluate(factory, n_episodes, seed=1)
        results[name], trajectories[name] = metrics, trajs
        print(f"{name:<30}{metrics['episodes_with_violation_pct']:>23.1f}%"
              f"{metrics['mean_goal_error_last10']:>14.3f}")

    err = rollout_error_by_horizon(model, np.random.default_rng(2))
    print("\nMean |imagined - real| by rollout step:",
          ", ".join(f"{e:.2f}" for e in err[:6]), "...")

    return {
        "model": {"theta": model.theta.tolist(), "sigma": model.sigma},
        "results": results,
        "rollout_error": err,
        "example_trajectories": {k: v[0].tolist() for k, v in trajectories.items()},
        "corridor": [X_MIN, X_MAX],
        "goal": GOAL,
    }


if __name__ == "__main__":
    out = main()
    with open("demo_results.json", "w") as fh:
        json.dump(out, fh)
