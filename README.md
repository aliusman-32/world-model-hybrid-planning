# World Model + Symbolic Safety Filter: A Minimal Hybrid Planning Demo

Companion code for the article **"From Predicting the Next Token To Predicting the World"**.

📄 **Article:** [https://medium.com/@aliusman3230/from-predicting-the-next-token-to-predicting-the-world-c9d54d41c8ce]

---

## What this is

A deliberately small 1-D example that makes the interfaces between **learning**, **planning** and **explicit constraints** concrete.

An agent must reach a goal that sits close to a hard safety boundary. It does this using a learned model of its environment, and a symbolic rule keeps its plans inside the safe region.

> **This is an illustration, not a real system.** It is not a deep world model, a robot controller, or a safety-certified implementation.

## The idea in one paragraph

A learned world model gives flexible prediction. Planning inside that model gives foresight. A symbolic rule gives a hard boundary that does not depend on the model being right. The link between the learned and symbolic parts is a safety margin that grows with the model's estimated uncertainty.

## Pipeline

1. **Collect** `(state, action, next_state)` transitions from a noisy environment.
2. **Learn** a linear dynamics model, `x' ≈ a·x + b·u + c`, by least squares, and estimate its residual noise (`sigma`).
3. **Plan** by enumerating short action sequences inside the learned model.
4. **Filter** with a symbolic rule: reject imagined trajectories that leave the safe corridor, using a margin of `k_sigma × sigma`, so a less certain model plans more conservatively.
5. **Act and re-plan** (receding horizon): execute only the first action, observe the real outcome, and plan again.

## What gets compared

Three controllers are evaluated over many episodes:

| Controller | Description |
|---|---|
| Reactive baseline | No model, no planning. Moves toward the goal. |
| Planner (no constraints) | Plans in the learned model and trusts it fully. |
| Hybrid planner (constraints) | Plans in the learned model with the symbolic safety filter. |

Metrics reported:
- percentage of episodes with at least one corridor violation
- mean distance to the goal over the last 10 steps
- open-loop **rollout error by horizon**, which shows how the model's imagined predictions drift from reality the further ahead it looks

## Setup in the environment

The environment is mildly nonlinear and noisy, and hidden from the agent, so the learned linear model is imperfect by construction. The goal (2.4) is close to the upper boundary (2.5) on purpose.

## Getting started

**Requirements:** Python 3.9+ and `numpy`.

```bash
pip install numpy
```

**Run the notebook** (recommended while reading the article):

```bash
pip install jupyter
jupyter notebook world_model_hybrid_planning_demo.ipynb
```

**Or run the script:**

```bash
python world_model_hybrid_planning_demo.py
```

Both print a results table and write `demo_results.json` (learned parameters, metrics, rollout error and an example trajectory per controller).

## Results

Run the demo to reproduce the numbers; they depend on the random seed (default `seed=0`).


### Learned model

| Parameter | Value |
|---|---|
| Dynamics | `x' = 0.925·x + 0.551·u + 0.001` |
| Residual std (σ) | 0.087 |

### Controller comparison

| Agent | Episodes with violation | Goal error (last 10 steps) |
|---|---:|---:|
| Reactive baseline | 0.0% | 0.295 |
| Planner (no constraints) | 92.7% | 0.062 |
| Hybrid planner (constraints) | 2.3% | 0.236 |


## Limitations

Please read these before drawing conclusions.

- **Not a safety guarantee.** The filter checks the model's *imagined* states, and the margin is derived from one-step residual error. Errors compound over a rollout, and real states can still leave the corridor.
- **Toy scale.** One dimension, a linear model and exhaustive search over a small action set. It shows the structure of the approach, not how it scales to high-dimensional or learned deep models.
- **Hand-written rule.** The safety rule here is a single interval. Real systems need richer, verified specifications.
- **Single seed by default.** For anything beyond illustration, run multiple seeds and report variance.

## Repository contents

```
.
├── README.md
├── world_model_hybrid_planning_demo.ipynb   # notebook version
├── world_model_hybrid_planning_demo.py      # script version
└── demo_results.json                        # generated when you run the demo
```

## Citation

If you use or reference this code, please cite the article:

```
Ali Usman. "From Predicting the Next Token To Predicting the World:
World Models, Planning, and the Case for Hybrid AI." - 2026. ```
[https://medium.com/@aliusman3230/from-predicting-the-next-token-to-predicting-the-world-c9d54d41c8ce]
