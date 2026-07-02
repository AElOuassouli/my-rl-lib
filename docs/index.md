# my-rl-lib

A minimal reinforcement learning toolkit built for learning purposes. This library prioritises clarity and readability over performance — it is a hands-on companion to Sutton & Barto's *Reinforcement Learning: An Introduction*.

## Features

- **On-policy algorithms** — SARSA, Expected SARSA, n-step SARSA
- **Off-policy algorithms** — Q-Learning, Double Q-Learning, n-step off-policy SARSA (with importance sampling)
- **Environments** — Abstract base class + Windy Grid World
- **Policies** — Epsilon-greedy, Greedy
- **Value functions** — Action-state values Q(s, a) with configurable initialisation strategies
- **Metrics system** — Modular collection with optional MLflow backend and in-memory storage

## Installation

```bash
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Quick start

```python
from my_rl_lib.environments.windy_grid_world import WindyGridWorld
from my_rl_lib.learning.on_policy.sarsa import sarsa
from my_rl_lib.values.initializer import Initializer, InitializerType

env = WindyGridWorld()

initializer = Initializer(
    initializer_type=InitializerType.CONSTANT,
    terminal_states_value=0.0,
    constant_value_non_terminal=0.0,
)

result = sarsa(
    environment=env,
    num_episodes=500,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
    initializer=initializer,
)

# result.values  — learned Q(s, a)
# result.policy  — learned epsilon-greedy policy
```

## Where to go next

- [Getting Started](getting-started.md) — installation, a full quick-start walkthrough, and the metrics/MLflow system.
- [API Reference](api/index.md) — auto-generated reference for every package, generated from the source docstrings.
- [Examples](examples/index.md) — complete runnable scripts, embedded directly from `examples/`.

## Disclaimer

This library is intentionally kept simple and is not optimised for speed or scale. It is a learning resource.
