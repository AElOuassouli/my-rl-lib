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

## Algorithms

| Algorithm | Type | Module |
|---|---|---|
| SARSA | On-policy | `learning.on_policy.sarsa` |
| Expected SARSA | On-policy | `learning.on_policy.expected_sarsa` |
| n-step SARSA | On-policy | `learning.on_policy.n_step_sarsa` |
| Q-Learning | Off-policy | `learning.off_policy.q_learning` |
| Double Q-Learning | Off-policy | `learning.off_policy.double_q_learning` |
| n-step SARSA (off-policy) | Off-policy | `learning.off_policy.n_step_sarsa_off_policy` |

## Metrics & MLflow

The library ships with an optional metrics system. Handlers compute scalar or media metrics (e.g. heatmaps) and an MLflow backend logs them asynchronously without slowing down training. Videos are exported as GIF artifacts.

```python
from my_rl_lib.metrics import MetricsCollector, MetricType, MediaType, MLflowBackend

collector = MetricsCollector(
    track={
        MetricType.EPISODE_REWARD: 1,   # log every episode
        MetricType.EPISODE_STEPS: 1,
        MetricType.EPSILON: 100,        # log every 100 episodes
        MediaType.STATE_VISITATION_HEATMAP: 50,
    },
    backend=MLflowBackend(
        tracking_uri="mlruns",
        experiment_name="my_experiment",
        run_name="my_run",
    ),
    batch_size=100,
)

result = sarsa(..., metrics_collector=collector)
collector.close()
```

Then launch the MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
# or
make run-mlflow
```

Open `http://localhost:5000` to view runs, metrics, and image/GIF artifacts.

See [examples/](examples/) for complete runnable scripts.

## Project structure

```
src/my_rl_lib/
├── environments/   # Abstract environment + Windy Grid World
├── learning/
│   ├── on_policy/  # SARSA, Expected SARSA, n-step SARSA
│   └── off_policy/ # Q-Learning, Double Q-Learning, n-step off-policy SARSA
├── metrics/        # Collectors, handlers, MLflow backend
├── policies/       # Epsilon-greedy, Greedy
└── values/         # Q(s,a) storage and initialisation
```

## Disclaimer

This library is intentionally kept simple and is not optimised for speed or scale. It is a learning resource.

## License

See [LICENSE](LICENSE).
