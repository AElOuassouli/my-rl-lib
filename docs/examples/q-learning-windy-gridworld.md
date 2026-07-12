# Q-Learning Windy Grid World with MLflow

This example trains a Q-Learning agent (off-policy TD control) on the Windy
Grid World environment with comprehensive MLflow logging.

Key features demonstrated:

- Q-Learning off-policy training using the library's built-in algorithm
- MLflow parameter logging (hyperparameters)
- Per-episode scalar metrics (episode reward, steps)
- State visitation & value function heatmaps logged every 500 episodes
- A periodic greedy-agent animation (with a small `max_steps` cap so early,
  near-random rollouts stay cheap to render)
- End-of-training artifacts produced automatically by the collector at
  `close()`: a full greedy animation, a policy visualization, final scalars
  (`training_time`, `episodes_per_second`), and a registered model

All media is rendered in the backend worker process, off the training thread.

Run it with:

```bash
uv run python examples/q_learning_windy_gridworld_mlflow.py
```

## Source

```python
--8<-- "examples/q_learning_windy_gridworld_mlflow.py"
```
