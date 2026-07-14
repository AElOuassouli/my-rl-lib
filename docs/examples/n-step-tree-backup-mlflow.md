# N-Step Tree Backup Windy Grid World with MLflow

This example trains an N-Step Tree Backup agent (off-policy TD control) on the
Windy Grid World environment with MLflow logging.

Tree Backup is off-policy: a fixed epsilon-greedy *behavior* policy explores
the environment, while the algorithm learns a separate *target* policy that
is greedy with respect to the learned action-values. No importance sampling
is needed because the n-step return is built from the target policy's full
action distribution at each branch.

Key features demonstrated:

- N-Step Tree Backup off-policy training using the library's built-in algorithm
- Building and initializing an explicit epsilon-greedy behavior policy
- MLflow parameter logging (hyperparameters)
- Per-episode scalar metrics (episode reward, steps)
- State visitation & value function heatmaps logged periodically
- End-of-training artifacts produced automatically by the collector at
  `close()`: a greedy-agent animation, a policy visualization, final scalars
  (`training_time`, `episodes_per_second`), and a registered model

All media is rendered in the backend worker process, off the training thread.

Run it with:

```bash
uv run python examples/n_step_tree_backup_mlflow_example.py
```

## Source

```python
--8<-- "examples/n_step_tree_backup_mlflow_example.py"
```
