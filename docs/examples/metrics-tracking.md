# Metrics Tracking

A minimal example showing n-step SARSA with a `MetricsCollector`. It
contrasts training with no metrics collector (fastest) against training with
a collector configured via the `track=` dict, which maps each `MetricType` to
how often (in episodes) it should be logged — e.g. `{MetricType.EPISODE_REWARD: 1}`
logs every episode, while a larger value logs periodically to keep overhead low.

Run it with:

```bash
uv run python examples/metrics_example.py
```

## Source

```python
--8<-- "examples/metrics_example.py"
```
