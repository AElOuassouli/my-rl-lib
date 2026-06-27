"""
Example demonstrating metrics tracking in n-step SARSA.
"""

from my_rl_lib.environments.windy_grid_world import WindyGridWorld
from my_rl_lib.learning.on_policy.n_step_sarsa import n_step_sarsa
from my_rl_lib.metrics import MetricsCollector, MetricType
from my_rl_lib.values.initializer import InitializerType, Initializer


def main():
    # Create environment
    env = WindyGridWorld()

    initilializer = Initializer(
        initializer_type=InitializerType.UNIFORM,
        terminal_states_value=0.0,
        range_uniform_non_terminal=(-10.0, -1.0),
    )

    # Example 1: Fast training without metrics
    print("Example 1: Training without metrics (fast)")
    n_step_sarsa(
        environment=env,
        num_episodes=100,
        n=4,
        alpha=0.1,
        gamma=0.9,
        initializer=initilializer,
    )
    print("Training complete (no metrics tracked)\n")

    # Example 2: Training with metrics
    print("Example 2: Training with metrics")
    collector = MetricsCollector(
        track=[
            MetricType.EPISODE_REWARD,
            MetricType.EPISODE_STEPS,
            MetricType.TD_ERROR,
            MetricType.EPSILON,
        ]
    )

    n_step_sarsa(
        environment=env,
        num_episodes=100,
        n=4,
        alpha=0.1,
        gamma=0.9,
        initializer=initilializer,
        metrics_collector=collector,
    )

    # Analyze results
    print("\n=== Training Metrics ===")
    print(f"Episode Reward Summary: {collector.get_summary(MetricType.EPISODE_REWARD)}")
    print(f"Episode Steps Summary: {collector.get_summary(MetricType.EPISODE_STEPS)}")
    print(f"TD Error Summary: {collector.get_summary(MetricType.TD_ERROR)}")

    # Show last 5 episode rewards
    rewards = collector.get_metric(MetricType.EPISODE_REWARD)
    print(f"\nLast 5 episode rewards: {rewards[-5:]}")

    # Export for later analysis/visualization
    import json

    with open("training_metrics.json", "w") as f:
        json.dump(collector.export(), f, indent=2)
    print("\nMetrics exported to training_metrics.json")


if __name__ == "__main__":
    main()
