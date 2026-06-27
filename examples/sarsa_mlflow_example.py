"""
Example: SARSA with MLflow visualization.

This example demonstrates how to train a SARSA agent on the Windy Grid World
environment while logging metrics to MLflow for real-time visualization.
"""

from my_rl_lib.environments.windy_grid_world import RenderPolicyOptions, WindyGridWorld
from my_rl_lib.learning.on_policy.sarsa import sarsa
from my_rl_lib.metrics import MediaType, MetricsCollector, MetricType, MLflowBackend
from my_rl_lib.values.initializer import InitializerType, Initializer


def main():
    """Train SARSA with MLflow logging."""
    print("=" * 70)
    print("SARSA Training with MLflow Visualization")
    print("=" * 70)
    print()

    # Create environment
    env = WindyGridWorld()
    print(f"Environment: {env.__class__.__name__}")
    print()

    # Value function initializer
    initializer = Initializer(
        initializer_type=InitializerType.UNIFORM,
        terminal_states_value=0.0,
        range_uniform_non_terminal=(-10.0, -1.0),
    )

    import time

    # Training parameters
    num_episodes = 1000000
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.05

    run_name = (
        f"sarsa_windy_gridworld_{int(time.time())}_alpha{alpha}_gamma{gamma}_epsilon{epsilon}"
    )

    # Create metrics collector with MLflow backend
    print("Setting up MLflow logging...")
    collector = MetricsCollector(
        track={
            MetricType.EPISODE_REWARD: 1,  # Log every episode
            MetricType.EPISODE_STEPS: 1,  # Log every episode
            MetricType.EPSILON: 1000,  # Log every 1000 episodes
            MediaType.STATE_VISITATION_HEATMAP: 100,
        },
        batch_size=100,
        backend=MLflowBackend(
            tracking_uri="mlruns",
            experiment_name="sarsa_windy_gridworld",
            run_name=run_name,
            max_queue_size=1000,
        ),
    )
    print("  MLflow backend configured")
    print()

    print("Training parameters:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Exploration rate (ε): {epsilon}")
    print()

    # Train the agent
    print("Starting training...")
    print("-" * 70)

    result = sarsa(
        environment=env,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        initializer=initializer,
        metrics_collector=collector,
    )

    env.render_policy(
        result.values, options=[RenderPolicyOptions.GREEDY_AGENT, RenderPolicyOptions.VALUES]
    )

    print("-" * 70)
    print("Training complete!")
    print()

    # Close the collector to ensure all data is flushed
    collector.close()
    print("Metrics flushed to MLflow")
    print()

    # Print some statistics
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(
        f"Total states in value function: {len(result.values.values) if result.values.values else 0}"
    )
    print()

    # Instructions for viewing results
    print("=" * 70)
    print("View Training Metrics in MLflow")
    print("=" * 70)
    print()
    print("Run the following command in your terminal:")
    print()
    print("  make run-mlflow")
    print()
    print("Or directly:")
    print()
    print("  mlflow ui --backend-store-uri mlruns")
    print()
    print("Then open your browser to:")
    print()
    print("  http://localhost:5000")
    print()
    print("You will see:")
    print("  • Episode rewards (logged every episode)")
    print("  • Episode steps (logged every episode)")
    print("  • Exploration rate epsilon (logged every 1000 episodes)")
    print("  • State visitation heatmap (logged every 100 episodes as image artifact)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
