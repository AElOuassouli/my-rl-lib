"""
Example: Q-Learning on Windy Grid World with MLflow tracking.

This example demonstrates how to train a Q-Learning agent (off-policy TD control)
on the Windy Grid World environment with comprehensive MLflow logging.

Key features demonstrated:
- Q-Learning off-policy training using the library's built-in algorithm
- MLflow parameters logging (hyperparameters)
- MLflow metrics logging (episode rewards, steps, epsilon)
- State visitation heatmap images logged every 500 episodes
- Value function heatmap images logged every 500 episodes
- GIF animation saved as artifact
"""

import time

import mlflow

from my_rl_lib.environments.windy_grid_world import WindyGridWorld
from my_rl_lib.learning.off_policy.q_learning import q_learning
from my_rl_lib.metrics import (
    AnimationConfig,
    ArtifactType,
    MediaType,
    MetricsCollector,
    MetricType,
    MLflowBackend,
    PolicyVizConfig,
    TrainedModelConfig,
)
from my_rl_lib.values.initializer import Initializer, InitializerType


def main():
    """Train Q-Learning agent on Windy Grid World with MLflow logging."""
    print("=" * 70)
    print("Q-Learning Training on Windy Grid World with MLflow")
    print("=" * 70)
    print()

    # ========================================================================
    # Training Parameters
    # ========================================================================
    num_episodes = 50000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    print("Training configuration:")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Exploration rate (ε): {epsilon}")
    print()

    # ========================================================================
    # Environment Setup
    # ========================================================================
    print("Creating Windy Grid World environment...")
    env = WindyGridWorld()
    print(f"  Grid size: {env.grid_height}x{env.grid_width}")
    print(f"  Start cell: {env.start_cell}")
    print(f"  Goal cell: {env.goal_cell}")
    print(f"  Wind columns: {env.wind_strength_by_column}")
    print()

    # ========================================================================
    # Value Function Initializer
    # ========================================================================
    print("Setting up value function initializer...")
    initializer = Initializer(
        initializer_type=InitializerType.UNIFORM,
        terminal_states_value=0.0,
        range_uniform_non_terminal=(-10.0, -1.0),
    )
    print("  Using uniform initialization for Q(s,a)")
    print()

    # ========================================================================
    # MLflow Backend + Metrics Collector
    # ========================================================================
    print("Setting up MLflow logging...")

    run_name = (
        f"q_learning_windy_gridworld_{int(time.time())}_"
        f"ep{num_episodes}_alpha{alpha}_gamma{gamma}_epsilon{epsilon}"
    )

    backend = MLflowBackend(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="q_learning_windy_gridworld",
        run_name=run_name,
    )

    collector = MetricsCollector(
        track={
            MetricType.EPISODE_REWARD: 1,
            MetricType.EPISODE_STEPS: 1,
            MediaType.STATE_VISITATION_HEATMAP: 500,
            MediaType.VALUE_FUNCTION_HEATMAP: 500,
        },
        track_at_end={
            # All produced once at close(), rendered in the backend worker
            # process (off the training thread) — no post-training code needed.
            MediaType.GREEDY_AGENT_ANIMATION: AnimationConfig(number_episodes=3, fps=10),
            MediaType.POLICY_VISUALIZATION: PolicyVizConfig(),
            ArtifactType.TRAINED_MODEL: TrainedModelConfig(model_name="windy_gridworld_q_values"),
            MetricType.TRAINING_TIME: None,
            MetricType.EPISODES_PER_SECOND: None,
        },
        backend=backend,
        # batch_size auto-resolved to 1 from MLflowBackend.preferred_batch_size
    )

    # Log hyperparameters — run is active after MLflowBackend creation
    mlflow.log_params(
        {
            "num_episodes": num_episodes,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "grid_height": env.grid_height,
            "grid_width": env.grid_width,
            "start_cell": str(env.start_cell),
            "goal_cell": str(env.goal_cell),
        }
    )

    print("  MLflow experiment: q_learning_windy_gridworld")
    print(f"  Run name: {run_name}")
    print(f"  Run ID: {backend._run_id}")
    print("  Tracking URI: sqlite:///mlflow.db")
    print("  Metrics collector configured")
    print()

    # ========================================================================
    # Training
    # ========================================================================
    print("Starting Q-Learning training...")
    print("-" * 70)

    start_time = time.time()

    result = q_learning(
        environment=env,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        initializer=initializer,
        metrics_collector=collector,
    )

    training_time = time.time() - start_time

    print("-" * 70)
    print(f"Training completed in {training_time:.2f} seconds")
    print()
    print(
        "Final artifacts (animation, policy visualization, trained model, and "
        "final scalars) are produced automatically by the collector at close()."
    )
    print()

    # ========================================================================
    # Statistics
    # ========================================================================
    print("=" * 70)
    print("Training Statistics")
    print("=" * 70)

    num_states = len(result.values.values) if result.values.values else 0
    print(f"Number of states in value function: {num_states}")

    total_action_state_pairs = (
        sum(len(av) for av in result.values.values.values()) if result.values.values else 0
    )
    print(f"Total (state, action) pairs: {total_action_state_pairs}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Episodes per second: {num_episodes / training_time:.1f}")
    print()

    # ========================================================================
    # MLflow Viewing Instructions
    # ========================================================================
    print("=" * 70)
    print("View Training Metrics in MLflow")
    print("=" * 70)
    print()
    print("To view the training metrics, visualizations, and artifacts:")
    print()
    print("  1. Start MLflow UI:")
    print()
    print("     mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print()
    print("  2. Open your browser to: http://localhost:5001")
    print()
    print("What you will see:")
    print("  ✓ Parameters: num_episodes, alpha, gamma, epsilon, grid dimensions")
    print("  ✓ Metrics: episode_reward, episode_steps, epsilon (over time)")
    print("  ✓ Images: state visitation heatmap, value function heatmap")
    print("  ✓ Artifacts: animations/final_agent_animation.gif, plots/final_policy.png")
    print("  ✓ Registered model: windy_gridworld_q_values (Model Registry)")
    print()
    print(f"  Run name: {run_name}")
    print("=" * 70)

    # Flush remaining batched data, shut down async worker, and end MLflow run
    collector.close()


if __name__ == "__main__":
    main()
