"""
Example: N-Step Tree Backup on Windy Grid World with MLflow tracking.

This example demonstrates how to train an N-Step Tree Backup agent (off-policy
TD control) on the Windy Grid World environment with MLflow logging.

Tree Backup is off-policy: an epsilon-greedy *behavior* policy explores the
environment, while the algorithm learns a separate *target* policy that is
greedy with respect to the learned action-values. No importance sampling is
needed because the n-step return is built from the target policy's full
action distribution at each branch (see n_step_tree_backup for details).
The behavior policy is updated in-place from the same learned action-values
as training progresses, so exploration gets sharper over time (episode 1 is
an uninformed random-ish walk that can take thousands of steps on this grid;
it converges to near-optimal within roughly a hundred episodes).

Key features demonstrated:
- N-Step Tree Backup off-policy training using the library's built-in algorithm
- Building and initializing an explicit epsilon-greedy behavior policy
- MLflow parameters logging (hyperparameters)
- Per-episode scalar metrics (episode reward, steps)
- State visitation & value function heatmaps logged periodically
- End-of-training artifacts produced automatically by the collector at close():
  a greedy-agent animation, a policy visualization, final scalars
  (training_time, episodes_per_second), and a registered model

All media is rendered in the backend worker process, off the training thread.
"""

import time

import mlflow

from my_rl_lib.environments.windy_grid_world import WindyGridWorld
from my_rl_lib.learning.off_policy.n_step_tree_backup import n_step_tree_backup
from my_rl_lib.metrics import (
    AgentAnimationHandler,
    AnimationConfig,
    ArtifactType,
    MediaType,
    MetricCollectionSettings,
    MetricsCollector,
    MetricType,
    MLflowBackend,
    PolicyVizConfig,
    TrainedModelConfig,
)
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer, InitializerType


def main():
    """Train N-Step Tree Backup agent on Windy Grid World with MLflow logging."""
    print("=" * 70)
    print("N-Step Tree Backup Training on Windy Grid World with MLflow")
    print("=" * 70)
    print()

    # ========================================================================
    # Training Parameters
    # ========================================================================
    num_episodes = 50000
    n = 3
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    print("Training configuration:")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  N-step: {n}")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Discount factor (γ): {gamma}")
    print(f"  Behavior policy exploration rate (ε): {epsilon}")
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
    # Behavior Policy
    # ========================================================================
    # Tree Backup is off-policy: the behavior policy only needs to explore
    # (non-zero probability everywhere). The initializer below only seeds its
    # starting values — n_step_tree_backup keeps it in sync with the learned
    # action-values as training progresses, so its exploration sharpens over
    # time even though the target policy it's evaluated against stays greedy.
    print("Setting up epsilon-greedy behavior policy...")
    behavior_values = ActionStateValues()
    behavior_values.init_from_environment(environment=env, initializer=initializer)
    behavior_policy = EpsilonGreedy(epsilon=epsilon)
    behavior_policy.init_from_environment_and_values(environment=env, values=behavior_values)
    print(f"  Behavior policy: epsilon-greedy (epsilon={epsilon})")
    print()

    # ========================================================================
    # MLflow Backend + Metrics Collector
    # ========================================================================
    print("Setting up MLflow logging...")

    run_name = (
        f"n_step_tree_backup_windy_gridworld_{int(time.time())}_"
        f"ep{num_episodes}_n{n}_alpha{alpha}_gamma{gamma}_epsilon{epsilon}"
    )

    backend = MLflowBackend(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="n_step_tree_backup_windy_gridworld",
        run_name=run_name,
        max_queue_size=500,
    )

    collector = MetricsCollector(
        track={
            # Simple items: an int is shorthand for
            # MetricCollectionSettings(frequency=n).
            MetricType.EPISODE_REWARD: 1,
            MetricType.EPISODE_STEPS: 1,
            # Heatmaps every 500 episodes (rendered in the worker process).
            MediaType.STATE_VISITATION_HEATMAP: 500,
            MediaType.VALUE_FUNCTION_HEATMAP: 500,
            # Periodic greedy-agent animation with a per-metric handler override:
            # a single short clip with a small max_steps cap so early-training
            # rollouts (near-random greedy policy) stay fast to render.
            MediaType.GREEDY_AGENT_ANIMATION: MetricCollectionSettings(
                frequency=10000,
                handler=AgentAnimationHandler(
                    AnimationConfig(number_episodes=1, fps=10, max_steps=50)
                ),
            ),
        },
        track_at_end={
            # All produced once at close(), rendered in the backend worker
            # process (off the training thread) — no post-training code needed.
            # The final policy has converged, so the full-length rollout is cheap.
            MediaType.GREEDY_AGENT_ANIMATION: AnimationConfig(number_episodes=3, fps=10),
            MediaType.POLICY_VISUALIZATION: PolicyVizConfig(),
            ArtifactType.TRAINED_MODEL: TrainedModelConfig(
                model_name="windy_gridworld_tree_backup_values"
            ),
            MetricType.TRAINING_TIME: None,
            MetricType.EPISODES_PER_SECOND: None,
        },
        backend=backend,
        batch_size=100,
    )

    # Log hyperparameters — run is active after MLflowBackend creation
    mlflow.log_params(
        {
            "num_episodes": num_episodes,
            "n": n,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "grid_height": env.grid_height,
            "grid_width": env.grid_width,
            "start_cell": str(env.start_cell),
            "goal_cell": str(env.goal_cell),
        }
    )

    print("  MLflow experiment: n_step_tree_backup_windy_gridworld")
    print(f"  Run name: {run_name}")
    print(f"  Run ID: {backend._run_id}")
    print("  Tracking URI: sqlite:///mlflow.db")
    print("  Metrics collector configured")
    print()

    # ========================================================================
    # Training
    # ========================================================================
    print("Starting N-Step Tree Backup training...")
    print("-" * 70)

    start_time = time.time()

    result = n_step_tree_backup(
        environment=env,
        behavior_policy=behavior_policy,
        num_episodes=num_episodes,
        n=n,
        alpha=alpha,
        gamma=gamma,
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
    print("  ✓ Parameters: num_episodes, n, alpha, gamma, epsilon, grid dimensions")
    print("  ✓ Metrics: episode_reward, episode_steps, training_time, episodes_per_second")
    print("  ✓ Images: images/state_heatmap, images/value_function (every 500 episodes),")
    print("            images/policy_visualization (final)")
    print("  ✓ Videos: videos/greedy_agent_animation (periodic clips + final)")
    print("  ✓ Registered model: windy_gridworld_tree_backup_values (Model Registry)")
    print()
    print(f"  Run name: {run_name}")
    print("=" * 70)

    # Flush remaining batched data, shut down async worker, and end MLflow run
    collector.close()


if __name__ == "__main__":
    main()
