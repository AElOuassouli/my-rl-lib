from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.n_step_utils import compute_n_step_return
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer

if TYPE_CHECKING:
    from my_rl_lib.visualization import WebDashboardVisualizer  # type: ignore[import-not-found]


def n_step_sarsa(
    environment: Environment,
    num_episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    initializer: Initializer,
    epsilon: float = 0.1,
    metrics_collector: MetricsCollector | None = None,  # Mutable: will be populated during training
    visualizer: WebDashboardVisualizer | None = None,  # Optional: real-time web dashboard
) -> LearningResult:
    """
    N-step SARSA on-policy learning algorithm.

    Args:
        environment: The environment to train in
        num_episodes: Number of training episodes
        n: Number of steps for n-step returns
        alpha: Learning rate (step size)
        gamma: Discount factor
        initializer: Value initialization strategy
        epsilon: Epsilon parameter for the epsilon-greedy policy (default: 0.1)
        metrics_collector: Optional metrics collector. If provided, it will be
                          populated with training metrics during execution.
                          The collector is mutable and modified in-place.
        visualizer: Optional WebDashboardVisualizer. If provided, will show
                   real-time training progress in browser. The visualizer runs
                   in a separate process and does not block training.

    Returns:
        Tuple of (learned values, learned policy)
    """
    values = ActionStateValues()
    values.init_from_environment(environment, initializer)

    epsilon_greedy_policy = EpsilonGreedy(epsilon=epsilon)
    epsilon_greedy_policy.init_from_environment_and_values(environment, values)

    # Start visualization process if provided
    if visualizer is not None:
        visualizer.start()

    try:
        for episode in trange(num_episodes, desc="N-Step SARSA Episodes", unit="episode"):
            store = EpisodeStepsCircularStore(n=n)

            environment.reset()
            episode_reward = 0.0
            episode_steps = 0
            T = float("inf")
            t = 0

            store.set_step(
                0,
                LearningStep(
                    state=environment.current_state,
                    action=epsilon_greedy_policy.select_action(environment.current_state),
                    reward=None,
                ),
            )

            while True:
                if t < T:
                    entry = store.get_step(t)
                    At = entry.action
                    step_result = environment.step(action=At)

                    St1 = step_result.next_state
                    Rt1 = step_result.reward

                    store.set_step(
                        t + 1,
                        LearningStep(
                            state=St1,
                            action=(
                                epsilon_greedy_policy.select_action(St1)
                                if not environment.is_current_state_terminal()
                                else None
                            ),
                            reward=Rt1,
                        ),
                    )

                    if metrics_collector is not None:
                        episode_reward += Rt1
                        episode_steps += 1

                    if environment.is_current_state_terminal():
                        T = t + 1

                tau = t - n + 1
                if tau >= 0:
                    n_step_return = compute_n_step_return(store, n, tau, T, gamma, values)

                    entry = store.get_step(tau)
                    S_tau = entry.state
                    A_tau = entry.action
                    previous_value = values.get_value((S_tau, A_tau))
                    td_error = n_step_return - previous_value
                    values.set_value((S_tau, A_tau), previous_value + alpha * td_error)

                    # update greedy policy for couple S_tau
                    epsilon_greedy_policy.update_probabilities_for_state(S_tau, values)

                    # Record step metrics
                    if metrics_collector is not None:
                        metrics_collector.on_step(
                            episode,
                            t,
                            td_error=abs(td_error),
                            value_change=abs(alpha * td_error),
                        )

                if tau == T - 1:
                    break

                t += 1

            # Record episode metrics
            if metrics_collector is not None:
                metrics_collector.on_episode_end(
                    episode,
                    episode_reward=episode_reward,
                    episode_steps=float(episode_steps),
                    epsilon=epsilon_greedy_policy.epsilon,
                )

                # Update visualization (non-blocking)
                if visualizer is not None:
                    visualizer.update(
                        episode=episode,
                        metrics_data=metrics_collector.export(),
                        values=values,
                        environment=environment,
                    )

    finally:
        # Clean shutdown of visualization
        if visualizer is not None:
            visualizer.stop()

    return LearningResult(values=values, policy=epsilon_greedy_policy)
