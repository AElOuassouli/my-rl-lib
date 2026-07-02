from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.n_step_utils import compute_n_step_return
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.abstract import Policy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.policies.importance_sampling import compute_importance_sampling_ratio_circular_buffer
from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer

if TYPE_CHECKING:
    from my_rl_lib.visualization import WebDashboardVisualizer  # type: ignore[import-not-found]


def n_step_sarsa_off_policy(
    environment: Environment[StateT, ActionT],
    num_episodes: int,
    behavior_policy: Policy[StateT, ActionT],
    n: int,
    alpha: float,
    gamma: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,  # Mutable: will be populated during training
    visualizer: WebDashboardVisualizer | None = None,  # Optional: real-time web dashboard
) -> LearningResult[StateT, ActionT]:
    """
    N-step SARSA off-policy learning algorithm with importance sampling.

    Args:
        environment: The environment to train in
        num_episodes: Number of training episodes
        behavior_policy: Policy used to generate behavior (must have non-zero
                        probability for all state-action pairs)
        n: Number of steps for n-step returns
        alpha: Learning rate (step size)
        gamma: Discount factor
        initializer: Value initialization strategy
        metrics_collector: Optional metrics collector. If provided, it will be
                          populated with training metrics during execution.
                          The collector is mutable and modified in-place.

    Returns:
        Tuple of (learned values, learned target policy)
    """
    values: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values.init_from_environment(environment=environment, initializer=initializer)

    policy: Greedy[StateT, ActionT] = Greedy()
    policy.init_from_environment_and_values(environment=environment, values=values)

    # Start visualization process if provided
    if visualizer is not None:
        visualizer.start()

    try:
        for episode in trange(
            num_episodes, desc="N-Step SARSA Off-Policy Episodes", unit="episode"
        ):
            environment.reset()

            store: EpisodeStepsCircularStore[StateT, ActionT] = EpisodeStepsCircularStore(n=n)
            episode_reward = 0.0
            episode_steps = 0

            initial_state = environment.current_state
            assert initial_state is not None  # set by reset()
            store.set_step(
                0,
                LearningStep(
                    state=initial_state,
                    action=behavior_policy.select_action(initial_state),
                    reward=None,
                ),
            )

            T = float("inf")
            t = 0

            while True:
                if t < T:
                    entry = store.get_step(t)

                    At = entry.action
                    assert At is not None  # only the terminal step stores a None action

                    step_result = environment.step(action=At)

                    store.set_step(
                        t + 1,
                        LearningStep(
                            state=step_result.next_state,
                            action=(
                                behavior_policy.select_action(step_result.next_state)
                                if not environment.is_current_state_terminal()
                                else None
                            ),
                            reward=step_result.reward,
                        ),
                    )

                    if metrics_collector is not None:
                        episode_reward += step_result.reward
                        episode_steps += 1

                    if environment.is_current_state_terminal():
                        T = t + 1

                tau = t - n + 1  # time whose estimate is being updated
                if tau >= 0:
                    # Compute G (n-step return)
                    G = compute_n_step_return(store, n, tau, T, gamma, values)

                    # Compute importance sampling ratio
                    upper_bound = int(min(tau + n - 1, T - 1)) if T != float("inf") else tau + n - 1
                    rho = compute_importance_sampling_ratio_circular_buffer(
                        target_policy=policy,
                        behavior_policy=behavior_policy,
                        steps=store,
                        lower_bound=tau + 1,
                        upper_bound=upper_bound,
                    )

                    # Update Q-value with importance sampling
                    entry_tau = store.get_step(tau)
                    S_tau = entry_tau.state
                    A_tau = entry_tau.action
                    previous_value = values.get_value((S_tau, A_tau))
                    td_error = rho * (G - previous_value)
                    values.set_value((S_tau, A_tau), previous_value + alpha * td_error)

                    # Update target policy
                    policy.update_probabilities_for_state(S_tau, values)

                    # Record step metrics
                    if metrics_collector is not None:
                        metrics_collector.on_step(
                            episode,
                            t,
                            td_error=abs(td_error),
                            value_change=abs(alpha * td_error),
                            importance_ratio=rho,
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

    return LearningResult(values=values, policy=policy)
