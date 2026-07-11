from __future__ import annotations

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.n_step_utils import compute_n_step_tree_backup_return
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.abstract import Policy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer


def n_step_tree_backup(
    environment: Environment[StateT, ActionT],
    behavior_policy: Policy[StateT, ActionT],
    num_episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,  # Mutable: will be populated during training
) -> LearningResult[StateT, ActionT]:
    """
    N-step tree backup off-policy learning algorithm.

    Args:
        environment: The environment to train in
        behavior_policy: Policy used to generate behavior (must have non-zero
                        probability for all state-action pairs)
        num_episodes: Number of training episodes
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

    for episode in trange(num_episodes, desc="N-Step Tree Backup Episodes", unit="episode"):
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
                G = compute_n_step_tree_backup_return(
                    store=store, n=n, tau=tau, t=t, T=T, gamma=gamma, values=values, policy=policy
                )

                step_tau = store.get_step(tau)
                value_tau = values.get_value((step_tau.state, step_tau.action))
                td_error = G - value_tau
                update = value_tau + alpha * td_error
                values.set_value((step_tau.state, step_tau.action), update)

                policy.update_probabilities_for_state(step_tau.state, values)

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
            )

    return LearningResult(values=values, policy=policy)
