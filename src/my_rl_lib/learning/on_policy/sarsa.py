from typing import Any

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.metrics.collector import MetricsCollector
from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer


def sarsa(
    environment: Environment[StateT, ActionT],
    num_episodes: int,
    alpha: float,  # step size
    gamma: float,
    epsilon: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,
) -> LearningResult[StateT, ActionT]:
    # validate parameters
    if num_episodes <= 0:
        raise ValueError("num_episodes must be a positive integer.")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in the range (0, 1].")

    # Initialize action-state values
    values: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values.init_from_environment(environment, initializer)

    # Initialize epsilon-greedy policy
    policy: EpsilonGreedy[StateT, ActionT] = EpsilonGreedy(epsilon=epsilon)
    policy.init_from_environment_and_values(environment, values)

    all_state_visits: dict[StateT, int] = {}

    for episode in trange(num_episodes):
        environment.reset()
        state = environment.current_state
        assert state is not None  # set by reset()
        action = policy.select_action(state)

        episode_reward = 0.0
        episode_steps = 0
        episode_state_visits: list[StateT] = []

        while not environment.is_current_state_terminal():
            episode_state_visits.append(state)

            step_result = environment.step(action)
            next_state = step_result.next_state

            next_action = policy.select_action(next_state)

            # Update Q-value
            td_target = step_result.reward + gamma * values.get_value((next_state, next_action))
            td_delta = td_target - values.get_value((state, action))
            values.set_value((state, action), values.get_value((state, action)) + alpha * td_delta)

            # Only update policy for the state that changed (incremental update)
            policy.update_probabilities_for_state(state, values)

            state = next_state
            action = next_action

            episode_reward += step_result.reward
            episode_steps += 1

        for state in episode_state_visits:
            all_state_visits[state] = all_state_visits.get(state, 0) + 1

        if metrics_collector is not None:
            context_data: dict[str, Any] = {
                ContextKey.EPISODE_REWARD.value: episode_reward,
                ContextKey.EPISODE_STEPS.value: episode_steps,
                ContextKey.EPSILON.value: epsilon,
            }
            required_keys = (
                metrics_collector._all_required_keys
                if hasattr(metrics_collector, "_all_required_keys")
                else set()
            )
            if ContextKey.STATE_VISITS in required_keys:
                context_data[ContextKey.STATE_VISITS.value] = dict(all_state_visits)
            if ContextKey.VALUE_FUNCTION in required_keys:
                context_data[ContextKey.VALUE_FUNCTION.value] = values
            if ContextKey.ENVIRONMENT in required_keys:
                context_data[ContextKey.ENVIRONMENT.value] = environment
            metrics_collector.on_episode_end(episode=episode, **context_data)

    return LearningResult(values=values, policy=policy)
