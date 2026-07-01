from typing import Any

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.metrics.collector import MetricsCollector
from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer


def q_learning(
    environment: Environment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,
) -> LearningResult:
    # Initialize action-state values
    values = ActionStateValues()
    values.init_from_environment(environment, initializer)

    epsilon_greedy_policy = EpsilonGreedy(epsilon=epsilon)
    epsilon_greedy_policy.init_from_environment_and_values(environment, values)

    all_state_visits: dict[Any, int] = {}

    for episode in trange(num_episodes, desc="Q-Learning Episodes", unit="episode"):
        environment.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_state_visits: list[Any] = []

        while not environment.is_current_state_terminal():
            St = environment.current_state
            At = epsilon_greedy_policy.select_action(St)
            step_result = environment.step(action=At)

            St1 = step_result.next_state
            Rt1 = step_result.reward

            # compute update
            _, max_value = values.get_max_action_and_value(St1)
            update = Rt1 + gamma * max_value - values.get_value((St, At))
            # update action-state value
            values.set_value((St, At), values.get_value((St, At)) + alpha * update)

            epsilon_greedy_policy.update_probabilities_for_state(St, values)

            episode_reward += Rt1
            episode_steps += 1

            # Track state visits
            episode_state_visits.append(St)

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

    return LearningResult(values=values, policy=epsilon_greedy_policy)
