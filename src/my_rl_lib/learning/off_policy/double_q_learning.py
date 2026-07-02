from typing import Any

import numpy as np
from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.result import DoubleQLearningResult
from my_rl_lib.metrics.collector import MetricsCollector
from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer


def double_q_learning(
    environment: Environment[StateT, ActionT],
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,
) -> DoubleQLearningResult[StateT, ActionT]:
    # Initialize action-state values for

    values_1: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values_1.init_from_environment(environment, initializer)

    values_2: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values_2.init_from_environment(environment, initializer)

    values_behavior_policy = values_1.add(values_2)
    behavior_policy: EpsilonGreedy[StateT, ActionT] = EpsilonGreedy(epsilon=epsilon)
    behavior_policy.init_from_environment_and_values(environment, values_behavior_policy)

    all_state_visits: dict[StateT, int] = {}

    for episode in trange(num_episodes, desc="Double Q-Learning Episodes", unit="episode"):
        environment.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_state_visits: list[StateT] = []

        while not environment.is_current_state_terminal():
            St = environment.current_state
            assert St is not None  # non-terminal loop guarantees a current state
            episode_state_visits.append(St)

            At = behavior_policy.select_action(St)
            step_result = environment.step(action=At)

            St1 = step_result.next_state
            Rt1 = step_result.reward

            # randomly choose which value function to update
            update_first = np.random.randint(0, 2)

            # select which value functions to use based on random choice
            values_to_update = values_1 if update_first else values_2
            values_for_target = values_2 if update_first else values_1

            # cache current Q-value to avoid redundant lookup
            current_q_value = values_to_update.get_value((St, At))

            # compute update
            max_action, _ = values_to_update.get_max_action_and_value(St1)
            target_value = values_for_target.get_value((St1, max_action))
            update = Rt1 + gamma * target_value - current_q_value
            values_to_update.set_value((St, At), current_q_value + alpha * update)

            # update behavior policy only for the updated state-action pair
            values_behavior_policy.set_value(
                (St, At), values_1.get_value((St, At)) + values_2.get_value((St, At))
            )
            behavior_policy.update_probabilities_for_state(St, values_behavior_policy)

            episode_reward += Rt1
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
                context_data[ContextKey.VALUE_FUNCTION.value] = values_behavior_policy
            if ContextKey.ENVIRONMENT in required_keys:
                context_data[ContextKey.ENVIRONMENT.value] = environment
            metrics_collector.on_episode_end(episode=episode, **context_data)

    return DoubleQLearningResult(values_a=values_1, values_b=values_2, policy=behavior_policy)
