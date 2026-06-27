import numpy as np
from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.result import DoubleQLearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer


def double_q_learning(
    environment: Environment,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    initializer: Initializer,
) -> DoubleQLearningResult:
    # Initialize action-state values for

    values_1 = ActionStateValues()
    values_1.init_from_environment(environment, initializer)

    values_2 = ActionStateValues()
    values_2.init_from_environment(environment, initializer)

    values_behavior_policy = values_1.add(values_2)
    behavior_policy = EpsilonGreedy(epsilon=epsilon)
    behavior_policy.init_from_environment_and_values(environment, values_behavior_policy)

    for _ in trange(num_episodes, desc="Double Q-Learning Episodes", unit="episode"):
        environment.reset()

        while not environment.is_current_state_terminal():
            St = environment.current_state
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

    return DoubleQLearningResult(values_a=values_1, values_b=values_2, policy=behavior_policy)
