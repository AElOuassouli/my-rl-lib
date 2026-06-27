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

    for episode in trange(num_episodes, desc="Q-Learning Episodes", unit="episode"):
        environment.reset()
        episode_reward = 0.0
        episode_steps = 0

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

        if metrics_collector is not None:
            metrics_collector.on_episode_end(
                episode=episode,
                **{
                    ContextKey.EPISODE_REWARD.value: episode_reward,
                    ContextKey.EPISODE_STEPS.value: episode_steps,
                    ContextKey.EPSILON.value: epsilon,
                },
            )

    return LearningResult(values=values, policy=epsilon_greedy_policy)
