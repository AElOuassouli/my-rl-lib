from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.types import ActionT, StateT
from my_rl_lib.policies.abstract import Policy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer
from my_rl_lib.metrics import MetricsCollector
from tqdm.auto import trange

from my_rl_lib.learning.n_step_utils import compute_n_step_tree_backup_return
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep


def n_step_tree_backup(
    environment: Environment[StateT, ActionT],
    behavior_policy: Policy[StateT, ActionT],
    num_episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    initializer: Initializer,
    metrics_collector: MetricsCollector | None = None,
) -> LearningResult[StateT, ActionT]:
    values: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values.init_from_environment(environment=environment, initializer=initializer)

    policy: Greedy[StateT, ActionT] = Greedy()
    policy.init_from_environment_and_values(environment=environment, values=values)

    for episode in trange(num_episodes, desc="N-Step SARSA Off-Policy Episodes", unit="episode"):
        environment.reset()

        store: EpisodeStepsCircularStore[StateT, ActionT] = EpisodeStepsCircularStore(n=n)
        episode_reward = 0.0
        episode_steps = 0

        initial_state = environment.current_state
        assert initial_state is not None  # set by reset()s

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
        tau = -float("inf")

        while tau < T - 1:
            if t < T:
                entry = store.get_step(t)

                At = entry.action
                assert At is not None

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
                update = value_tau + alpha * (G - value_tau)
                values.set_value((step_tau.state, step_tau.action), update)

                policy.update_probabilities_for_state(step_tau.state, values)

    return LearningResult(values=values, policy=policy)
