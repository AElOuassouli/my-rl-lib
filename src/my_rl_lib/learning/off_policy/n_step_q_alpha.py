from __future__ import annotations

from tqdm.auto import trange

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.learning.n_step_utils import compute_n_step_sigma_tree_backup_return
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.abstract import Policy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.policies.importance_sampling import compute_importance_sampling_ratio
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
    values: ActionStateValues[StateT, ActionT] = ActionStateValues()
    values.init_from_environment(environment=environment, initializer=initializer)

    policy: Greedy[StateT, ActionT] = Greedy()
    policy.init_from_environment_and_values(environment=environment, values=values)

    for episode in trange(num_episodes, desc="N-Step Tree Backup Episodes", unit="episode"):
        environment.reset()

        store: EpisodeStepsCircularStore[StateT, ActionT] = EpisodeStepsCircularStore(n=n)

        initial_state = environment.current_state
        assert initial_state is not None  # set by reset

        t = 0
        T = float("inf")

        store.set_step(
            t,
            LearningStep(
                state=initial_state,
                action=behavior_policy.select_action(initial_state),
                reward=None,
            ),
        )

        while True:
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

                if environment.is_current_state_terminal():
                    T = t + 1
                else:
                    store.set_step_rho(
                        t + 1,
                        compute_importance_sampling_ratio(
                            target_policy=policy,
                            behavior_policy=behavior_policy,
                            state=store.get_step(t + 1).state,
                            action=store.get_step(t + 1).action,
                        ),
                    )
                    store.set_step_sigma(
                        t + 1,
                        (t + 1) % 2,  # alternating between sampling and not sampling
                    )

            tau = t - n + 1  # time whose estimate is being updated

            if tau >= 0:
                G = 0.0
                if t + 1 < T:
                    st1 = store.get_step(t + 1).state
                    at1 = store.get_step(t + 1).action
                    G = values.get_value((st1, at1))

                G = compute_n_step_sigma_tree_backup_return(
                    store=store, G=G, tau=tau, t=t, T=T, gamma=gamma, values=values, policy=policy
                )

                # update Q value
                entry_tau = store.get_step(tau)
                S_tau = entry_tau.state
                A_tau = entry_tau.action

                previous_value = values.get_value((S_tau, A_tau))

                values.set_value((S_tau, A_tau), previous_value + alpha * (G - previous_value))

                # update target policy
                policy.update_probabilities_for_state(S_tau, values)

            if tau == T - 1:
                break

            t += 1

    return LearningResult(values=values, policy=policy)
