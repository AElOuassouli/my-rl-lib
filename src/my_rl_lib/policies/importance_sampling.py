from typing import Any

from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore
from my_rl_lib.policies.abstract import Policy


def compute_importance_sampling_ratio(
    target_policy: Policy, behavior_policy: Policy, state: Any, action: Any
) -> float:
    probability_target = target_policy.get_action_probability_given_state(state, action)
    probability_behavior = behavior_policy.get_action_probability_given_state(state, action)

    if probability_behavior is None or probability_behavior == 0.0:
        raise ValueError("Probability from behavior policy is zero.")

    return probability_target / probability_behavior


def compute_importance_sampling_ratio_circular_buffer(
    target_policy: Policy,
    behavior_policy: Policy,
    steps: EpisodeStepsCircularStore,  # circular buffer exposing get_step(time)
    lower_bound: int,
    upper_bound: int,
) -> float:
    if lower_bound > upper_bound:
        raise ValueError("Lower bound cannot be greater than upper bound.")

    ratio = 1.0
    for t in range(lower_bound, upper_bound + 1):
        try:
            step = steps.get_step(t)
        except ValueError:
            raise ValueError(f"Store entry missing at time step {t}.")

        state = step.state
        action = step.action
        try:
            step_ratio = compute_importance_sampling_ratio(
                target_policy, behavior_policy, state, action
            )
            ratio *= step_ratio
        except ValueError:
            raise ValueError(f"Failed to compute ratio at time step {t}.")

    return ratio
