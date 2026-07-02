from typing import Any

from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore
from my_rl_lib.values.action_state import ActionStateValues


def compute_n_step_return(
    store: EpisodeStepsCircularStore[Any, Any],
    n: int,
    tau: int,
    T: int | float,
    gamma: float,
    values: ActionStateValues[Any, Any],
) -> float:
    """
    Compute the n-step return for SARSA algorithms.

    Args:
        store: Circular buffer containing states, actions, and rewards
        n: Number of steps for the return calculation
        tau: Time step whose estimate is being updated
        T: Terminal time step (or inf if episode hasn't terminated)
        gamma: Discount factor
        values: Action-state value function for bootstrapping

    Returns:
        The n-step return G_tau
    """
    G = 0.0
    range_upper_bound = T
    if tau + n < T:
        range_upper_bound = tau + n
    range_upper_bound = int(range_upper_bound)

    # Sum rewards from R_{tau+1} to R_{min(tau+n, T)}
    for i in range(tau + 1, range_upper_bound + 1):
        entry = store.get_step(i)
        reward = entry.reward
        if reward is None:
            raise ValueError(f"Reward at time step {i} should not be None.")
        G += pow(gamma, i - tau - 1) * reward

    # Add bootstrapped value if episode didn't terminate within n steps
    if tau + n < T:
        entry = store.get_step(tau + n)
        S_t_plus_n = entry.state
        A_t_plus_n = entry.action
        G += pow(gamma, n) * values.get_value((S_t_plus_n, A_t_plus_n))

    return G
