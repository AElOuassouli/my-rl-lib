from typing import Any

from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.policies.abstract import Policy


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


def compute_n_step_tree_backup_return(
    store: EpisodeStepsCircularStore[Any, Any],
    n: int,
    tau: int,
    t: int,
    T: int | float,
    gamma: float,
    values: ActionStateValues[Any, Any],
    policy: Policy[Any, Any],
) -> float:
    G = 0.0

    if t + 1 >= T:
        assert isinstance(T, int)
        terminal_reward = store.get_step(T).reward
        if terminal_reward is None:
            raise ValueError(f"Reward at time step {T} should not be None.")
        G = terminal_reward
    else:
        step_t_1 = store.get_step(t + 1)
        if step_t_1.reward is None:
            raise ValueError(f"Reward at time step {t + 1} should not be None.")
        G = step_t_1.reward

        state_t_1 = step_t_1.state

        possible_actions_probs = policy.get_actions_probabilities_given_state(state_t_1)

        for possible_action, possible_action_probability in possible_actions_probs.items():
            G += (
                gamma * possible_action_probability * values.get_value((state_t_1, possible_action))
            )

    k = min(t, T - 1) if isinstance(T, int) else t
    assert G is not None

    for index in list(reversed(range(tau + 1, k + 1))):
        print(index)
        step_k = store.get_step(k)
        if step_k.reward is None:
            raise ValueError(f"Reward at time step {k} should not be None.")
        G_new = step_k.reward
        possible_actions_probs_k = policy.get_actions_probabilities_given_state(step_k.state)
        for possible_action_k, possible_action_probability_k in possible_actions_probs_k.items():
            if possible_action_k == step_k.action:
                continue

            G_new += (
                gamma
                * possible_action_probability_k
                * values.get_value((step_k.state, possible_action_k))
            )

        assert step_k.action in possible_actions_probs_k

        action_k_prob = possible_actions_probs_k[step_k.action]

        assert action_k_prob is not None
        assert isinstance(G_new, float)

        G_new += gamma * action_k_prob * G

        G = G_new

    return G
