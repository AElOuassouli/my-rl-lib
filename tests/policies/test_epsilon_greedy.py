"""Unit tests for EpsilonGreedy policy."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer, InitializerType


def _make_env_and_values(q_vals: dict | None = None):
    env = MagicMock()
    env.get_terminal_states.return_value = [1]
    env.get_actions_per_state.return_value = {0: [0, 1], 1: [0, 1]}
    init = Initializer(
        initializer_type=InitializerType.CONSTANT,
        terminal_states_value=0.0,
        constant_value_non_terminal=0.0,
    )
    values = ActionStateValues()
    values.init_from_environment(env, init)
    if q_vals:
        for (s, a), v in q_vals.items():
            values.set_value((s, a), v)
    return env, values


class TestProbabilitiesSum:
    def test_probs_sum_to_one_for_all_states(self):
        env, values = _make_env_and_values()
        policy = EpsilonGreedy(epsilon=0.1)
        policy.init_from_environment_and_values(env, values)
        for state, probs in policy.action_probabilities_per_state.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_probs_sum_after_update(self):
        env, values = _make_env_and_values({(0, 0): 5.0})
        policy = EpsilonGreedy(epsilon=0.2)
        policy.init_from_environment_and_values(env, values)
        values.set_value((0, 1), 10.0)
        policy.update_probabilities_for_state(0, values)
        probs = policy.action_probabilities_per_state[0]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


class TestGreedyActionProbability:
    def test_greedy_action_has_higher_probability(self):
        # action 0 has higher Q → greedy
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=0.1)
        policy.init_from_environment_and_values(env, values)
        probs = policy.action_probabilities_per_state[0]
        assert probs[0] > probs[1]

    def test_greedy_action_exact_probability_with_two_actions(self):
        # epsilon=0.1, 2 actions: greedy = 1 - 0.1 + 0.1/2 = 0.95
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=0.1)
        policy.init_from_environment_and_values(env, values)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(0.95)
        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.05)

    def test_non_greedy_action_probability(self):
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=0.2)
        policy.init_from_environment_and_values(env, values)
        # epsilon=0.2, 2 actions: non-greedy = 0.2/2 = 0.1
        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.1)


class TestEpsilonEdgeCases:
    def test_epsilon_zero_is_pure_greedy(self):
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=0.0)
        policy.init_from_environment_and_values(env, values)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(1.0)
        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.0)

    def test_epsilon_one_is_uniform(self):
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=1.0)
        policy.init_from_environment_and_values(env, values)
        # epsilon=1: greedy = 1 - 1 + 1/2 = 0.5; others = 1/2 = 0.5
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(0.5)
        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.5)


class TestUpdateProbabilities:
    def test_update_after_value_change_shifts_greedy(self):
        env, values = _make_env_and_values({(0, 0): 5.0, (0, 1): 0.0})
        policy = EpsilonGreedy(epsilon=0.1)
        policy.init_from_environment_and_values(env, values)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(0.95)

        # Flip: make action 1 better
        values.set_value((0, 0), 0.0)
        values.set_value((0, 1), 10.0)
        policy.update_probabilities_for_state(0, values)

        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.95)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(0.05)
