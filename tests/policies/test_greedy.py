"""Unit tests for Greedy policy."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.policies.greedy import Greedy
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


class TestGreedyProbabilities:
    def test_best_action_gets_probability_one(self):
        env, values = _make_env_and_values({(0, 0): 3.0, (0, 1): 1.0})
        policy = Greedy()
        policy.init_from_environment_and_values(env, values)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(1.0)
        assert policy.action_probabilities_per_state[0][1] == pytest.approx(0.0)

    def test_probabilities_sum_to_one(self):
        env, values = _make_env_and_values({(0, 0): 3.0, (0, 1): 1.0})
        policy = Greedy()
        policy.init_from_environment_and_values(env, values)
        for state, probs in policy.action_probabilities_per_state.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_states_initialized(self):
        env, values = _make_env_and_values()
        policy = Greedy()
        policy.init_from_environment_and_values(env, values)
        assert set(policy.action_probabilities_per_state.keys()) == {0, 1}


class TestUpdateProbabilities:
    def test_update_reflects_new_best_action(self):
        env, values = _make_env_and_values({(0, 0): 3.0, (0, 1): 1.0})
        policy = Greedy()
        policy.init_from_environment_and_values(env, values)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(1.0)

        # Flip: make action 1 the best
        values.set_value((0, 0), 0.0)
        values.set_value((0, 1), 5.0)
        policy.update_probabilities_for_state(0, values)

        assert policy.action_probabilities_per_state[0][1] == pytest.approx(1.0)
        assert policy.action_probabilities_per_state[0][0] == pytest.approx(0.0)

    def test_probabilities_sum_to_one_after_update(self):
        env, values = _make_env_and_values({(0, 0): 3.0})
        policy = Greedy()
        policy.init_from_environment_and_values(env, values)
        values.set_value((0, 1), 10.0)
        policy.update_probabilities_for_state(0, values)
        probs = policy.action_probabilities_per_state[0]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
