"""Unit tests for ActionStateValues."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer, InitializerType


def _make_values(q_vals: dict | None = None) -> ActionStateValues:
    """Build ActionStateValues for states {0:[0,1], 1:[0,1]}, optionally seeded with Q-values."""
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
        for (state, action), val in q_vals.items():
            values.set_value((state, action), val)
    return values


class TestGetSetValue:
    def test_get_after_set(self):
        v = _make_values()
        v.set_value((0, 0), 3.14)
        assert v.get_value((0, 0)) == pytest.approx(3.14)

    def test_initial_values_are_zero(self):
        v = _make_values()
        assert v.get_value((0, 0)) == 0.0
        assert v.get_value((0, 1)) == 0.0
        assert v.get_value((1, 0)) == 0.0

    def test_set_does_not_affect_other_entries(self):
        v = _make_values()
        v.set_value((0, 0), 7.0)
        assert v.get_value((0, 1)) == 0.0
        assert v.get_value((1, 0)) == 0.0

    def test_invalid_state_raises(self):
        v = _make_values()
        with pytest.raises(ValueError, match="not found"):
            v.get_value((99, 0))

    def test_invalid_action_raises(self):
        v = _make_values()
        with pytest.raises(ValueError, match="not found"):
            v.get_value((0, 99))

    def test_non_tuple_entry_raises(self):
        v = _make_values()
        with pytest.raises(ValueError, match="tuple"):
            v.get_value(0)

    def test_uninitialised_raises(self):
        v = ActionStateValues()
        with pytest.raises(ValueError, match="not been initialized"):
            v.get_value((0, 0))


class TestGetMaxActionAndValue:
    def test_returns_action_with_highest_q(self):
        v = _make_values({(0, 0): 2.0, (0, 1): 5.0})
        action, value = v.get_max_action_and_value(0)
        assert action == 1
        assert value == pytest.approx(5.0)

    def test_all_equal_returns_a_valid_action(self):
        v = _make_values()
        action, value = v.get_max_action_and_value(0)
        assert action in [0, 1]
        assert value == pytest.approx(0.0)

    def test_negative_values_handled(self):
        v = _make_values({(0, 0): -3.0, (0, 1): -1.0})
        action, value = v.get_max_action_and_value(0)
        assert action == 1
        assert value == pytest.approx(-1.0)

    def test_invalid_state_raises(self):
        v = _make_values()
        with pytest.raises(ValueError, match="not found"):
            v.get_max_action_and_value(99)


class TestGetExpectedValue:
    def test_weighted_sum_matches_manual_calculation(self):
        # E[Q(1,*)] = 0.6 * 4.0 + 0.4 * 2.0 = 3.2
        v = _make_values({(1, 0): 4.0, (1, 1): 2.0})
        policy = MagicMock()
        policy.get_action_probability_given_state.side_effect = (
            lambda state, action: 0.6 if action == 0 else 0.4
        )
        assert v.get_expected_value(1, policy) == pytest.approx(3.2)

    def test_uniform_policy_gives_mean(self):
        v = _make_values({(0, 0): 2.0, (0, 1): 4.0})
        policy = MagicMock()
        policy.get_action_probability_given_state.return_value = 0.5
        assert v.get_expected_value(0, policy) == pytest.approx(3.0)

    def test_zero_q_values_give_zero(self):
        v = _make_values()
        policy = MagicMock()
        policy.get_action_probability_given_state.return_value = 0.5
        assert v.get_expected_value(0, policy) == pytest.approx(0.0)


class TestAdd:
    def test_add_combines_values_element_wise(self):
        v1 = _make_values({(0, 0): 2.0, (0, 1): 3.0})
        v2 = _make_values({(0, 0): 1.0, (0, 1): 4.0})
        v3 = v1.add(v2)
        assert v3.get_value((0, 0)) == pytest.approx(3.0)
        assert v3.get_value((0, 1)) == pytest.approx(7.0)

    def test_add_with_all_zeros(self):
        v1 = _make_values()
        v2 = _make_values()
        v3 = v1.add(v2)
        assert v3.get_value((0, 0)) == pytest.approx(0.0)

    def test_add_non_action_state_values_raises(self):
        v1 = _make_values()
        with pytest.raises(ValueError):
            v1.add(MagicMock())
