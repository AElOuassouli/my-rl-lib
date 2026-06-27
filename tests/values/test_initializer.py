"""Unit tests for Initializer."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.values.initializer import Initializer, InitializerType


def _make_env(terminal_states, states_actions):
    env = MagicMock()
    env.get_terminal_states.return_value = terminal_states
    env.get_actions_per_state.return_value = states_actions
    return env


class TestConstantInitializer:
    def test_non_terminal_state_gets_constant_value(self):
        env = _make_env([1], {0: [0, 1], 1: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=0.0,
            constant_value_non_terminal=5.0,
        )
        fn = init.get_state_action_values_initializer_function(env)
        assert fn((0, 0)) == 5.0
        assert fn((0, 1)) == 5.0

    def test_terminal_state_gets_terminal_value(self):
        env = _make_env([1], {0: [0, 1], 1: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=-99.0,
            constant_value_non_terminal=5.0,
        )
        fn = init.get_state_action_values_initializer_function(env)
        assert fn((1, 0)) == -99.0
        assert fn((1, 1)) == -99.0

    def test_multiple_non_terminal_states(self):
        env = _make_env([2], {0: [0], 1: [0], 2: [0]})
        init = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=0.0,
            constant_value_non_terminal=3.0,
        )
        fn = init.get_state_action_values_initializer_function(env)
        assert fn((0, 0)) == 3.0
        assert fn((1, 0)) == 3.0
        assert fn((2, 0)) == 0.0  # terminal

    def test_missing_constant_value_raises(self):
        env = _make_env([1], {0: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=0.0,
            constant_value_non_terminal=None,
        )
        with pytest.raises(ValueError, match="constant_value_non_terminal"):
            init.get_state_action_values_initializer_function(env)


class TestUniformInitializer:
    def test_non_terminal_value_within_range(self):
        env = _make_env([1], {0: [0, 1], 1: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.UNIFORM,
            terminal_states_value=0.0,
            range_uniform_non_terminal=(-1.0, 1.0),
        )
        fn = init.get_state_action_values_initializer_function(env)
        for _ in range(20):
            val = fn((0, 0))
            assert -1.0 <= val <= 1.0

    def test_terminal_state_gets_terminal_value(self):
        env = _make_env([1], {0: [0, 1], 1: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.UNIFORM,
            terminal_states_value=42.0,
            range_uniform_non_terminal=(-1.0, 1.0),
        )
        fn = init.get_state_action_values_initializer_function(env)
        assert fn((1, 0)) == 42.0

    def test_missing_range_raises(self):
        env = _make_env([1], {0: [0, 1]})
        init = Initializer(
            initializer_type=InitializerType.UNIFORM,
            terminal_states_value=0.0,
            range_uniform_non_terminal=None,
        )
        with pytest.raises(ValueError, match="range_uniform_non_terminal"):
            init.get_state_action_values_initializer_function(env)
