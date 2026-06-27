"""Shared fixtures for all test modules."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.environments.abstract import StepResult
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer, InitializerType


@pytest.fixture
def make_simple_env():
    """Factory fixture: creates a fresh mock 2-state, 2-action environment each call.

    States: 0 (start), 1 (terminal).
    Actions per state: [0, 1].
    One step: reward=1.0, transitions 0→1.
    reset() sets current_state=0.
    is_current_state_terminal() checks current_state dynamically.
    """

    def _make():
        env = MagicMock()
        env.get_states.return_value = [0, 1]
        env.get_terminal_states.return_value = [1]
        env.get_actions_per_state.return_value = {0: [0, 1], 1: [0, 1]}
        env.get_current_possible_actions.return_value = [0, 1]
        env.current_state = 0

        def reset_effect():
            env.current_state = 0

        env.reset.side_effect = reset_effect

        def step_effect(action):
            env.current_state = 1
            return StepResult(next_state=1, reward=1.0)

        env.step.side_effect = step_effect

        # Dynamically checks current_state — works for both regular loops and n-step
        env.is_current_state_terminal.side_effect = lambda: env.current_state == 1

        return env

    return _make


@pytest.fixture
def simple_env(make_simple_env):
    """A single ready-to-use mock environment instance."""
    return make_simple_env()


@pytest.fixture
def basic_initializer():
    """CONSTANT initializer that sets all Q-values to 0.0."""
    return Initializer(
        initializer_type=InitializerType.CONSTANT,
        terminal_states_value=0.0,
        constant_value_non_terminal=0.0,
    )


@pytest.fixture
def basic_values(simple_env, basic_initializer):
    """ActionStateValues initialized from simple_env with all Q=0.0."""
    values = ActionStateValues()
    values.init_from_environment(simple_env, basic_initializer)
    return values
