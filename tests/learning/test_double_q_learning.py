"""Unit tests for Double Q-Learning."""

import pytest

from my_rl_lib.learning.off_policy.double_q_learning import double_q_learning
from my_rl_lib.learning.result import DoubleQLearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues


class TestDoubleQLearningReturnType:
    def test_returns_double_q_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = double_q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, DoubleQLearningResult)

    def test_result_has_two_value_functions(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = double_q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.values_a, ActionStateValues)
        assert isinstance(result.values_b, ActionStateValues)
        assert result.values_a is not result.values_b

    def test_result_has_epsilon_greedy_policy(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = double_q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, EpsilonGreedy)


class TestDoubleQLearningUpdate:
    def test_exactly_one_q_table_updated_per_episode(self, make_simple_env, basic_initializer):
        # After 1 episode: exactly one of values_a or values_b gets Q(0,0) updated to 1.0.
        # The other stays 0.0. Their sum must equal 1.0.
        env = make_simple_env()
        result = double_q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        q_a = result.values_a.get_value((0, 0))
        q_b = result.values_b.get_value((0, 0))
        assert q_a + q_b == pytest.approx(1.0)

    def test_value_functions_start_initialised(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = double_q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values_a.values is not None
        assert result.values_b.values is not None
