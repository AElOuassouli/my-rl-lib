"""Unit tests for Q-Learning (off-policy TD(0))."""

import pytest

from my_rl_lib.learning.off_policy.q_learning import q_learning
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues


class TestQLearningReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, LearningResult)

    def test_result_has_action_state_values(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.values, ActionStateValues)
        assert result.values.values is not None

    def test_result_has_epsilon_greedy_policy(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, EpsilonGreedy)


class TestQLearningTDUpdate:
    def test_q_value_updated_with_max_bootstrap(self, make_simple_env, basic_initializer):
        # Q-learning: TD target = R + gamma * max_a Q(S', a)
        # max Q(1, *) = 0.0, so TD target = 1.0 + 1.0 * 0.0 = 1.0
        # new Q(0, 0) = 0 + 1.0 * (1.0 - 0) = 1.0
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(1.0)

    def test_alpha_scales_the_update(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(0.5)

    def test_untouched_q_values_remain_zero(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = q_learning(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 1)) == pytest.approx(0.0)
        assert result.values.get_value((1, 0)) == pytest.approx(0.0)
