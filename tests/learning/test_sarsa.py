"""Unit tests for SARSA (on-policy TD(0))."""

import pytest

from my_rl_lib.learning.on_policy.sarsa import sarsa
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues


class TestSarsaReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = sarsa(
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
        result = sarsa(
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
        result = sarsa(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, EpsilonGreedy)


class TestSarsaTDUpdate:
    def test_q_value_updated_after_one_episode(self, make_simple_env, basic_initializer):
        # Setup: all Q=0, alpha=1.0, gamma=1.0, epsilon=0 (pure greedy)
        # Trajectory: S0 --A0--> S1(terminal), R=1.0
        # Greedy action in S0: action 0 (first key, all Q equal)
        # Greedy action in S1: action 0
        # TD target = R + gamma * Q(S1, A1) = 1.0 + 1.0 * 0.0 = 1.0
        # new Q(0, 0) = 0 + 1.0 * (1.0 - 0.0) = 1.0
        env = make_simple_env()
        result = sarsa(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(1.0)

    def test_alpha_scales_the_update(self, make_simple_env, basic_initializer):
        # alpha=0.5: new Q(0,0) = 0 + 0.5 * (1.0 - 0) = 0.5
        env = make_simple_env()
        result = sarsa(
            env,
            num_episodes=1,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(0.5)

    def test_untouched_q_values_remain_zero(self, make_simple_env, basic_initializer):
        # Only Q(0, 0) is on the trajectory; others unchanged
        env = make_simple_env()
        result = sarsa(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 1)) == pytest.approx(0.0)
        assert result.values.get_value((1, 0)) == pytest.approx(0.0)
        assert result.values.get_value((1, 1)) == pytest.approx(0.0)


class TestSarsaValidation:
    def test_zero_episodes_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError, match="num_episodes"):
            sarsa(
                env,
                num_episodes=0,
                alpha=1.0,
                gamma=1.0,
                epsilon=0.0,
                initializer=basic_initializer,
            )

    def test_alpha_above_one_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError, match="alpha"):
            sarsa(
                env,
                num_episodes=1,
                alpha=1.5,
                gamma=1.0,
                epsilon=0.0,
                initializer=basic_initializer,
            )

    def test_alpha_zero_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError, match="alpha"):
            sarsa(
                env,
                num_episodes=1,
                alpha=0.0,
                gamma=1.0,
                epsilon=0.0,
                initializer=basic_initializer,
            )
