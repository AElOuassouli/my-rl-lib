"""Unit tests for Expected SARSA (on-policy)."""

import pytest

from my_rl_lib.learning.on_policy.expected_sarsa import expected_sarsa
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues


class TestExpectedSarsaReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = expected_sarsa(
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
        result = expected_sarsa(
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
        result = expected_sarsa(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, EpsilonGreedy)


class TestExpectedSarsaTDUpdate:
    def test_q_value_updated_with_expected_bootstrap(self, make_simple_env, basic_initializer):
        # Expected SARSA: TD target = R + gamma * E_pi[Q(S', *)]
        # With all Q(1,*)=0 and epsilon=0: E_pi[Q(1,*)] = Q(1, greedy_action) = 0.0
        # TD target = 1.0 + 1.0 * 0.0 = 1.0
        # new Q(0, 0) = 1.0
        env = make_simple_env()
        result = expected_sarsa(
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
        result = expected_sarsa(
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
        result = expected_sarsa(
            env,
            num_episodes=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 1)) == pytest.approx(0.0)
        assert result.values.get_value((1, 0)) == pytest.approx(0.0)


class TestExpectedSarsaValidation:
    def test_zero_episodes_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError):
            expected_sarsa(
                env,
                num_episodes=0,
                alpha=1.0,
                gamma=1.0,
                epsilon=0.0,
                initializer=basic_initializer,
            )

    def test_gamma_zero_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError, match="gamma"):
            expected_sarsa(
                env,
                num_episodes=1,
                alpha=1.0,
                gamma=0.0,
                epsilon=0.0,
                initializer=basic_initializer,
            )

    def test_epsilon_out_of_range_raises(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        with pytest.raises(ValueError, match="epsilon"):
            expected_sarsa(
                env,
                num_episodes=1,
                alpha=1.0,
                gamma=1.0,
                epsilon=1.5,
                initializer=basic_initializer,
            )
