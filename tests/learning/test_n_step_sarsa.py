"""Unit tests for n-step SARSA (on-policy)."""

import pytest

from my_rl_lib.learning.on_policy.n_step_sarsa import n_step_sarsa
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.values.action_state import ActionStateValues


class TestNStepSarsaReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, LearningResult)

    def test_result_has_action_state_values(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.values, ActionStateValues)
        assert result.values.values is not None

    def test_result_has_epsilon_greedy_policy(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, EpsilonGreedy)


class TestNStepSarsaUpdate:
    def test_n1_update_matches_sarsa(self, make_simple_env, basic_initializer):
        # n=1: single-step return with no bootstrap (episode ends after 1 step)
        # G = R_1 = 1.0 (T=1, tau+n=1, so tau+n >= T → no bootstrap)
        # new Q(0, 0) = 0 + 1.0 * (1.0 - 0) = 1.0
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(1.0)

    def test_alpha_scales_the_update(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(0.5)

    def test_untouched_q_values_remain_zero(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        result = n_step_sarsa(
            env,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            epsilon=0.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 1)) == pytest.approx(0.0)
        assert result.values.get_value((1, 0)) == pytest.approx(0.0)
