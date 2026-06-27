"""Unit tests for n-step SARSA off-policy with importance sampling."""

import pytest
from unittest.mock import patch

from my_rl_lib.learning.off_policy.n_step_sarsa_off_policy import n_step_sarsa_off_policy
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.values.action_state import ActionStateValues

# Patch target: IS function as imported in the algorithm's module
_IS_PATCH = (
    "my_rl_lib.learning.off_policy.n_step_sarsa_off_policy"
    ".compute_importance_sampling_ratio_circular_buffer"
)


def _make_behavior_policy(env, initializer):
    """Build a deterministic (epsilon=0) EpsilonGreedy behavior policy."""
    values = ActionStateValues()
    values.init_from_environment(env, initializer)
    policy = EpsilonGreedy(epsilon=0.0)
    policy.init_from_environment_and_values(env, values)
    return policy


class TestNStepSarsaOffPolicyReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=1.0):
            result = n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        assert isinstance(result, LearningResult)

    def test_result_has_action_state_values(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=1.0):
            result = n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        assert isinstance(result.values, ActionStateValues)
        assert result.values.values is not None

    def test_target_policy_is_greedy(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=1.0):
            result = n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        assert isinstance(result.policy, Greedy)


class TestNStepSarsaOffPolicyISRatio:
    def test_importance_sampling_function_is_called(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=1.0) as mock_is:
            n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        mock_is.assert_called()

    def test_q_update_scales_with_is_ratio(self, make_simple_env, basic_initializer):
        # With rho=0.5 (mocked): td_error = 0.5 * (G - Q) = 0.5 * 1.0 = 0.5
        # new Q(0, 0) = 0 + 1.0 * 0.5 = 0.5
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=0.5):
            result = n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        assert result.values.get_value((0, 0)) == pytest.approx(0.5)

    def test_q_value_updated_with_rho_one(self, make_simple_env, basic_initializer):
        # rho=1.0: same as on-policy, new Q(0,0) = 1.0
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        with patch(_IS_PATCH, return_value=1.0):
            result = n_step_sarsa_off_policy(
                env,
                num_episodes=1,
                behavior_policy=behavior,
                n=1,
                alpha=1.0,
                gamma=1.0,
                initializer=basic_initializer,
            )
        assert result.values.get_value((0, 0)) == pytest.approx(1.0)
