"""Unit tests for n-step tree backup (off-policy)."""

from unittest.mock import MagicMock

import pytest

from my_rl_lib.learning.off_policy.n_step_tree_backup import n_step_tree_backup
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.values.action_state import ActionStateValues


def _make_behavior_policy(env, initializer):
    """Build a deterministic (epsilon=0) EpsilonGreedy behavior policy."""
    values = ActionStateValues()
    values.init_from_environment(env, initializer)
    policy = EpsilonGreedy(epsilon=0.0)
    policy.init_from_environment_and_values(env, values)
    return policy


class TestNStepTreeBackupReturnType:
    def test_returns_learning_result(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, LearningResult)

    def test_result_has_action_state_values(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
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
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert isinstance(result.policy, Greedy)


class TestNStepTreeBackupUpdate:
    def test_n1_update_matches_expected_sarsa(self, make_simple_env, basic_initializer):
        # n=1, single-step episode: G = R_1 = 1.0 (T=1, no bootstrap)
        # new Q(0, 0) = 0 + 1.0 * (1.0 - 0) = 1.0
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(1.0)

    def test_alpha_scales_the_update(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=0.5,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 0)) == pytest.approx(0.5)

    def test_untouched_q_values_remain_zero(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert result.values.get_value((0, 1)) == pytest.approx(0.0)
        assert result.values.get_value((1, 0)) == pytest.approx(0.0)

    def test_terminates_for_multi_step_n(self, make_simple_env, basic_initializer):
        # Regression test: the loop must advance t each iteration so that it
        # terminates even when n exceeds the episode length.
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=3,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, LearningResult)


class TestNStepTreeBackupMetricsCollector:
    def test_on_step_and_on_episode_end_are_called(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        metrics_collector = MagicMock(spec=MetricsCollector)

        n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=2,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
            metrics_collector=metrics_collector,
        )

        assert metrics_collector.on_step.called
        assert metrics_collector.on_episode_end.call_count == 2

    def test_on_episode_end_receives_reward_and_steps(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        metrics_collector = MagicMock(spec=MetricsCollector)

        n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
            metrics_collector=metrics_collector,
        )

        _, kwargs = metrics_collector.on_episode_end.call_args
        assert kwargs["episode_reward"] == pytest.approx(1.0)
        assert kwargs["episode_steps"] == pytest.approx(1.0)

    def test_on_step_receives_td_error_and_value_change(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        metrics_collector = MagicMock(spec=MetricsCollector)

        n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
            metrics_collector=metrics_collector,
        )

        _, kwargs = metrics_collector.on_step.call_args
        assert "td_error" in kwargs
        assert "value_change" in kwargs

    def test_no_metrics_collector_does_not_raise(self, make_simple_env, basic_initializer):
        env = make_simple_env()
        behavior = _make_behavior_policy(env, basic_initializer)
        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=basic_initializer,
        )
        assert isinstance(result, LearningResult)
