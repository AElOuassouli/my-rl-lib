"""Unit tests for n-step tree backup (off-policy)."""

from unittest.mock import MagicMock

import pytest

from my_rl_lib.environments.abstract import StepResult
from my_rl_lib.learning.off_policy.n_step_tree_backup import n_step_tree_backup
from my_rl_lib.learning.result import LearningResult
from my_rl_lib.metrics import MetricsCollector
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy
from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.values.action_state import ActionStateValues
from my_rl_lib.values.initializer import Initializer, InitializerType


def _make_behavior_policy(env, initializer):
    """Build a deterministic (epsilon=0) EpsilonGreedy behavior policy."""
    values = ActionStateValues()
    values.init_from_environment(env, initializer)
    policy = EpsilonGreedy(epsilon=0.0)
    policy.init_from_environment_and_values(env, values)
    return policy


def _make_chain_env():
    """Deterministic 4-state chain: 0->1->2->3(terminal), single action, rewards 1,2,3.

    A single action per state makes the tree-backup branch sum collapse to the
    chosen action's term, so G_tau reduces to the plain discounted return
    R_1 + gamma*R_2 + gamma^2*R_3 — letting the recursion's step-by-step
    correctness be checked against a hand-computed value.
    """
    env = MagicMock()
    env.get_states.return_value = [0, 1, 2, 3]
    env.get_terminal_states.return_value = [3]
    env.get_actions_per_state.return_value = {0: [0], 1: [0], 2: [0], 3: [0]}
    env.get_current_possible_actions.return_value = [0]
    env.current_state = 0

    rewards_by_state = {0: 1.0, 1: 2.0, 2: 3.0}
    next_state_by_state = {0: 1, 1: 2, 2: 3}

    def reset_effect():
        env.current_state = 0

    env.reset.side_effect = reset_effect

    def step_effect(action):
        reward = rewards_by_state[env.current_state]
        env.current_state = next_state_by_state[env.current_state]
        return StepResult(next_state=env.current_state, reward=reward)

    env.step.side_effect = step_effect
    env.is_current_state_terminal.side_effect = lambda: env.current_state == 3

    return env


def _make_action_dependent_reward_env():
    """Single-step 2-state env where the reward depends on the action taken.

    Action 0 -> reward +1.0, action 1 -> reward -1.0. Lets a test drive the
    behavior policy's choice at t=0 and observe the resulting value update.
    """
    env = MagicMock()
    env.get_states.return_value = [0, 1]
    env.get_terminal_states.return_value = [1]
    env.get_actions_per_state.return_value = {0: [0, 1], 1: [0, 1]}
    env.get_current_possible_actions.return_value = [0, 1]
    env.current_state = 0

    def reset_effect():
        env.current_state = 0

    env.reset.side_effect = reset_effect

    reward_by_action = {0: 1.0, 1: -1.0}

    def step_effect(action):
        env.current_state = 1
        return StepResult(next_state=1, reward=reward_by_action[action])

    env.step.side_effect = step_effect
    env.is_current_state_terminal.side_effect = lambda: env.current_state == 1

    return env


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

    def test_behavior_policy_tracks_learned_values_across_episodes(self):
        # The behavior policy must track the same learned action-values as the
        # target policy (it is never used in the return computation, so this
        # cannot affect correctness — only how quickly exploration improves).
        # Behavior initially (wrongly) prefers action 1 in state 0, independent
        # of the algorithm's own zero-initialized values. Action 1 earns a
        # -1.0 reward, so after episode 1's update, Q(0,0)=0.0 > Q(0,1)=-1.0
        # and the behavior policy's greedy pick for state 0 should flip to
        # action 0 for episode 2.
        env = _make_action_dependent_reward_env()
        initializer = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=0.0,
            constant_value_non_terminal=0.0,
        )

        behavior_values = ActionStateValues()
        behavior_values.init_from_environment(env, initializer)
        behavior_values.set_value((0, 0), 0.0)
        behavior_values.set_value((0, 1), 5.0)
        behavior = EpsilonGreedy(epsilon=0.0)
        behavior.init_from_environment_and_values(env, behavior_values)

        n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=2,
            n=1,
            alpha=1.0,
            gamma=1.0,
            initializer=initializer,
        )

        actions_taken = [call.kwargs["action"] for call in env.step.call_args_list]
        assert actions_taken == [1, 0]

    def test_multi_step_return_matches_hand_computed_value(self):
        # Regression test for the backward-recursion bug where the loop index
        # `k` was never decremented, so every iteration reused the same step
        # instead of walking S_2, then S_1. With a single action per state the
        # tree-backup branch sum collapses to the chosen action's term, so
        # G_0 must equal the plain discounted return:
        #   G_0 = R_1 + gamma*R_2 + gamma^2*R_3 = 1 + 0.5*2 + 0.25*3 = 2.75
        env = _make_chain_env()
        initializer = Initializer(
            initializer_type=InitializerType.CONSTANT,
            terminal_states_value=0.0,
            constant_value_non_terminal=0.0,
        )
        behavior = _make_behavior_policy(env, initializer)

        result = n_step_tree_backup(
            env,
            behavior_policy=behavior,
            num_episodes=1,
            n=3,
            alpha=1.0,
            gamma=0.5,
            initializer=initializer,
        )

        assert result.values.get_value((0, 0)) == pytest.approx(2.75)


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
