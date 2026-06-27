"""Unit tests for importance sampling utilities."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.policies.importance_sampling import (
    compute_importance_sampling_ratio,
    compute_importance_sampling_ratio_circular_buffer,
)
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep


def _policy_with_probs(probs: dict):
    """Mock policy returning specified probabilities for (state, action) pairs."""
    policy = MagicMock()
    policy.get_action_probability_given_state.side_effect = lambda state, action: probs.get(
        (state, action), 0.0
    )
    return policy


def _store_with_steps(n: int, steps: list[tuple]) -> EpisodeStepsCircularStore:
    """Build a circular store from a list of (state, action) pairs."""
    store = EpisodeStepsCircularStore(n=n)
    for t, (state, action) in enumerate(steps):
        store.set_step(t, LearningStep(state=state, action=action, reward=0.0))
    return store


class TestComputeImportanceSamplingRatio:
    def test_ratio_equals_target_over_behavior(self):
        target = _policy_with_probs({(0, 0): 0.8})
        behavior = _policy_with_probs({(0, 0): 0.4})
        ratio = compute_importance_sampling_ratio(target, behavior, state=0, action=0)
        assert ratio == pytest.approx(2.0)

    def test_equal_policies_give_ratio_one(self):
        target = _policy_with_probs({(0, 0): 0.5})
        behavior = _policy_with_probs({(0, 0): 0.5})
        ratio = compute_importance_sampling_ratio(target, behavior, state=0, action=0)
        assert ratio == pytest.approx(1.0)

    def test_target_zero_gives_ratio_zero(self):
        target = _policy_with_probs({(0, 0): 0.0})
        behavior = _policy_with_probs({(0, 0): 0.5})
        ratio = compute_importance_sampling_ratio(target, behavior, state=0, action=0)
        assert ratio == pytest.approx(0.0)

    def test_zero_behavior_probability_raises(self):
        target = _policy_with_probs({(0, 0): 0.5})
        behavior = _policy_with_probs({(0, 0): 0.0})
        with pytest.raises(ValueError, match="zero"):
            compute_importance_sampling_ratio(target, behavior, state=0, action=0)


class TestComputeImportanceSamplingRatioCircularBuffer:
    def test_product_of_two_ratios(self):
        # ratio = (0.8/0.4) * (0.6/0.3) = 2.0 * 2.0 = 4.0
        target = _policy_with_probs({(0, 0): 0.8, (1, 1): 0.6})
        behavior = _policy_with_probs({(0, 0): 0.4, (1, 1): 0.3})
        store = _store_with_steps(n=3, steps=[(0, 0), (1, 1)])
        ratio = compute_importance_sampling_ratio_circular_buffer(
            target, behavior, store, lower_bound=0, upper_bound=1
        )
        assert ratio == pytest.approx(4.0)

    def test_single_step_equals_single_ratio(self):
        target = _policy_with_probs({(0, 0): 0.9})
        behavior = _policy_with_probs({(0, 0): 0.3})
        store = _store_with_steps(n=2, steps=[(0, 0)])
        ratio = compute_importance_sampling_ratio_circular_buffer(
            target, behavior, store, lower_bound=0, upper_bound=0
        )
        assert ratio == pytest.approx(3.0)

    def test_equal_policies_give_ratio_one(self):
        target = _policy_with_probs({(0, 0): 0.5, (1, 0): 0.5})
        behavior = _policy_with_probs({(0, 0): 0.5, (1, 0): 0.5})
        store = _store_with_steps(n=3, steps=[(0, 0), (1, 0)])
        ratio = compute_importance_sampling_ratio_circular_buffer(
            target, behavior, store, lower_bound=0, upper_bound=1
        )
        assert ratio == pytest.approx(1.0)

    def test_lower_greater_than_upper_raises(self):
        target = _policy_with_probs({})
        behavior = _policy_with_probs({})
        store = EpisodeStepsCircularStore(n=2)
        with pytest.raises(ValueError, match="[Ll]ower"):
            compute_importance_sampling_ratio_circular_buffer(
                target, behavior, store, lower_bound=2, upper_bound=1
            )

    def test_missing_store_entry_raises(self):
        target = _policy_with_probs({(0, 0): 0.5})
        behavior = _policy_with_probs({(0, 0): 0.5})
        store = EpisodeStepsCircularStore(n=3)  # empty store
        with pytest.raises(ValueError):
            compute_importance_sampling_ratio_circular_buffer(
                target, behavior, store, lower_bound=0, upper_bound=0
            )
