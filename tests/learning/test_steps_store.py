"""Unit tests for EpisodeStepsCircularStore."""

import pytest

from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep


class TestEpisodeStepsCircularStore:
    def test_set_and_get_basic(self):
        store = EpisodeStepsCircularStore(n=3)
        step = LearningStep(state=5, action=2, reward=1.5)
        store.set_step(0, step)
        retrieved = store.get_step(0)
        assert retrieved.state == 5
        assert retrieved.action == 2
        assert retrieved.reward == pytest.approx(1.5)

    def test_get_unset_slot_raises(self):
        store = EpisodeStepsCircularStore(n=3)
        with pytest.raises(ValueError, match="None"):
            store.get_step(0)

    def test_circular_wrap_overwrites_earlier_entry(self):
        # n=2 → store size = 3; index is time % 3
        store = EpisodeStepsCircularStore(n=2)
        step_a = LearningStep(state=0, action=0, reward=1.0)
        step_b = LearningStep(state=99, action=99, reward=99.0)
        store.set_step(0, step_a)  # index 0 % 3 = 0
        store.set_step(3, step_b)  # index 3 % 3 = 0 → overwrites time 0
        retrieved = store.get_step(3)
        assert retrieved.state == 99

    def test_get_with_original_index_after_wrap(self):
        # After wrapping, get_step(0) and get_step(3) share the same slot
        store = EpisodeStepsCircularStore(n=2)
        step_b = LearningStep(state=42, action=1, reward=0.5)
        store.set_step(3, step_b)
        # slot 0 now contains step_b
        assert store.get_step(0).state == 42

    def test_multiple_steps_stored_and_retrieved(self):
        store = EpisodeStepsCircularStore(n=4)
        for t in range(5):
            store.set_step(t, LearningStep(state=t * 10, action=t, reward=float(t)))
        for t in range(5):
            assert store.get_step(t).state == t * 10
            assert store.get_step(t).action == t
            assert store.get_step(t).reward == pytest.approx(float(t))

    def test_none_reward_stored_correctly(self):
        store = EpisodeStepsCircularStore(n=2)
        step = LearningStep(state="s0", action="a0", reward=None)
        store.set_step(0, step)
        assert store.get_step(0).reward is None

    def test_store_size_is_n_plus_one(self):
        store = EpisodeStepsCircularStore(n=5)
        assert len(store.store) == 6
