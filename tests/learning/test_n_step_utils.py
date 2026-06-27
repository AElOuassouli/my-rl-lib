"""Unit tests for compute_n_step_return."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.learning.n_step_utils import compute_n_step_return
from my_rl_lib.learning.steps_store import EpisodeStepsCircularStore, LearningStep


def _make_store(n: int, steps: list[tuple]) -> EpisodeStepsCircularStore:
    """Build a store from (state, action, reward) tuples."""
    store = EpisodeStepsCircularStore(n=n)
    for t, (state, action, reward) in enumerate(steps):
        store.set_step(t, LearningStep(state=state, action=action, reward=reward))
    return store


def _make_values(q_dict: dict) -> MagicMock:
    v = MagicMock()
    v.get_value.side_effect = lambda entry: q_dict.get(entry, 0.0)
    return v


class TestComputeNStepReturn:
    def test_n1_no_bootstrap_at_episode_end(self):
        # n=1, tau=0, T=1: episode ended, no bootstrap
        # G = R_1 = 2.0
        store = _make_store(n=1, steps=[(0, 0, None), (1, 0, 2.0)])
        values = _make_values({})
        G = compute_n_step_return(store, n=1, tau=0, T=1, gamma=0.99, values=values)
        assert G == pytest.approx(2.0)

    def test_n1_with_bootstrap(self):
        # n=1, tau=0, T=inf: G = R_1 + gamma * Q(S1, A1) = 1.0 + 0.5*3.0 = 2.5
        store = _make_store(n=2, steps=[(0, 0, None), (1, 0, 1.0)])
        values = _make_values({(1, 0): 3.0})
        G = compute_n_step_return(store, n=1, tau=0, T=float("inf"), gamma=0.5, values=values)
        assert G == pytest.approx(2.5)

    def test_n2_multi_step_return_with_bootstrap(self):
        # n=2, tau=0, T=inf: G = R_1 + gamma*R_2 + gamma^2 * Q(S2, A2)
        store = _make_store(n=3, steps=[(0, 0, None), (1, 0, 1.0), (2, 1, 2.0)])
        values = _make_values({(2, 1): 4.0})
        G = compute_n_step_return(store, n=2, tau=0, T=float("inf"), gamma=0.9, values=values)
        expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 4.0
        assert G == pytest.approx(expected)

    def test_terminal_episode_no_bootstrap(self):
        # n=2, tau=0, T=2 (tau+n == T): no bootstrap
        # G = R_1 + gamma*R_2 = 1.0 + 0.9*3.0 = 3.7
        store = _make_store(n=3, steps=[(0, 0, None), (1, 0, 1.0), (2, 1, 3.0)])
        values = _make_values({(2, 1): 99.0})  # Q-value should NOT be used
        G = compute_n_step_return(store, n=2, tau=0, T=2, gamma=0.9, values=values)
        assert G == pytest.approx(1.0 + 0.9 * 3.0)

    def test_gamma_zero_returns_only_next_reward(self):
        # gamma=0: G = R_1 only (no future discounting)
        store = _make_store(n=2, steps=[(0, 0, None), (1, 0, 5.0), (2, 1, 10.0)])
        values = _make_values({(2, 1): 100.0})
        G = compute_n_step_return(store, n=2, tau=0, T=float("inf"), gamma=0.0, values=values)
        # G = 5.0 + 0.0*10.0 + 0.0^2 * 100.0 = 5.0
        assert G == pytest.approx(5.0)

    def test_n1_bootstrap_at_non_zero_tau(self):
        # tau=1, n=1, T=inf: G = R_2 + gamma * Q(S2, A2)
        store = _make_store(n=3, steps=[(0, 0, None), (1, 0, 1.0), (2, 1, 3.0)])
        values = _make_values({(2, 1): 2.0})
        G = compute_n_step_return(store, n=1, tau=1, T=float("inf"), gamma=0.5, values=values)
        # G = R_2 + 0.5 * Q(S2, A2) = 3.0 + 0.5 * 2.0 = 4.0
        assert G == pytest.approx(4.0)
