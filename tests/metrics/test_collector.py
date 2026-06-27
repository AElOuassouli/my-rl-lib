"""Unit tests for MetricsCollector."""

import pytest
from unittest.mock import MagicMock

from my_rl_lib.metrics.collector import MetricsCollector
from my_rl_lib.metrics.metric_type import MetricType


class TestInMemoryRecording:
    def test_records_episode_reward(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.on_episode_end(0, episode_reward=10.0)
        assert collector.get_metric(MetricType.EPISODE_REWARD) == [10.0]

    def test_multiple_episodes_accumulated(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        for i, r in enumerate([1.0, 2.0, 3.0]):
            collector.on_episode_end(i, episode_reward=r)
        assert collector.get_metric(MetricType.EPISODE_REWARD) == [1.0, 2.0, 3.0]

    def test_records_multiple_metric_types(self):
        collector = MetricsCollector(
            track=[MetricType.EPISODE_REWARD, MetricType.EPISODE_STEPS],
            keep_in_memory=True,
        )
        collector.on_episode_end(0, episode_reward=5.0, episode_steps=20)
        assert collector.get_metric(MetricType.EPISODE_REWARD) == [5.0]
        assert collector.get_metric(MetricType.EPISODE_STEPS) == [20]


class TestGetSummary:
    def test_mean_min_max_count(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        for i, r in enumerate([2.0, 4.0, 6.0]):
            collector.on_episode_end(i, episode_reward=r)
        summary = collector.get_summary(MetricType.EPISODE_REWARD)
        assert summary["mean"] == pytest.approx(4.0)
        assert summary["min"] == pytest.approx(2.0)
        assert summary["max"] == pytest.approx(6.0)
        assert summary["count"] == 3

    def test_empty_metric_returns_empty_dict(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        summary = collector.get_summary(MetricType.EPISODE_REWARD)
        assert summary == {}


class TestGetLastValue:
    def test_returns_most_recent_value(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.on_episode_end(0, episode_reward=5.0)
        collector.on_episode_end(1, episode_reward=9.0)
        assert collector.get_last_value(MetricType.EPISODE_REWARD) == pytest.approx(9.0)

    def test_returns_none_when_empty(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        assert collector.get_last_value(MetricType.EPISODE_REWARD) is None


class TestExport:
    def test_export_contains_metrics_and_metadata(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.on_episode_end(0, episode_reward=7.0)
        exported = collector.export()
        assert "metrics" in exported
        assert "metadata" in exported
        assert "tracked" in exported

    def test_export_metrics_match_recorded(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.on_episode_end(0, episode_reward=3.0)
        exported = collector.export()
        assert exported["metrics"]["episode_reward"] == [3.0]


class TestReset:
    def test_reset_clears_all_metrics(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.on_episode_end(0, episode_reward=7.0)
        collector.reset()
        assert collector.get_metric(MetricType.EPISODE_REWARD) == []

    def test_reset_without_memory_raises(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=False)
        with pytest.raises(RuntimeError):
            collector.reset()


class TestClose:
    def test_close_does_not_raise(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True)
        collector.close()


class TestContextManager:
    def test_context_manager_calls_close_on_exit(self):
        with MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=True) as c:
            c.on_episode_end(0, episode_reward=1.0)
        # No exception means __exit__ (close) was called successfully


class TestWithoutMemory:
    def test_get_metric_raises_when_no_memory(self):
        collector = MetricsCollector(track=[MetricType.EPISODE_REWARD], keep_in_memory=False)
        with pytest.raises(RuntimeError, match="keep_in_memory"):
            collector.get_metric(MetricType.EPISODE_REWARD)

    def test_auto_disable_memory_when_backend_provided(self):
        mock_backend = MagicMock()
        collector = MetricsCollector(
            track=[MetricType.EPISODE_REWARD],
            backend=mock_backend,
        )
        # keep_in_memory defaults to False when backend is provided
        assert collector.metrics is None


class TestOnStep:
    def test_records_td_error_per_step(self):
        collector = MetricsCollector(track=[MetricType.TD_ERROR], keep_in_memory=True)
        collector.on_step(0, 0, td_error=0.5)
        collector.on_step(0, 1, td_error=0.3)
        values = collector.get_metric(MetricType.TD_ERROR)
        assert 0.5 in values
        assert 0.3 in values

    def test_on_step_without_memory_does_not_raise(self):
        collector = MetricsCollector(track=[MetricType.TD_ERROR], keep_in_memory=False)
        collector.on_step(0, 0, td_error=0.5)  # should silently no-op


class TestFrequencyControlWithBackend:
    def test_backend_receives_batched_episodes(self):
        mock_backend = MagicMock()
        # batch_size=5: backend is called once per 5 eligible episodes
        collector = MetricsCollector(
            track=[MetricType.EPISODE_REWARD],
            backend=mock_backend,
            batch_size=5,
        )
        for i in range(10):
            collector.on_episode_end(i, episode_reward=float(i))
        collector.close()
        # 10 episodes in 2 batches of 5 → 2 backend calls
        assert mock_backend.log_episode.call_count == 2
