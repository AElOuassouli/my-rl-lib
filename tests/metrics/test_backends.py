"""Unit tests for MLflowBackend (all mlflow calls are mocked)."""

import pytest
from unittest.mock import MagicMock, patch

from my_rl_lib.metrics.backends.mlflow_backend import MLflowBackend


@pytest.fixture()
def mlflow_mocks():
    """Patch all top-level mlflow calls so no real server or file I/O occurs."""
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", return_value=mock_run) as mock_start,
        patch("mlflow.end_run") as mock_end,
    ):
        yield {"start_run": mock_start, "end_run": mock_end, "run": mock_run}


class TestMLflowBackendConstruction:
    def test_sync_mode_has_no_queue_or_process(self, mlflow_mocks):
        backend = MLflowBackend(async_logging=False)
        assert backend.queue is None
        assert backend.process is None

    def test_run_id_stored_from_active_run(self, mlflow_mocks):
        backend = MLflowBackend(async_logging=False)
        assert backend._run_id == "test-run-id"

    def test_tracking_uri_stored(self, mlflow_mocks):
        backend = MLflowBackend(tracking_uri="my_runs", async_logging=False)
        assert backend.tracking_uri == "my_runs"

    def test_experiment_name_stored(self, mlflow_mocks):
        backend = MLflowBackend(experiment_name="my_exp", async_logging=False)
        assert backend.experiment_name == "my_exp"


class TestMLflowBackendClose:
    def test_close_calls_end_run(self, mlflow_mocks):
        backend = MLflowBackend(async_logging=False)
        backend.close()
        mlflow_mocks["end_run"].assert_called_once()

    def test_close_does_not_raise(self, mlflow_mocks):
        backend = MLflowBackend(async_logging=False)
        backend.close()


class TestMLflowBackendLogEpisode:
    def test_log_episode_in_sync_mode_does_not_raise(self, mlflow_mocks):
        backend = MLflowBackend(async_logging=False)
        # Sync mode passes episode_data directly to _process_episode
        episode_data = {
            "episode": 0,
            "context_data": {},
            "handlers_info": {},
        }
        backend.log_episode(episode_data)

    def test_log_episode_in_async_mode_drops_when_queue_full(self, mlflow_mocks):
        # In async mode, when queue is full, a warning is issued (no exception)
        with (
            patch("multiprocessing.Process") as mock_proc,
            patch("multiprocessing.Queue") as mock_queue_cls,
            patch("time.sleep"),
            patch("atexit.register"),
        ):
            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue
            mock_proc_instance = MagicMock()
            mock_proc.return_value = mock_proc_instance

            import queue

            mock_queue.put_nowait.side_effect = queue.Full

            backend = MLflowBackend(async_logging=True, max_queue_size=1)
            with pytest.warns(RuntimeWarning, match="queue full"):
                backend.log_episode({"type": "batch", "episodes": []})
