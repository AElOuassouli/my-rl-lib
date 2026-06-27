"""MLflow backend for metrics logging."""

from __future__ import annotations

import atexit
import multiprocessing
import queue
import tempfile
import time
import warnings
from typing import Any

from my_rl_lib.metrics.backends.base import LoggingBackend


class MLflowBackend(LoggingBackend):
    """
    MLflow logging backend with async processing.

    Uses a separate process to handle all I/O operations, ensuring
    training performance is not impacted by logging overhead.

    Features:
    - Non-blocking episode logging
    - Handlers run in worker process (media generation doesn't block training)
    - Automatic cleanup on exit
    - Drop strategy when queue is full (never blocks training)
    - Scalars logged via mlflow.log_metric
    - Images logged as PNG artifacts via mlflow.log_image
    - Videos logged as GIF artifacts via imageio
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "default",
        run_name: str | None = None,
        max_queue_size: int = 100,
        async_logging: bool = True,
    ):
        """
        Initialize MLflow backend.

        Args:
            tracking_uri: MLflow tracking URI (local path or server URL)
            experiment_name: MLflow experiment name
            run_name: Optional name for this run
            max_queue_size: Max episodes in queue before dropping
            async_logging: Enable async logging (multiprocessing).
                          Set to False for debugging (runs synchronously).
        """
        import mlflow

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.async_logging = async_logging

        # Create the MLflow run in the main process
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._active_run = mlflow.start_run(run_name=run_name)
        self._run_id = self._active_run.info.run_id

        if async_logging:
            self.queue: multiprocessing.Queue[Any] | None = multiprocessing.Queue(
                maxsize=max_queue_size
            )

            self.process: multiprocessing.Process | None = multiprocessing.Process(
                target=self._logging_worker,
                args=(self.queue, self._run_id, tracking_uri),
                daemon=True,
            )
            self.process.start()

            atexit.register(self.close)

            # Wait for worker to warm up
            print("Waiting 3 seconds for MLflow worker warm-up...")
            time.sleep(3.0)
        else:
            import mlflow as _mlflow

            self._mlflow = _mlflow
            self.queue = None
            self.process = None

    def log_episode(self, episode_data: dict[str, Any]) -> None:
        """
        Log episode data (non-blocking).

        Args:
            episode_data: Contains episode, context_data, and handlers_info
        """
        if self.async_logging and self.queue is not None:
            try:
                self.queue.put_nowait(episode_data)
            except queue.Full:
                warnings.warn(
                    "MLflow queue full, dropping episode. "
                    "Consider increasing max_queue_size or reducing logging frequency.",
                    RuntimeWarning,
                )
        else:
            import numpy as np

            handler_cache: dict[str, Any] = {}
            self._process_episode(self._mlflow, episode_data, np, handler_cache)

    def close(self) -> None:
        """Shutdown logging process, flush remaining data, and end MLflow run."""
        if (
            self.async_logging
            and self.process
            and self.process.is_alive()
            and self.queue is not None
        ):
            try:
                self.queue.put(None, timeout=1.0)
            except queue.Full:
                pass

            self.process.join(timeout=10.0)

            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
                warnings.warn(
                    "MLflow logging process terminated forcefully. "
                    "Some metrics may not have been written.",
                    RuntimeWarning,
                )

        # End the MLflow run in the main process
        import mlflow

        mlflow.end_run()

    @staticmethod
    def _logging_worker(
        episode_queue: multiprocessing.Queue[Any],
        run_id: str,
        tracking_uri: str,
    ) -> None:
        """
        Persistent worker process that handles all I/O operations.

        Resumes the MLflow run created in the main process using the run_id,
        then processes episodes from the queue until shutdown.

        Args:
            episode_queue: Queue for receiving episode data
            run_id: MLflow run ID to resume
            tracking_uri: MLflow tracking URI
        """
        import numpy as np

        import mlflow

        mlflow.set_tracking_uri(tracking_uri)

        handler_cache: dict[str, Any] = {}

        try:
            with mlflow.start_run(run_id=run_id):
                while True:
                    data = episode_queue.get()

                    if data is None:
                        break

                    if isinstance(data, dict) and data.get("type") == "batch":
                        for ep_data in data.get("episodes", []):
                            MLflowBackend._process_episode(mlflow, ep_data, np, handler_cache)
                    else:
                        MLflowBackend._process_episode(mlflow, data, np, handler_cache)

        except Exception as e:
            import traceback

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".log", prefix="mlflow_error_", delete=False
                ) as f:
                    f.write(f"MLflow logging worker crashed: {e}\n")
                    traceback.print_exc(file=f)
            except Exception:
                pass
            raise

    @staticmethod
    def _process_episode(
        mlflow_module: Any,
        episode_data: dict[str, Any],
        np: Any,
        handler_cache: dict[str, Any],
    ) -> None:
        """
        Process a single episode in the worker.

        Runs handlers and writes results to MLflow.

        Args:
            mlflow_module: The mlflow module (passed to avoid repeated imports)
            episode_data: Episode data with context and handlers_info
            np: numpy module
            handler_cache: Cache of instantiated handlers
        """
        episode = episode_data["episode"]
        context_data = episode_data.get("context_data", {})
        handlers_info = episode_data.get("handlers_info", {})

        for metric_key, handler_info in handlers_info.items():
            try:
                if metric_key not in handler_cache:
                    handler_class = handler_info["class"]
                    handler_cache[metric_key] = handler_class()

                handler = handler_cache[metric_key]

                handler_context = {
                    key: context_data[key] for key in handler.required_keys if key in context_data
                }

                missing_keys = handler.required_keys - handler_context.keys()
                if missing_keys:
                    warnings.warn(
                        f"Handler '{metric_key}' missing required keys: {missing_keys}. Skipping.",
                        RuntimeWarning,
                    )
                    continue

                result = handler.process(handler_context, episode)

                if result["type"] == "scalar":
                    mlflow_module.log_metric(result["tag"], result["value"], step=episode)

                elif result["type"] == "image":
                    image = result["data"]
                    if not isinstance(image, np.ndarray):
                        image = np.array(image)

                    # Normalize to uint8 for PNG export
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)

                    # Ensure (H, W, C) or (H, W) — mlflow.log_image accepts both
                    mlflow_module.log_image(
                        image,
                        artifact_file=f"images/{result['tag']}/step_{episode:06d}.png",
                    )

                elif result["type"] == "video":
                    frames = result.get("frames", [])
                    fps = result.get("fps", 30)

                    if frames:
                        try:
                            import imageio

                            if isinstance(frames[0], np.ndarray):
                                frame_array = [
                                    f.astype(np.uint8) if f.dtype != np.uint8 else f for f in frames
                                ]
                            else:
                                frame_array = frames

                            with tempfile.NamedTemporaryFile(
                                suffix=".gif", delete=False
                            ) as tmp_file:
                                tmp_path = tmp_file.name

                            imageio.mimsave(tmp_path, frame_array, fps=fps)
                            mlflow_module.log_artifact(
                                tmp_path,
                                artifact_path=f"videos/{result['tag']}",
                            )

                            import os

                            os.remove(tmp_path)

                        except ImportError:
                            warnings.warn(
                                "imageio is required for video logging. "
                                "Install with: pip install imageio",
                                RuntimeWarning,
                            )

            except Exception as e:
                warnings.warn(f"Handler '{metric_key}' failed: {e}. Skipping.", RuntimeWarning)
