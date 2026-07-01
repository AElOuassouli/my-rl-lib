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

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


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
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = "default",
        run_name: str | None = None,
        max_queue_size: int = 100,
        async_logging: bool = True,
        shutdown_timeout: float = 180.0,
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
            shutdown_timeout: Max seconds to wait at close() for the worker to
                          finish draining before force-terminating. Generous by
                          default because end-of-training work (rendering,
                          model registration) can take tens of seconds.
        """
        import mlflow

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.async_logging = async_logging
        self.shutdown_timeout = shutdown_timeout

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

    @property
    def preferred_batch_size(self) -> int:
        return 1

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
            self._process_batch_or_episode(self._mlflow, episode_data, np, handler_cache)

    def log_final(self, episode_data: dict[str, Any]) -> None:
        """Log end-of-training data with guaranteed delivery (never dropped).

        Called once at close() after training, so a blocking enqueue is fine —
        it waits for worker capacity rather than dropping under a full queue.
        """
        if self.async_logging and self.queue is not None:
            # Blocking put: guarantees the end-of-training batch reaches the worker.
            self.queue.put(episode_data)
        else:
            import numpy as np

            handler_cache: dict[str, Any] = {}
            self._process_batch_or_episode(self._mlflow, episode_data, np, handler_cache)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an MLflow artifact on the active run.

        Intended for post-training artifacts (plots, animations). The run is
        always active in the main process between __init__ and close().

        Args:
            local_path: Path to the local file to log.
            artifact_path: Optional subdirectory within the artifact store.
        """
        import mlflow

        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def register_trained_model(self, model_name: str, values: Any) -> None:
        """Serialize Q-values and register them in the MLflow Model Registry.

        Main-process entry point (kept for direct/manual use). The same logic
        runs in the worker via the ``model`` result type.

        Args:
            model_name: Name under which to register the model.
            values: ActionStateValues instance; its .values dict is serialized.
        """
        import mlflow

        MLflowBackend._log_model(mlflow, model_name, values)

    @staticmethod
    def _log_model(mlflow_module: Any, model_name: str, values: Any) -> None:
        """Log a pyfunc model artifact holding the Q-values and register it.

        Runs in whichever process is logging (main or worker); the MLflow run
        is active in both.
        """
        import json
        import os

        class _QValueModel(mlflow_module.pyfunc.PythonModel):  # type: ignore[misc]
            def load_context(self, context: Any) -> None:
                with open(context.artifacts["q_values"]) as f:
                    self._q_values = json.load(f)

            def predict(self, context: Any, model_input: Any) -> Any:
                return self._q_values

        serializable: dict[str, Any] = {
            str(state): {str(a): v for a, v in action_values.items()}
            for state, action_values in values.values.items()
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="q_values_", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(serializable, tmp_file)

        try:
            mlflow_module.pyfunc.log_model(
                artifact_path="model",
                python_model=_QValueModel(),
                artifacts={"q_values": tmp_path},
                registered_model_name=model_name,
            )
        finally:
            os.remove(tmp_path)

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

            self.process.join(timeout=self.shutdown_timeout)

            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
                warnings.warn(
                    "MLflow logging process terminated forcefully after "
                    f"{self.shutdown_timeout}s. Some metrics/artifacts may not "
                    "have been written; increase shutdown_timeout if needed.",
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

                    MLflowBackend._process_batch_or_episode(mlflow, data, np, handler_cache)

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
    def _process_batch_or_episode(
        mlflow_module: Any,
        data: dict[str, Any],
        np: Any,
        handler_cache: dict[str, Any],
    ) -> None:
        """Dispatch a queue payload that may be a single episode or a batch."""
        if isinstance(data, dict) and data.get("type") == "batch":
            for ep_data in data.get("episodes", []):
                MLflowBackend._process_episode(mlflow_module, ep_data, np, handler_cache)
        else:
            MLflowBackend._process_episode(mlflow_module, data, np, handler_cache)

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
                    config = handler_info.get("config")
                    handler_cache[metric_key] = (
                        handler_class(config=config) if config is not None else handler_class()
                    )

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

                            import os

                            # Use a stable, human-readable artifact filename
                            # (mlflow.log_artifact keeps the file's basename).
                            tmp_dir = tempfile.mkdtemp(prefix="video_")
                            gif_path = os.path.join(tmp_dir, f"{result['tag']}.gif")

                            imageio.mimsave(gif_path, frame_array, fps=fps)
                            mlflow_module.log_artifact(
                                gif_path,
                                artifact_path=f"videos/{result['tag']}",
                            )

                            os.remove(gif_path)
                            os.rmdir(tmp_dir)

                        except ImportError:
                            warnings.warn(
                                "imageio is required for video logging. "
                                "Install with: pip install imageio",
                                RuntimeWarning,
                            )

                elif result["type"] == "model":
                    MLflowBackend._log_model(mlflow_module, result["tag"], result["values"])

            except Exception as e:
                warnings.warn(f"Handler '{metric_key}' failed: {e}. Skipping.", RuntimeWarning)
