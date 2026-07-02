"""Metrics collection for RL training."""

import atexit
import time
import warnings
from typing import Any, Literal, Set

from my_rl_lib.metrics.backends.base import LoggingBackend
from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers import (
    AgentAnimationHandler,
    EpisodeRewardHandler,
    EpisodesPerSecondHandler,
    EpisodeStepsHandler,
    EpsilonHandler,
    ImportanceRatioHandler,
    MetricHandler,
    PolicyVisualizationHandler,
    StateVisitationHeatmapHandler,
    TDErrorHandler,
    TrainedModelHandler,
    TrainingTimeHandler,
    ValueChangeHandler,
    ValueFunctionHeatmapHandler,
)
from my_rl_lib.metrics.metric_type import ArtifactType, MediaType, MetricType
from my_rl_lib.metrics.settings import MetricCollectionSettings

# Default handlers for standard metrics
DEFAULT_HANDLERS: dict[MetricType, MetricHandler] = {
    MetricType.EPISODE_REWARD: EpisodeRewardHandler(),
    MetricType.EPISODE_STEPS: EpisodeStepsHandler(),
    MetricType.TD_ERROR: TDErrorHandler(),
    MetricType.VALUE_CHANGE: ValueChangeHandler(),
    MetricType.EPSILON: EpsilonHandler(),
    MetricType.IMPORTANCE_RATIO: ImportanceRatioHandler(),
}

# Media handlers (users opt-in via track parameter)
MEDIA_HANDLERS: dict[str, MetricHandler] = {
    MediaType.STATE_VISITATION_HEATMAP: StateVisitationHeatmapHandler(),
    MediaType.VALUE_FUNCTION_HEATMAP: ValueFunctionHeatmapHandler(),
}

# End-of-training handlers (users opt-in via track_at_end parameter).
# Stored as classes because they are configurable — instantiated with the
# per-item config supplied in track_at_end.
END_OF_TRAINING_HANDLERS: dict[Any, type[MetricHandler]] = {
    MediaType.GREEDY_AGENT_ANIMATION: AgentAnimationHandler,
    MediaType.POLICY_VISUALIZATION: PolicyVisualizationHandler,
    ArtifactType.TRAINED_MODEL: TrainedModelHandler,
    MetricType.TRAINING_TIME: TrainingTimeHandler,
    MetricType.EPISODES_PER_SECOND: EpisodesPerSecondHandler,
}


class MetricsCollector:
    """
    Collects training metrics with optional backend logging.

    Supports both in-memory storage and async logging to backends (e.g., TensorBoard).
    Handlers can be configured per-metric with custom logging frequencies.
    """

    def __init__(
        self,
        track: (
            dict[MetricType | MediaType, MetricCollectionSettings | int]
            | list[MetricType | MediaType]
            | None
        ) = None,
        track_at_end: dict[Any, Any] | list[Any] | None = None,
        backend: LoggingBackend | None = None,
        custom_handlers: dict[str, MetricHandler] | None = None,
        keep_in_memory: bool | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize metrics collector.

        Args:
            track: Periodic metrics to track. A dict mapping each metric/media
                   type to a MetricCollectionSettings (logging frequency + an
                   optional handler override). An int value is accepted as
                   shorthand for MetricCollectionSettings(frequency=int). Can be:
                   - dict[MetricType | MediaType, MetricCollectionSettings | int]
                   - list[MetricType | MediaType]: all logged every episode
                   - None: Default metrics (episode_reward, episode_steps) every episode
            track_at_end: Artifacts produced once at the end of training (in the
                   backend worker process, off the training thread). Can be:
                   - dict[MetricType | MediaType | ArtifactType, config | None]:
                     each item mapped to a typed config model (e.g. AnimationConfig)
                     or None for defaults
                   - list[...]: all items with default configuration
                   - None: nothing produced at end of training
            backend: Optional logging backend (e.g., MLflowBackend)
            custom_handlers: Custom handlers for user-defined metrics
            keep_in_memory: Whether to store metrics in memory.
                           If None, auto-disable when backend is provided.
            batch_size: Episodes to batch before flushing to backend. When None,
                       uses backend.preferred_batch_size (e.g. 1 for MLflow live
                       updates, 100 otherwise). Higher values reduce queue pressure.

        Example:
            >>> collector = MetricsCollector(
            ...     track={MetricType.EPISODE_REWARD: 1, MediaType.VALUE_FUNCTION_HEATMAP: 500},
            ...     track_at_end={
            ...         MediaType.GREEDY_AGENT_ANIMATION: AnimationConfig(number_episodes=3),
            ...         MediaType.POLICY_VISUALIZATION: None,
            ...         ArtifactType.TRAINED_MODEL: TrainedModelConfig(model_name="q_values"),
            ...     },
            ...     backend=MLflowBackend(experiment_name="exp1"),
            ... )
        """
        # Parse track parameter into per-metric settings.
        # Each value may be a MetricCollectionSettings, or an int (shorthand for
        # MetricCollectionSettings(frequency=int)).
        if track is None:
            clean_track: dict[MetricType | MediaType, MetricCollectionSettings] = {
                MetricType.EPISODE_REWARD: MetricCollectionSettings(),
                MetricType.EPISODE_STEPS: MetricCollectionSettings(),
            }
        elif isinstance(track, list):
            clean_track = {metric: MetricCollectionSettings() for metric in track}
        else:
            clean_track = {
                metric: (
                    value
                    if isinstance(value, MetricCollectionSettings)
                    else MetricCollectionSettings(frequency=value)
                )
                for metric, value in track.items()
            }

        self._metric_frequencies: dict[MetricType | MediaType, int] = {
            metric: settings.frequency for metric, settings in clean_track.items()
        }

        # Resolve batch_size: use backend's preference when not explicitly set
        if batch_size is None:
            batch_size = backend.preferred_batch_size if backend is not None else 100

        # Build handler registry (periodic handlers)
        self._handlers: dict[MetricType | MediaType, MetricHandler] = {}

        for metric_key, settings in clean_track.items():
            if settings.handler is not None:
                # Per-metric handler override (configured animation, etc.)
                self._handlers[metric_key] = settings.handler
            elif isinstance(metric_key, MetricType) and metric_key in DEFAULT_HANDLERS:
                # Use default handler
                self._handlers[metric_key] = DEFAULT_HANDLERS[metric_key]
            elif isinstance(metric_key, MediaType) and metric_key in MEDIA_HANDLERS:
                # Use built-in media handler
                self._handlers[metric_key] = MEDIA_HANDLERS[metric_key]
            elif custom_handlers and metric_key in custom_handlers:
                # Backward-compatible custom-handler map
                self._handlers[metric_key] = custom_handlers[metric_key]
            elif metric_key in END_OF_TRAINING_HANDLERS:
                # Render-capable handlers (animation, policy viz, ...) can also be
                # used periodically; instantiate with default config. Provide a
                # configured instance via the settings handler to override.
                self._handlers[metric_key] = END_OF_TRAINING_HANDLERS[metric_key]()
            # If no handler found, we'll try direct context lookup later

        # Build end-of-training handler registry, instantiating with per-item config
        self._final_handlers: dict[Any, MetricHandler] = {}
        if track_at_end is None:
            end_track: dict[Any, Any] = {}
        elif isinstance(track_at_end, list):
            end_track = {item: None for item in track_at_end}
        else:
            end_track = track_at_end

        for end_key, config in end_track.items():
            handler_cls = END_OF_TRAINING_HANDLERS.get(end_key)
            if handler_cls is not None:
                self._final_handlers[end_key] = (
                    handler_cls(config=config) if config is not None else handler_cls()
                )
            elif custom_handlers and end_key in custom_handlers:
                self._final_handlers[end_key] = custom_handlers[end_key]

        # Required keys for periodic handlers (serialized/shipped every matching episode)
        self._episode_required_keys: Set[ContextKey] = set()
        for handler in self._handlers.values():
            self._episode_required_keys.update(handler.required_keys)

        # Required keys for end-of-training handlers (raw-cached, snapshotted at close)
        self._final_required_keys: Set[ContextKey] = set()
        for handler in self._final_handlers.values():
            self._final_required_keys.update(handler.required_keys)

        # Union exposed to training loops so they know what context to pass
        self._all_required_keys: Set[ContextKey] = (
            self._episode_required_keys | self._final_required_keys
        )

        # Create mapping: ContextKey -> list of periodic handlers that need it
        self._key_to_handlers: dict[ContextKey, list[MetricType | MediaType]] = {}
        for metric_key, handler in self._handlers.items():
            for context_key in handler.required_keys:
                if context_key not in self._key_to_handlers:
                    self._key_to_handlers[context_key] = []
                self._key_to_handlers[context_key].append(metric_key)

        # Raw references to context needed by end-of-training handlers.
        # Snapshotted (deep-copied) at close, so we hold the final trained state.
        self._final_context_cache: dict[ContextKey, Any] = {}
        self._start_time = time.time()
        self._episode_count = 0
        self._last_episode = 0

        self.backend = backend

        # Batching for backend logging (reduces queue pressure)
        self._batch_size = batch_size
        self._episode_batch: list[dict[str, Any]] = []

        # Auto-detect memory storage need
        if keep_in_memory is None:
            keep_in_memory = backend is None

        # In-memory storage (optional)
        self.metrics: dict[str, list[float]] | None = None
        self._metadata: dict[str, list[dict[str, int]]] | None = None

        if keep_in_memory:
            # Only store scalar metrics in memory
            scalar_metrics = {
                metric_key
                for metric_key in clean_track.keys()
                if isinstance(metric_key, MetricType)
            }
            self.metrics = {metric.value: [] for metric in scalar_metrics}
            self._metadata = {metric.value: [] for metric in scalar_metrics}

        # Register cleanup
        if backend:
            atexit.register(self.close)

    def is_tracking(self, metric: MetricType) -> bool:
        """Check if a metric is being tracked."""
        return metric in self._metric_frequencies

    def on_episode_end(self, episode: int, **context_data: Any) -> None:
        """
        Record episode data.

        Args:
            episode: Episode number
            **context_data: Episode context using ContextKey names.
                           e.g., episode_reward=100, policy=my_policy

        Example:
            >>> collector.on_episode_end(
            ...     episode=1,
            ...     episode_reward=100.0,
            ...     episode_steps=50,
            ...     policy=policy,
            ...     state_visits=visits
            ... )
        """
        # Convert kwargs to ContextKey dict
        context: dict[ContextKey, Any] = {ContextKey.EPISODE: episode}

        for key, value in context_data.items():
            try:
                context_key = ContextKey(key)
                context[context_key] = value
            except ValueError:
                # Unknown key - warn and skip
                warnings.warn(f"Unknown context key: '{key}'. Skipping.", RuntimeWarning)

        # Validate that keys required by periodic handlers are present
        missing_keys = self._episode_required_keys - context.keys()
        if missing_keys:
            # Determine which handlers are affected
            affected_metrics = set()
            for key in missing_keys:
                affected_metrics.update(self._key_to_handlers.get(key, []))

            if affected_metrics:
                warnings.warn(
                    f"Missing context keys {missing_keys} required by metrics: {affected_metrics}",
                    RuntimeWarning,
                )

        # Cache raw references for end-of-training handlers; snapshotted at close
        # so we retain the final trained env/values without extra per-episode work.
        for context_key in self._final_required_keys:
            if context_key in context:
                self._final_context_cache[context_key] = context[context_key]

        self._last_episode = episode
        self._episode_count += 1

        # Extract and serialize raw data ONCE for periodic handlers (deduplicated)
        extracted_data: dict[ContextKey, Any] = {}

        for context_key in self._episode_required_keys:
            if context_key in context:
                # Apply serialization logic based on type
                extracted_data[context_key] = self._serialize_value(
                    context_key, context[context_key]
                )

        # Store scalars in memory if enabled
        if self.metrics is not None and self._metadata is not None:
            for context_key in [
                ContextKey.EPISODE_REWARD,
                ContextKey.EPISODE_STEPS,
                ContextKey.TD_ERROR,
                ContextKey.VALUE_CHANGE,
                ContextKey.EPSILON,
                ContextKey.IMPORTANCE_RATIO,
            ]:
                if context_key in context:
                    metric_name = context_key.value
                    if metric_name in self.metrics:
                        self.metrics[metric_name].append(context[context_key])
                        self._metadata[metric_name].append({"episode": episode})

        # Determine which metrics to log this episode (based on frequency)
        handlers_to_run: dict[MetricType | MediaType, MetricHandler] = {}

        for metric_key, frequency in self._metric_frequencies.items():
            if episode % frequency == 0 and metric_key in self._handlers:
                handlers_to_run[metric_key] = self._handlers[metric_key]

        # Send to backend if there are handlers to run
        if self.backend and handlers_to_run:
            # Don't send handler objects (expensive to pickle)
            # Send only the data needed to recreate handlers in worker
            handlers_info: dict[MetricType | MediaType, Any] = {}
            for handler_key, handler in handlers_to_run.items():
                handlers_info[handler_key] = {
                    "class": handler.__class__,
                    "required_keys": list(handler.required_keys),
                    "config": getattr(handler, "config", None),
                }

            # Only ship context keys actually needed by this episode's handlers
            episode_required_keys: set[ContextKey] = set()
            for handler in handlers_to_run.values():
                episode_required_keys.update(handler.required_keys)

            episode_data = {
                "episode": episode,
                "context_data": {
                    k: v for k, v in extracted_data.items() if k in episode_required_keys
                },
                "handlers_info": handlers_info,
            }

            # Media/artifact handlers are infrequent but important, so they must
            # not be dropped under queue pressure (unlike high-frequency scalars,
            # where the drop-on-full strategy protects training throughput).
            # Deliver such episodes with a guaranteed (blocking) enqueue.
            contains_media = any(
                isinstance(key, (MediaType, ArtifactType)) for key in handlers_to_run
            )
            if contains_media:
                # Preserve ordering of any pending droppable episodes first.
                self._flush_batch()
                self.backend.log_final({"type": "batch", "episodes": [episode_data]})
            else:
                # Add to batch
                self._episode_batch.append(episode_data)

                # Send batch when full
                if len(self._episode_batch) >= self._batch_size:
                    self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush accumulated episode batch to backend."""
        if self.backend and self._episode_batch:
            batch_data = {
                "type": "batch",
                "episodes": self._episode_batch,
            }
            self.backend.log_episode(batch_data)
            self._episode_batch.clear()

    def close(self) -> None:
        """Close the collector and flush any remaining data."""
        # Flush remaining batch
        self._flush_batch()

        # Produce end-of-training artifacts (rendered in the worker, off the
        # training thread) BEFORE shutting the backend down so the run is active.
        self._log_end_of_training()

        # Close backend
        if self.backend:
            self.backend.close()

    def _log_end_of_training(self) -> None:
        """Enqueue end-of-training handlers as a final batch for the worker.

        Snapshots the cached env/values, injects the collector-computed final
        scalars, and ships everything to the backend. All rendering / model
        registration then happens in the worker process.
        """
        if not (self.backend and self._final_handlers):
            return

        training_time = time.time() - self._start_time
        episodes_per_second = self._episode_count / training_time if training_time > 0 else 0.0

        # Snapshot cached context (deep-copy pydantic env/values) + computed scalars
        end_context: dict[ContextKey, Any] = {}
        for context_key, value in self._final_context_cache.items():
            end_context[context_key] = self._serialize_value(context_key, value)
        end_context[ContextKey.TRAINING_TIME] = training_time
        end_context[ContextKey.EPISODES_PER_SECOND] = episodes_per_second

        handlers_info: dict[Any, Any] = {}
        running_required_keys: set[ContextKey] = set()
        for handler_key, handler in self._final_handlers.items():
            missing_keys = handler.required_keys - end_context.keys()
            if missing_keys:
                warnings.warn(
                    f"Skipping end-of-training '{handler_key}': "
                    f"missing context keys {missing_keys}.",
                    RuntimeWarning,
                )
                continue
            handlers_info[handler_key] = {
                "class": handler.__class__,
                "required_keys": list(handler.required_keys),
                "config": getattr(handler, "config", None),
            }
            running_required_keys.update(handler.required_keys)

        if not handlers_info:
            return

        episode_data = {
            "episode": self._last_episode,
            "context_data": {k: v for k, v in end_context.items() if k in running_required_keys},
            "handlers_info": handlers_info,
        }
        # Guaranteed delivery — end-of-training artifacts must not be dropped.
        self.backend.log_final({"type": "batch", "episodes": [episode_data]})

    def _serialize_value(self, key: ContextKey, value: Any) -> Any:
        """
        Serialize context value for IPC (pickling to worker process).

        Different context keys may need different serialization:
        - Scalars: pass as-is
        - Policy/Environment: serialize state/config
        - Arrays: copy to ensure no shared state

        Args:
            key: Context key type
            value: Value to serialize

        Returns:
            Serialized value safe for pickling
        """
        # Serialization logic based on key type
        if key == ContextKey.POLICY:
            # Check if policy has get_state method
            if hasattr(value, "get_state"):
                return value.get_state()
            else:
                # Fallback: try to pickle as-is
                return value

        elif key == ContextKey.ENVIRONMENT:
            # Ship a deep snapshot so worker-side rendering is isolated from the
            # live env being mutated during training (avoids the Queue feeder race).
            if hasattr(value, "model_copy"):
                return value.model_copy(deep=True)
            elif hasattr(value, "get_config"):
                return value.get_config()
            else:
                return value

        elif key == ContextKey.VALUE_FUNCTION:
            # Prefer a deep snapshot of the Values object (needed for rendering /
            # model registration); fall back to a dict copy for non-pydantic values.
            if hasattr(value, "model_copy"):
                return value.model_copy(deep=True)
            elif hasattr(value, "get_all_values"):
                return value.get_all_values()
            elif hasattr(value, "values"):
                # For action-state values, get the underlying dict
                return dict(value.values) if hasattr(value.values, "items") else value
            else:
                return value

        elif key in [
            ContextKey.STATE_VISITS,
            ContextKey.ACTION_HISTORY,
            ContextKey.TRAJECTORY,
            ContextKey.Q_VALUES,
        ]:
            if isinstance(value, dict):
                return dict(value)
            elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                return list(value)
            else:
                return value

        else:
            # Scalars and simple types - pass through
            return value

    def on_step(self, episode: int, step: int, **metrics: float) -> None:
        """
        Record step-level metrics (in-memory only).

        Step-level metrics are not sent to backends, only stored in memory.
        Use on_episode_end for metrics you want to visualize in TensorBoard.

        Args:
            episode: Current episode number
            step: Current step number
            **metrics: Metric values with keys matching MetricType.value
        """
        if self.metrics is None or self._metadata is None:
            return  # No in-memory storage

        for metric_name, value in metrics.items():
            try:
                metric_type = MetricType(metric_name)
                if self.is_tracking(metric_type) and metric_name in self.metrics:
                    self.metrics[metric_name].append(value)
                    self._metadata[metric_name].append({"episode": episode, "step": step})
            except ValueError:
                # Skip unknown metrics
                pass

    def get_metric(self, metric: MetricType) -> list[float]:
        """
        Get all recorded values for a specific metric.

        Requires keep_in_memory=True.

        Args:
            metric: The metric to retrieve

        Returns:
            List of recorded values

        Raises:
            RuntimeError: If in-memory storage is disabled
        """
        if self.metrics is None:
            raise RuntimeError(
                "Metrics not stored in memory. "
                "Initialize with keep_in_memory=True to use this method."
            )
        return self.metrics.get(metric.value, [])

    def get_last_value(self, metric: MetricType) -> float | None:
        """
        Get the most recent value for a metric.

        Requires keep_in_memory=True.
        """
        if self.metrics is None:
            raise RuntimeError(
                "Metrics not stored in memory. "
                "Initialize with keep_in_memory=True to use this method."
            )
        values = self.get_metric(metric)
        return values[-1] if values else None

    def get_summary(self, metric: MetricType) -> dict[str, float]:
        """
        Get summary statistics for a metric.

        Requires keep_in_memory=True.

        Returns:
            Dict with keys: mean, std, min, max, count
        """
        if self.metrics is None:
            raise RuntimeError(
                "Metrics not stored in memory. "
                "Initialize with keep_in_memory=True to use this method."
            )

        values = self.get_metric(metric)
        if not values:
            return {}

        import statistics

        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def export(self) -> dict[str, Any]:
        """
        Export all metrics in a serializable format.

        Requires keep_in_memory=True.

        Returns:
            Dict containing all metrics and their metadata
        """
        if self.metrics is None:
            raise RuntimeError(
                "Metrics not stored in memory. "
                "Initialize with keep_in_memory=True to use this method."
            )

        return {
            "metrics": self.metrics,
            "metadata": self._metadata,
            "tracked": list(self._metric_frequencies.keys()),
        }

    def reset(self) -> None:
        """
        Clear all recorded metrics.

        Requires keep_in_memory=True.
        """
        if self.metrics is None or self._metadata is None:
            raise RuntimeError(
                "Metrics not stored in memory. "
                "Initialize with keep_in_memory=True to use this method."
            )

        for metric_name in self.metrics:
            self.metrics[metric_name].clear()
            self._metadata[metric_name].clear()

    def __enter__(self) -> "MetricsCollector":
        """Context manager support."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Ensure cleanup on context exit."""
        self.close()
        return False
