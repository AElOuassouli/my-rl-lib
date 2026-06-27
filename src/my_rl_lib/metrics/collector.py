"""Metrics collection for RL training."""

import atexit
import warnings
from typing import Any, Literal, Set

from my_rl_lib.metrics.backends.base import LoggingBackend
from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers import (
    EpisodeRewardHandler,
    EpisodeStepsHandler,
    EpsilonHandler,
    ImportanceRatioHandler,
    MetricHandler,
    StateVisitationHeatmapHandler,
    TDErrorHandler,
    ValueChangeHandler,
    ValueFunctionHeatmapHandler,
)
from my_rl_lib.metrics.metric_type import MediaType, MetricType

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


class MetricsCollector:
    """
    Collects training metrics with optional backend logging.

    Supports both in-memory storage and async logging to backends (e.g., TensorBoard).
    Handlers can be configured per-metric with custom logging frequencies.
    """

    def __init__(
        self,
        track: dict[MetricType | MediaType, int] | list[MetricType | MediaType] | None = None,
        backend: LoggingBackend | None = None,
        custom_handlers: dict[str, MetricHandler] | None = None,
        keep_in_memory: bool | None = None,
        batch_size: int = 100,
    ):
        """
        Initialize metrics collector.

        Args:
            track: Metrics to track with logging frequency.
                   Can be:
                   - dict[MetricType | str, int]: {metric: log_every_n_episodes}
                   - list[MetricType]: All metrics logged every episode
                   - None: Default metrics (episode_reward, episode_steps) every episode
            backend: Optional logging backend (e.g., TensorBoardBackend)
            custom_handlers: Custom handlers for user-defined metrics
            keep_in_memory: Whether to store metrics in memory.
                           If None, auto-disable when backend is provided.
            batch_size: Number of episodes to batch before sending to backend (default: 100)
                       Higher values reduce queue pressure but increase memory usage
                           If None, auto-disable when backend is provided.

        Example:
            >>> collector = MetricsCollector(
            ...     track={
            ...         MetricType.EPISODE_REWARD: 1,
            ...         MetricType.EPSILON: 10,
            ...         'state_visitation_heatmap': 50
            ...     },
            ...     backend=TensorBoardBackend(log_dir="runs/exp1")
            ... )
        """
        # Parse track parameter
        clean_track: dict[MetricType | MediaType, int] = {}
        if track is None:
            clean_track = {MetricType.EPISODE_REWARD: 1, MetricType.EPISODE_STEPS: 1}
        elif isinstance(track, list):
            # Convert list to dict (all with frequency=1)
            clean_track = {metric: 1 for metric in track}

        self._metric_frequencies = clean_track

        # Build handler registry
        self._handlers: dict[MetricType | MediaType, MetricHandler] = {}

        for metric_key in clean_track.keys():
            if isinstance(metric_key, MetricType) and metric_key in DEFAULT_HANDLERS:
                # Use default handler
                self._handlers[metric_key] = DEFAULT_HANDLERS[metric_key]
            elif isinstance(metric_key, MediaType) and metric_key in MEDIA_HANDLERS:
                # Use built-in media handler
                self._handlers[metric_key] = MEDIA_HANDLERS[metric_key]
            elif custom_handlers and metric_key in custom_handlers:
                # Use custom handler
                self._handlers[metric_key] = custom_handlers[metric_key]
            # If no handler found, we'll try direct context lookup later

        # Compute ALL required keys upfront (union across all handlers)
        self._all_required_keys: Set[ContextKey] = set()
        for handler in self._handlers.values():
            self._all_required_keys.update(handler.required_keys)

        # Create mapping: ContextKey -> list of handlers that need it
        self._key_to_handlers: dict[ContextKey, list[MetricType | MediaType]] = {}
        for metric_key, handler in self._handlers.items():
            for context_key in handler.required_keys:
                if context_key not in self._key_to_handlers:
                    self._key_to_handlers[context_key] = []
                self._key_to_handlers[context_key].append(metric_key)

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

        # Validate that required keys are present
        missing_keys = self._all_required_keys - context.keys()
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

        # Extract and serialize raw data ONCE for all handlers (deduplicated)
        extracted_data: dict[ContextKey, Any] = {}

        for context_key in self._all_required_keys:
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
                }

            episode_data = {
                "episode": episode,
                "context_data": extracted_data,  # Deduplicated context
                "handlers_info": handlers_info,  # Handler metadata, not objects
            }

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

        # Close backend
        if self.backend:
            self.backend.close()

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
            # Check if environment has get_config method
            if hasattr(value, "get_config"):
                return value.get_config()
            else:
                return value

        elif key == ContextKey.VALUE_FUNCTION:
            # Serialize value function
            if hasattr(value, "get_all_values"):
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
            # Lists/arrays - copy to avoid shared state
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
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
