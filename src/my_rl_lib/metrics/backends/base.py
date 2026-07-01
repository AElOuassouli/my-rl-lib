"""Abstract base class for logging backends."""

from abc import ABC, abstractmethod
from typing import Any


class LoggingBackend(ABC):
    """
    Abstract interface for metrics logging backends.

    Backends handle the persistence of metrics data (e.g., TensorBoard, WandB, etc.).
    """

    @property
    def preferred_batch_size(self) -> int:
        """Episodes to accumulate before flushing. Backends can override this."""
        return 100

    @abstractmethod
    def log_episode(self, episode_data: dict[str, Any]) -> None:
        """
        Log data for a single episode.

        Args:
            episode_data: Episode record containing:
                - episode: int - Episode number
                - context_data: dict[ContextKey, Any] - Extracted context (deduplicated)
                - handlers: dict[str, MetricHandler] - Handlers to run
        """
        pass

    def log_final(self, episode_data: dict[str, Any]) -> None:
        """Log critical end-of-training data with guaranteed delivery.

        Unlike log_episode (which may drop data under load to avoid blocking
        training), this is used once at close() for end-of-training artifacts,
        so it must not be dropped. The default simply delegates to log_episode;
        async backends should override to block until the data is accepted.
        """
        self.log_episode(episode_data)

    @abstractmethod
    def close(self) -> None:
        """
        Cleanup and flush any remaining data.

        Should ensure all buffered data is written before returning.
        """
        pass
