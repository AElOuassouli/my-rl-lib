"""Abstract base class for logging backends."""

from abc import ABC, abstractmethod
from typing import Any


class LoggingBackend(ABC):
    """
    Abstract interface for metrics logging backends.

    Backends handle the persistence of metrics data (e.g., TensorBoard, WandB, etc.).
    """

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

    @abstractmethod
    def close(self) -> None:
        """
        Cleanup and flush any remaining data.

        Should ensure all buffered data is written before returning.
        """
        pass
