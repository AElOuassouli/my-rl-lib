"""
Abstract base class for metric handlers.

Handlers process episode context data and generate loggable outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Set

from my_rl_lib.metrics.context_key import ContextKey


class MetricHandler(ABC):
    """
    Base class for metric handlers.

    Each handler processes specific episode context data and produces
    output suitable for logging to TensorBoard or other backends.

    Handlers are stateless and define their requirements declaratively.
    """

    @property
    @abstractmethod
    def required_keys(self) -> Set[ContextKey]:
        """
        Context keys required by this handler.

        This is a static declaration - same for all instances.
        The collector uses this to determine what data to extract.

        Returns:
            Set of ContextKey values needed for processing
        """
        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        Unique name for this metric.

        This identifies the metric and is used for logging.

        Returns:
            Metric name string
        """
        pass

    @abstractmethod
    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        """
        Process context data into loggable format.

        This method runs in the WORKER process (can be expensive).
        It receives only the data specified in required_keys.

        Args:
            context_data: Subset of episode context containing required_keys
            episode: Episode number

        Returns:
            Dict with output specification:
            - For scalars: {'type': 'scalar', 'tag': str, 'value': float}
            - For images: {'type': 'image', 'tag': str, 'data': np.ndarray}
            - For videos: {'type': 'video', 'tag': str, 'frames': list, 'fps': int}
        """
        pass
