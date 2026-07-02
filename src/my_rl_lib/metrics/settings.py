"""Per-metric collection settings for the periodic ``track`` parameter."""

from pydantic import BaseModel, ConfigDict, PositiveInt

from my_rl_lib.metrics.handlers.base import MetricHandler


class MetricCollectionSettings(BaseModel):
    """How to collect a single tracked metric/media item during training.

    Attributes:
        frequency: Log this item every ``frequency`` episodes.
        handler: Optional handler override. When set, it is used instead of the
            built-in default handler for this metric/media type — this is how you
            configure things like animation ``number_episodes`` / ``fps`` /
            ``max_steps``. When ``None``, the default handler is used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frequency: PositiveInt = 1
    handler: MetricHandler | None = None
