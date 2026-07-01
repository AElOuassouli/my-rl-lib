"""Metric handlers for media (images, videos)."""

from typing import Any, Set

from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers.base import MetricHandler
from my_rl_lib.metrics.handlers.config import (
    AnimationConfig,
    PolicyVizConfig,
    TrainedModelConfig,
)


def _as_value_dict(value: Any) -> Any:
    """Normalize a value-function payload to a plain ``{state: value}`` dict.

    Accepts either a serialized dict or a ``Values`` object (which is what the
    collector now ships as a snapshot).
    """
    if isinstance(value, dict):
        return value
    if hasattr(value, "get_all_values"):
        return value.get_all_values()
    if hasattr(value, "values") and not callable(value.values):
        return value.values
    return value


class StateVisitationHeatmapHandler(MetricHandler):
    """
    Handler for state visitation heatmap.

    Visualizes which states the agent visited during the episode.
    Assumes states are 2D grid coordinates (x, y).
    """

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.STATE_VISITS}

    @property
    def metric_name(self) -> str:
        return "state_visitation_heatmap"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        import numpy as np

        state_visits = context_data[ContextKey.STATE_VISITS]

        if not state_visits:
            # Return empty heatmap if no visits
            return {"type": "image", "tag": "state_heatmap", "data": np.zeros((1, 1))}

        # Auto-detect grid dimensions from visits
        max_x = max(s[0] for s in state_visits) + 1
        max_y = max(s[1] for s in state_visits) + 1

        # Create heatmap — supports both list and counter dict {state: count}
        heatmap = np.zeros((max_y, max_x), dtype=np.float32)
        if isinstance(state_visits, dict):
            for state, count in state_visits.items():
                x, y = state
                heatmap[y, x] = count
        else:
            for state in state_visits:
                x, y = state
                heatmap[y, x] += 1

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return {"type": "image", "tag": "state_heatmap", "data": heatmap}


class ValueFunctionHeatmapHandler(MetricHandler):
    """
    Handler for value function visualization.

    Shows learned state values as a heatmap.
    Assumes value_function is a dict {state: value} with 2D states.
    """

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.VALUE_FUNCTION}

    @property
    def metric_name(self) -> str:
        return "value_function_heatmap"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        import numpy as np

        value_dict = _as_value_dict(context_data[ContextKey.VALUE_FUNCTION])

        if not value_dict:
            return {"type": "image", "tag": "value_function", "data": np.zeros((1, 1))}

        states = list(value_dict.keys())
        max_x = max(s[0] for s in states) + 1
        max_y = max(s[1] for s in states) + 1

        heatmap = np.zeros((max_y, max_x), dtype=np.float32)
        for state, value in value_dict.items():
            x, y = state
            # Support both state values (scalar) and action-state values (dict)
            heatmap[y, x] = max(value.values()) if isinstance(value, dict) else value

        return {"type": "image", "tag": "value_function", "data": heatmap}


class PolicyVisualizationHandler(MetricHandler):
    """Render the learned policy (values + greedy trajectory) as an image.

    Requires a render-capable environment exposing ``render_policy_array`` and
    the value function. Runs in the worker process; both the environment and
    values are shipped as snapshots by the collector.
    """

    def __init__(self, config: PolicyVizConfig | None = None) -> None:
        super().__init__(config or PolicyVizConfig())

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.ENVIRONMENT, ContextKey.VALUE_FUNCTION}

    @property
    def metric_name(self) -> str:
        return "policy_visualization"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        environment = context_data[ContextKey.ENVIRONMENT]
        values = context_data[ContextKey.VALUE_FUNCTION]

        image = environment.render_policy_array(values)

        return {"type": "image", "tag": "policy_visualization", "data": image}


class AgentAnimationHandler(MetricHandler):
    """Render an animation of a greedy agent acting in the environment.

    Requires a render-capable environment exposing ``render_greedy_agent_frames``.
    Runs in the worker process.
    """

    def __init__(self, config: AnimationConfig | None = None) -> None:
        super().__init__(config or AnimationConfig())

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.ENVIRONMENT, ContextKey.VALUE_FUNCTION}

    @property
    def metric_name(self) -> str:
        return "greedy_agent_animation"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        environment = context_data[ContextKey.ENVIRONMENT]
        values = context_data[ContextKey.VALUE_FUNCTION]

        frames = environment.render_greedy_agent_frames(
            values, number_episodes=self.config.number_episodes
        )

        return {
            "type": "video",
            "tag": "greedy_agent_animation",
            "frames": frames,
            "fps": self.config.fps,
        }


class TrainedModelHandler(MetricHandler):
    """Emit the trained value function for registration in the backend registry."""

    def __init__(self, config: TrainedModelConfig | None = None) -> None:
        super().__init__(config or TrainedModelConfig())

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.VALUE_FUNCTION}

    @property
    def metric_name(self) -> str:
        return "trained_model"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        values = context_data[ContextKey.VALUE_FUNCTION]

        return {"type": "model", "tag": self.config.model_name, "values": values}
