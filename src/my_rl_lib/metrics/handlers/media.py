"""Metric handlers for media (images, videos)."""

from typing import Any, Set

from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers.base import MetricHandler


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

        # Create heatmap
        heatmap = np.zeros((max_y, max_x), dtype=np.float32)
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

        value_dict = context_data[ContextKey.VALUE_FUNCTION]

        if not value_dict:
            return {"type": "image", "tag": "value_function", "data": np.zeros((1, 1))}

        # Convert dict to 2D grid
        states = list(value_dict.keys())
        max_x = max(s[0] for s in states) + 1
        max_y = max(s[1] for s in states) + 1

        heatmap = np.zeros((max_y, max_x), dtype=np.float32)
        for state, value in value_dict.items():
            x, y = state
            heatmap[y, x] = value

        return {"type": "image", "tag": "value_function", "data": heatmap}
