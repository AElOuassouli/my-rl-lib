from enum import Enum


class MetricType(str, Enum):
    """Available metrics that can be tracked during training."""

    # Episode-level metrics
    EPISODE_REWARD = "episode_reward"
    EPISODE_STEPS = "episode_steps"

    # Learning metrics
    TD_ERROR = "td_error"
    VALUE_CHANGE = "value_change"

    # Policy metrics
    EPSILON = "epsilon"

    # Off-policy specific
    IMPORTANCE_RATIO = "importance_ratio"


class MediaType(str, Enum):
    # Policy plots
    VALUE_FUNCTION_HEATMAP = "value_function_heatmap"

    # Values
    STATE_VISITATION_HEATMAP = "state_visitation_heatmap"
