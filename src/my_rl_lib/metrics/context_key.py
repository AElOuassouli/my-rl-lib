"""
Standardized context keys for episode data.

These keys are used to pass data from training code to metric handlers.
Handlers declare which keys they need, and the collector extracts the corresponding data.
"""

from enum import Enum


class ContextKey(str, Enum):
    """
    Standardized keys for episode context data.

    These keys are used to pass data to MetricHandlers.
    Handlers declare which keys they need via their required_keys property.
    """

    # Episode info
    EPISODE = "episode"

    # Scalar metrics (computed values from training)
    EPISODE_REWARD = "episode_reward"
    EPISODE_STEPS = "episode_steps"
    TD_ERROR = "td_error"
    VALUE_CHANGE = "value_change"
    EPSILON = "epsilon"
    IMPORTANCE_RATIO = "importance_ratio"

    # Training artifacts (for media generation)
    POLICY = "policy"
    VALUE_FUNCTION = "value_function"
    ENVIRONMENT = "environment"
    STATE_VISITS = "state_visits"
    ACTION_HISTORY = "action_history"
    Q_VALUES = "q_values"
    TRAJECTORY = "trajectory"

    # Pre-rendered data (optional, if user wants to render in main process)
    RENDERED_FRAMES = "rendered_frames"
    STATE_HEATMAP_DATA = "state_heatmap_data"
