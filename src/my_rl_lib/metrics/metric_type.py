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

    # End-of-training scalars (computed by the collector at close)
    TRAINING_TIME = "training_time"
    EPISODES_PER_SECOND = "episodes_per_second"


class MediaType(str, Enum):
    # Policy plots
    VALUE_FUNCTION_HEATMAP = "value_function_heatmap"

    # Values
    STATE_VISITATION_HEATMAP = "state_visitation_heatmap"

    # Rendered artifacts (typically produced once at end of training)
    GREEDY_AGENT_ANIMATION = "greedy_agent_animation"
    POLICY_VISUALIZATION = "policy_visualization"


class ArtifactType(str, Enum):
    """Non-metric, non-media artifacts produced at end of training."""

    # Register the trained value function in the backend's model registry
    TRAINED_MODEL = "trained_model"
