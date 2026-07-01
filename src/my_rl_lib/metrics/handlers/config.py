"""Typed configuration models for configurable metric handlers.

These are Pydantic models so they are both well-typed and picklable, which
lets the collector ship a handler's configuration to the worker process where
the handler is re-instantiated and run.
"""

from pydantic import BaseModel, PositiveInt


class AnimationConfig(BaseModel):
    """Configuration for the greedy-agent animation handler."""

    number_episodes: PositiveInt = 3
    fps: PositiveInt = 10


class PolicyVizConfig(BaseModel):
    """Configuration for the policy-visualization handler.

    Intentionally free of environment-specific render options so the metrics
    package stays decoupled from concrete environments; the environment uses
    its own sensible default rendering options.
    """


class TrainedModelConfig(BaseModel):
    """Configuration for registering the trained value function."""

    model_name: str = "trained_model"
