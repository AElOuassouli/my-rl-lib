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
    # Cap on greedy-rollout length per episode. Early in training the greedy
    # policy is near-random and rarely reaches the goal, so a lower cap keeps
    # animation generation fast instead of running the full rollout each time.
    max_steps: PositiveInt = 1000


class PolicyVizConfig(BaseModel):
    """Configuration for the policy-visualization handler.

    Intentionally free of environment-specific render options so the metrics
    package stays decoupled from concrete environments; the environment uses
    its own sensible default rendering options.
    """

    # Cap on the greedy-trajectory rollout length (see AnimationConfig.max_steps).
    max_steps: PositiveInt = 1000


class TrainedModelConfig(BaseModel):
    """Configuration for registering the trained value function."""

    model_name: str = "trained_model"
