from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_rl_lib.policies.abstract import Policy
    from my_rl_lib.values.action_state import ActionStateValues


@dataclass
class LearningResult:
    """Result returned by all single-value-function learning algorithms."""

    values: ActionStateValues
    policy: Policy


@dataclass
class DoubleQLearningResult:
    """Result returned by Double Q-Learning, which maintains two value functions."""

    values_a: ActionStateValues
    values_b: ActionStateValues
    policy: Policy
