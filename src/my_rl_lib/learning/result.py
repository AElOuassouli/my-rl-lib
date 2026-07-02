from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from my_rl_lib.types import ActionT, StateT

if TYPE_CHECKING:
    from my_rl_lib.policies.abstract import Policy
    from my_rl_lib.values.action_state import ActionStateValues


@dataclass
class LearningResult(Generic[StateT, ActionT]):
    """Result returned by all single-value-function learning algorithms."""

    values: ActionStateValues[StateT, ActionT]
    policy: Policy[StateT, ActionT]


@dataclass
class DoubleQLearningResult(Generic[StateT, ActionT]):
    """Result returned by Double Q-Learning, which maintains two value functions."""

    values_a: ActionStateValues[StateT, ActionT]
    values_b: ActionStateValues[StateT, ActionT]
    policy: Policy[StateT, ActionT]
