from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from my_rl_lib.values.initializer import Initializer

if TYPE_CHECKING:
    from my_rl_lib.environments.abstract import Environment


class ValuesType(str, Enum):
    STATE_VALUES = "state_values"
    ACTION_STATE_VALUES = "action_state_values"


class Values(BaseModel, ABC):
    """Abstract base class for value representations in reinforcement learning."""

    values: dict[Any, Any] | None = None
    type: ValuesType

    @abstractmethod
    def init_from_environment(
        self,
        environment: "Environment",
        initializer: Initializer,
    ) -> None:
        """Initialize values from the environment using the provided initializer."""

        pass

    @abstractmethod
    def get_value(self, entry: Any) -> float:
        """Get the value for a given state or state-action pair."""
        pass

    @abstractmethod
    def set_value(self, entry: Any, value: float) -> None:
        """Set the value for a given state or state-action pair."""
        pass

    @abstractmethod
    def get_max_action_and_value(self, state: Any) -> tuple[Any, float]:
        """Get the action with the maximum value for a given state."""
        pass

    @abstractmethod
    def get_expected_value(
        self,
        state: Any,
        policy: Any,
    ) -> float:
        """Get the expected value for a given state under a specified policy."""
        pass

    @abstractmethod
    def add(self, other: "Values") -> "Values":
        """Add two Values objects together."""
        pass
