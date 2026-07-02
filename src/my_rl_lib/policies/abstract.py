from abc import ABC, abstractmethod
from functools import wraps
from random import choices
from typing import TYPE_CHECKING, Any, Callable, Generic

from pydantic import BaseModel

from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.abstract import Values

if TYPE_CHECKING:
    from my_rl_lib.environments.abstract import Environment

# Default tolerance for probability validation
PROBABILITY_VALIDATION_TOLERANCE = 1e-6


def validate_probabilities(
    tolerance: float = PROBABILITY_VALIDATION_TOLERANCE,
) -> Callable[..., Any]:
    """
    Decorator that validates action probabilities sum to 1 after the decorated method executes.

    Args:
        tolerance: The tolerance for floating-point comparison (default: 1e-6)

    Returns:
        Decorated function that validates probabilities after execution

    Raises:
        ValueError: If the sum of probabilities for any state is not approximately 1
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: "Policy[Any, Any]", *args: Any, **kwargs: Any) -> Any:
            result = func(self, *args, **kwargs)

            if self.action_probabilities_per_state is not None:
                # Check if sum is approximately 1 for each state
                for state, action_probs in self.action_probabilities_per_state.items():
                    total_prob = sum(action_probs.values())
                    if abs(total_prob - 1.0) > tolerance:
                        raise ValueError(
                            f"Sum of action probabilities for state {state} is {total_prob}, "
                            f"which is not within tolerance {tolerance} of 1.0"
                        )

            return result

        return wrapper

    return decorator


class Policy(BaseModel, ABC, Generic[StateT, ActionT]):
    """Abstract base class for policies in reinforcement learning."""

    action_probabilities_per_state: dict[StateT, dict[ActionT, float]] | None = None

    def init_from_environment_and_values(
        self,
        environment: "Environment[StateT, ActionT]",
        values: Values[StateT, ActionT],
    ) -> None:
        """Initialize the policy from the environment and values."""
        actions_per_state = environment.get_actions_per_state()
        self.action_probabilities_per_state = {
            state: {action: 0.0 for action in actions}
            for state, actions in actions_per_state.items()
        }

        self.update_probabilities(values)

    def get_action_probability_given_state(self, state: StateT, action: ActionT) -> float:
        """Get the probability of selecting a given action in a given state."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        if state not in self.action_probabilities_per_state:
            return 0.0

        return self.action_probabilities_per_state[state].get(action, 0.0)

    def select_action(self, state: StateT) -> ActionT:
        """Select an action based on the policy's action probabilities for the given state."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        if state not in self.action_probabilities_per_state:
            raise ValueError(f"No possible actions found for state {state}.")

        state_actions = self.action_probabilities_per_state[state]

        if not state_actions:
            raise ValueError(f"No possible actions found for state {state}.")

        possible_actions = list(state_actions.keys())
        action_weights = list(state_actions.values())

        return choices(possible_actions, weights=action_weights, k=1)[0]

    @abstractmethod
    @validate_probabilities()
    def update_probabilities(self, values: Values[StateT, ActionT]) -> None:
        """Update probabilities for all states."""
        pass

    @abstractmethod
    def update_probabilities_for_state(
        self, state: StateT, values: Values[StateT, ActionT]
    ) -> None:
        """Update probabilities for a single state only (incremental update)."""
        pass
