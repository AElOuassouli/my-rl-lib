from typing import TYPE_CHECKING, Any

from my_rl_lib.environments.abstract import Environment
from my_rl_lib.values.abstract import Values, ValuesType
from my_rl_lib.values.initializer import Initializer

if TYPE_CHECKING:
    from my_rl_lib.policies.abstract import Policy


class ActionStateValues(Values):
    # action-state values nested dictionary
    # represents Q(s, a)
    # Outer keys are states
    # Inner keys are actions, values are floats
    # Structure: {state: {action: value}}

    type: ValuesType = ValuesType.ACTION_STATE_VALUES

    def init_from_environment(
        self,
        environment: Environment,
        initializer: Initializer,
    ) -> None:
        """
        Initialize action-state values from the environment using the provided initializer.
        """
        actions_per_state = environment.get_actions_per_state()
        value_init_function = initializer.get_state_action_values_initializer_function(environment)

        self.values = {}
        for state, actions in actions_per_state.items():
            self.values[state] = {
                action: value_init_function((state, action)) for action in actions
            }

    def get_value(self, entry: Any) -> float:
        """
        Get the value for a given state-action pair.

        Args:
            entry: Tuple of (state, action).
        Returns:
            The value associated with the state-action pair.
        """
        if self.values is None:
            raise ValueError("Action-state values have not been initialized.")

        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("Entry must be a tuple of (state, action).")

        state, action = entry

        if state not in self.values:
            raise ValueError(f"State {state} not found in action-state values.")

        if action not in self.values[state]:
            raise ValueError(f"Action {action} not found for state {state}.")

        return float(self.values[state][action])

    def set_value(self, entry: Any, value: float) -> None:
        """
        Set the value for a given state-action pair.

        Args:
            entry: Tuple of (state, action).
            value: The value to set for the state-action pair.
        """
        if self.values is None:
            raise ValueError("Action-state values have not been initialized.")

        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("Entry must be a tuple of (state, action).")

        state, action = entry

        if state not in self.values:
            raise ValueError(f"State {state} not found in action-state values.")

        if action not in self.values[state]:
            raise ValueError(f"Action {action} not found for state {state}.")

        self.values[state][action] = value

    def get_max_action_and_value(self, state: Any) -> tuple[Any, float]:
        """
        Get the action with the maximum value for a given state.

        Args:
            state: The state for which to find the best action.

        Returns:
            A tuple containing the best action and its corresponding value.
        """
        if self.values is None:
            raise ValueError("Action-state values have not been initialized.")

        if state not in self.values:
            raise ValueError(f"State {state} not found in action-state values.")

        state_actions = self.values[state]

        if not state_actions:
            raise ValueError(f"No possible actions found for state {state}.")

        # Much cleaner with max() on dict items
        best_action = max(state_actions, key=state_actions.get)
        best_value = state_actions[best_action]

        return best_action, best_value

    def get_expected_value(
        self,
        state: Any,
        policy: "Policy",
    ) -> float:
        """
        Get the expected value for a given state under a specified policy.

        Args:
            state: The state for which to calculate the expected value.
            policy: The policy to use for calculating action probabilities.
        Returns:
            The expected value for the given state.
        """
        if self.values is None:
            raise ValueError("Action-state values have not been initialized.")

        if state not in self.values:
            raise ValueError(f"State {state} not found in action-state values.")

        state_actions = self.values[state]

        if not state_actions:
            raise ValueError(f"No possible actions found for state {state}.")

        expected_value = 0.0
        for action, action_value in state_actions.items():
            action_probability = policy.get_action_probability_given_state(state, action)
            expected_value += action_probability * action_value

        return expected_value

    def add(self, other: "Values") -> "Values":
        """
        Add two ActionStateValues objects together.

        Args:
            other: Another ActionStateValues object to add.

        Returns:
            A new ActionStateValues object representing the sum.
        """
        if not isinstance(other, ActionStateValues):
            raise ValueError("Can only add another ActionStateValues object.")

        if self.values is None or other.values is None:
            raise ValueError("Both ActionStateValues must be initialized before addition.")

        result = ActionStateValues()
        result.values = {}

        for state in self.values.keys():
            if state not in other.values:
                raise ValueError(f"State {state} not found in both ActionStateValues.")

            result.values[state] = {}
            for action in self.values[state].keys():
                if action not in other.values[state]:
                    raise ValueError(
                        f"Action {action} for state {state} not found in both ActionStateValues."
                    )

                summed_value = self.values[state][action] + other.values[state][action]
                result.values[state][action] = summed_value

        return result
