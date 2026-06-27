from typing import Any

from my_rl_lib.policies.abstract import Policy
from my_rl_lib.values.abstract import Values


class Greedy(Policy):
    """Greedy policy for reinforcement learning."""

    def update_probabilities(self, values: Values) -> None:
        """Update action probabilities for all states using greedy strategy."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        for state in self.action_probabilities_per_state.keys():
            self.update_probabilities_for_state(state, values)

    def update_probabilities_for_state(self, state: Any, values: Values) -> None:
        """Update action probabilities for a single state using greedy strategy."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        if state not in self.action_probabilities_per_state:
            raise ValueError(f"State {state} not found in action probabilities.")

        max_action, _ = values.get_max_action_and_value(state)
        state_actions = self.action_probabilities_per_state[state]

        for action in state_actions:
            if action == max_action:
                state_actions[action] = 1.0
            else:
                state_actions[action] = 0.0
