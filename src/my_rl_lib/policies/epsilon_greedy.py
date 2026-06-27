from typing import Any

from my_rl_lib.policies.abstract import Policy
from my_rl_lib.values.abstract import Values


class EpsilonGreedy(Policy):
    """Epsilon-greedy policy for reinforcement learning."""

    # specific to epsilon-greedy
    epsilon: float = 0.1

    def init_from_environment_and_values(
        self,
        environment: Any,
        values: Values,
    ) -> None:
        """Initialize the epsilon-greedy policy from the environment and values."""
        super().init_from_environment_and_values(environment, values)

    def update_probabilities(self, values: Values) -> None:
        """Update action probabilities for all states using epsilon-greedy strategy."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        for state in self.action_probabilities_per_state.keys():
            self.update_probabilities_for_state(state, values)

    def update_probabilities_for_state(self, state: Any, values: Values) -> None:
        """Update action probabilities for a single state using epsilon-greedy strategy."""
        if self.action_probabilities_per_state is None:
            raise ValueError("Action probabilities have not been initialized.")

        if state not in self.action_probabilities_per_state:
            raise ValueError(f"State {state} not found in action probabilities.")

        max_action, _ = values.get_max_action_and_value(state)
        state_actions = self.action_probabilities_per_state[state]
        num_actions = len(state_actions)

        for action in state_actions:
            if action == max_action:
                state_actions[action] = 1 - self.epsilon + (self.epsilon / num_actions)
            else:
                state_actions[action] = self.epsilon / num_actions
