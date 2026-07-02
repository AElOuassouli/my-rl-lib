from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel

from my_rl_lib.types import ActionT, StateT
from my_rl_lib.values.abstract import Values

if TYPE_CHECKING:
    from my_rl_lib.policies.greedy import Greedy


class RenderMode(str, Enum):
    HUMAN = "human"


class EnvironmentType(str, Enum):
    MODEL = "model"
    SIMULATOR = "simulator"


class RenderPolicyOptions(str, Enum):
    VALUES = "values"
    GREEDY_AGENT = "greedy_agent"


class StepResult(BaseModel, Generic[StateT]):
    next_state: StateT
    reward: float


class Environment(BaseModel, ABC, Generic[StateT, ActionT]):
    """Abstract base class for reinforcement learning environments."""

    type: EnvironmentType
    current_timestep: int | None = None
    current_state: StateT | None = None

    #######
    # Environment Info getters
    @abstractmethod
    def get_states(self) -> list[StateT]:
        """Get the list of all possible states in the environment.

        Returns:
            List of all possible states.
        """
        pass

    @abstractmethod
    def get_actions_per_state(
        self,
    ) -> dict[StateT, list[ActionT]]:  # key : state, value : list of possible actions
        """Get the mapping of states to their possible actions.

        Returns:
            A dictionary mapping each state to a list of possible actions.
        """
        pass

    @abstractmethod
    def get_terminal_states(self) -> list[StateT]:
        """Get the list of terminal states in the environment.

        Returns:
            List of terminal states.
        """
        pass

    #######
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to an initial state and return the initial observation."""
        pass

    @abstractmethod
    def get_current_possible_actions(self) -> list[ActionT]:
        """Get the list of possible actions in the environment at the current state."""
        pass

    @abstractmethod
    def is_current_state_terminal(self) -> bool:
        """Check if the current state is terminal.

        Returns:
            True if the current state is terminal, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, action: ActionT) -> StepResult[StateT]:
        """Take an action in the environment.

        Args:
            action: The action to take.

        Returns:
            A StepResult containing the next state and reward.
        """
        pass

    @abstractmethod
    def render(self, mode: RenderMode = RenderMode.HUMAN) -> None:
        """Render the environment.

        Args:
            mode: The mode to render with.
        """
        pass

    @abstractmethod
    def render_policy(
        self,
        values: Values[StateT, ActionT],
        options: list[RenderPolicyOptions],
        file_path: str | None = None,
    ) -> None:
        """Render the given values in the environment.

        Args:
            values: The values to render.
            options: List of rendering options.
            file_path: Optional path to save the figure.
        """
        pass

    @abstractmethod
    def generate_agent_animation_greedy_policy_from_values(
        self,
        values: Values[StateT, ActionT],
        number_episodes: int,
        fps: int = 10,
        file_path: str | None = None,
    ) -> None:
        """Generate an animation of a greedy agent acting in the environment based on the given values.

        Args:
            values: The values to base the greedy policy on.
            number_episodes: Number of episodes to simulate.
            fps: Frames per second for the animation.
            file_path: Optional path to save the animation.
        """
        pass

    def simulate_greedy_agent(
        self, policy: "Greedy[StateT, ActionT]", max_steps: int = 1000
    ) -> list[tuple[StateT | None, ActionT | None, float]]:
        """Simulate an episode in the environment using the given policy.

        Args:
            policy: The policy to use for action selection.
            max_steps: Maximum number of steps to simulate.
        """

        # store (state, action, reward) tuples. The index represents the timestep.
        history: list[tuple[StateT | None, ActionT | None, float]] = []

        self.reset()

        history.append((self.current_state, None, 0.0))  # initial state, no action, no reward

        for t in range(max_steps):
            state = self.current_state
            if state is None:
                raise ValueError("Environment state is None after reset/step.")
            action = policy.select_action(state)
            step_result = self.step(action)
            reward = step_result.reward

            history.append((step_result.next_state, action, reward))

            if self.is_current_state_terminal():
                break

        return history
