from pydantic import BaseModel
from my_rl_lib.values import ActionStateValues
from my_rl_lib.types import StateT, ActionT
from my_rl_lib.values.initializer import Initializer
from my_rl_lib.policies.epsilon_greedy import EpsilonGreedy

from typing import Generic, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from my_rl_lib.environments.abstract import Environment


class PlanningModel(BaseModel, Generic[StateT, ActionT]):
    # describes a model of an environment. Provides for each state-action pair, the reward and
    # the next state.
    model: dict[tuple[StateT, ActionT], tuple[StateT, float]]

    def __init__(self) -> None:
        self.model: dict[tuple[StateT, ActionT], tuple[StateT, float]] = {}

    def get_random_previously_taken_state_action_pair(self) -> tuple[StateT, ActionT] | None:
        return random.choice(list(self.model.keys())) if len(self.model) > 0 else None

    def get_reward_and_next_state(self, state: StateT, action: ActionT) -> tuple[StateT, float]:
        if (state, action) not in self.model:
            raise ValueError(f"State-Action pair {(state, action)} does not exist in this model")

        return self.model[(state, action)]

    def update_model(
        self, state: StateT, action: ActionT, reward: float, next_state: StateT
    ) -> None:
        self.model[(state, action)] = (next_state, reward)


class DynaQ(BaseModel, Generic[StateT, ActionT]):
    def __init__(
        self,
        environment: Environment[StateT, ActionT],
        initializer: Initializer,
        epsilon: float,
        gamma: float,
        alpha: float,  # learning rate
        n: int,  # planning depth
    ):
        self.environment = environment

        self.values: ActionStateValues[StateT, ActionT] = ActionStateValues()
        self.values.init_from_environment(environment, initializer)

        self.planning_model: PlanningModel[StateT, ActionT] = PlanningModel()

        assert epsilon > 0 and epsilon < 1

        self.epsilon: float = epsilon
        self.epsilon_greedy_policy: EpsilonGreedy[StateT, ActionT] = EpsilonGreedy(epsilon=epsilon)
        self.epsilon_greedy_policy.init_from_environment_and_values(self.environment, self.values)

        self.gamma: float = gamma

        self.alpha: float = alpha

        self.n: int = n

    def run(self, run_time: int = 1000) -> None:
        iterations = 0

        while iterations < run_time:
            # ensures the environment is not at a terminal state.
            if self.environment.is_current_state_terminal():
                self.environment.reset()
            state_t = self.environment.current_state

            assert state_t is not None

            # selects an action according to an epsilon-greedy policy
            action_t = self.epsilon_greedy_policy.select_action(state_t)

            step_result = self.environment.step(action_t)

            reward_t = step_result.reward
            state_t1 = step_result.next_state

            # updates the value function.
            _, max_value = self.values.get_max_action_and_value(state_t1)
            update = reward_t + self.gamma * max_value - self.values.get_value((state_t, action_t))
            new_value = self.values.get_value((state_t, action_t)) + self.alpha * update
            self.values.set_value((state_t, action_t), new_value)

            # update the epsilon greedy policy according to the value update
            self.epsilon_greedy_policy.update_probabilities_for_state(state_t, self.values)

            # updates the model
            self.planning_model.update_model(state_t, action_t, reward_t, state_t1)

            for i in range(self.n):
                pair = self.planning_model.get_random_previously_taken_state_action_pair()
                if pair is None:
                    break  # no experience to draw from.

                state_t, action_t = pair

                state_t1, reward_t = self.planning_model.get_reward_and_next_state(
                    state_t, action_t
                )

                # updates the value function
                _, max_value_p = self.values.get_max_action_and_value(state_t1)
                update_p = (
                    reward_t + self.gamma * max_value_p - self.values.get_value((state_t, action_t))
                )
                new_value_p = self.values.get_value((state_t, action_t)) + self.alpha * update_p
                self.values.set_value((state_t, action_t), new_value_p)

                self.epsilon_greedy_policy.update_probabilities_for_state(state_t, self.values)
