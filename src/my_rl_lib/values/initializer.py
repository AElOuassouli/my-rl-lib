import random
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from my_rl_lib.environments.abstract import Environment


class InitializerType(str, Enum):
    UNIFORM = "uniform"
    CONSTANT = "constant"


class Initializer(BaseModel):
    initializer_type: InitializerType

    # optional parameters
    terminal_states_value: float | None
    range_uniform_non_terminal: tuple[float, float] | None = None
    constant_value_non_terminal: float | None = None

    def get_state_action_values_initializer_function(
        self,
        environment: "Environment",
    ) -> Callable[[tuple[Any, Any]], float]:
        terminal_states = environment.get_terminal_states()

        if self.initializer_type == InitializerType.UNIFORM:
            if self.range_uniform_non_terminal is None:
                raise ValueError(
                    "range_uniform_non_terminal must be provided for UNIFORM initializer."
                )

            def initializer(state_action: tuple[Any, Any]) -> float:
                if self.range_uniform_non_terminal is None:
                    raise ValueError(
                        "range_uniform_non_terminal must be provided for UNIFORM initializer."
                    )

                state, _ = state_action
                if state in terminal_states and self.terminal_states_value is not None:
                    return self.terminal_states_value
                return random.uniform(*self.range_uniform_non_terminal)

            return initializer

        elif self.initializer_type == InitializerType.CONSTANT:
            if self.constant_value_non_terminal is None:
                raise ValueError(
                    "constant_value_non_terminal must be provided for CONSTANT initializer."
                )

            def initializer(state_action: tuple[Any, Any]) -> float:
                if self.constant_value_non_terminal is None:
                    raise ValueError(
                        "constant_value_non_terminal must be provided for CONSTANT initializer."
                    )

                state, _ = state_action
                if state in terminal_states and self.terminal_states_value is not None:
                    return self.terminal_states_value
                return self.constant_value_non_terminal

            return initializer
        else:
            raise NotImplementedError(
                f"Initializer type {self.initializer_type} is not implemented."
            )
