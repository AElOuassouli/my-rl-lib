from typing import Generic

from pydantic import BaseModel

from my_rl_lib.types import ActionT, StateT


class LearningStep(BaseModel, Generic[StateT, ActionT]):
    state: StateT
    action: ActionT | None
    reward: float | int | None
    rho: float | None = None
    sigma: int | None = None


class EpisodeStepsCircularStore(Generic[StateT, ActionT]):
    def __init__(self, n: int):
        self.store: list[LearningStep[StateT, ActionT] | None] = [None] * (n + 1)
        self.n = n

    def set_step(self, time_step: int, step: LearningStep[StateT, ActionT]) -> None:
        self.store[time_step % (self.n + 1)] = step

    def set_step_rho(self, time_step: int, rho: float) -> None:
        entry = self.store[time_step % (self.n + 1)]
        if entry is None:
            raise ValueError("Store entry is None when it should not be")

        self.store[time_step % (self.n + 1)].rho = rho  # type: ignore

    def set_step_sigma(self, time_step: int, sigma: int) -> None:
        entry = self.store[time_step % (self.n + 1)]
        if entry is None:
            raise ValueError("Store entry is None when it should not be")

        self.store[time_step % (self.n + 1)].sigma = sigma  # type: ignore

    def get_step(self, time_step: int) -> LearningStep[StateT, ActionT]:
        entry = self.store[time_step % (self.n + 1)]
        if entry is None:
            raise ValueError("Store entry is None when it should not be.")
        return entry
