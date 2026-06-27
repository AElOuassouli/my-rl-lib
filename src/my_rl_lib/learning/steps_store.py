from typing import Any

from pydantic import BaseModel


class LearningStep(BaseModel):
    state: Any
    action: Any
    reward: float | int | None


class EpisodeStepsCircularStore:
    def __init__(self, n: int):
        self.store: list[LearningStep | None] = [None] * (n + 1)
        self.n = n

    def set_step(self, time_step: int, step: LearningStep) -> None:
        self.store[time_step % (self.n + 1)] = step

    def get_step(self, time_step: int) -> LearningStep:
        entry = self.store[time_step % (self.n + 1)]
        if entry is None:
            raise ValueError("Store entry is None when it should not be.")
        return entry
