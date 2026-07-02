"""Shared type variables for the state/action-parametrized framework.

The library is generic over the environment's state and action types. These
type variables thread that generality through environments, values, policies,
and the learning algorithms so a concrete environment (e.g. a grid world with
``tuple[int, int]`` states) yields fully-typed algorithms.
"""

from typing import TypeVar

# State type (e.g. ``tuple[int, int]`` for a grid world).
StateT = TypeVar("StateT")

# Action type (e.g. ``tuple[int, int]`` displacement for a grid world).
ActionT = TypeVar("ActionT")
