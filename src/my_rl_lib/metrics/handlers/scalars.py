"""Metric handlers for scalar metrics."""

from typing import Any, Set

from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers.base import MetricHandler


class EpisodeRewardHandler(MetricHandler):
    """Handler for episode reward metric."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.EPISODE_REWARD}

    @property
    def metric_name(self) -> str:
        return "episode_reward"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "episode_reward",
            "value": context_data[ContextKey.EPISODE_REWARD],
        }


class EpisodeStepsHandler(MetricHandler):
    """Handler for episode steps metric."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.EPISODE_STEPS}

    @property
    def metric_name(self) -> str:
        return "episode_steps"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "episode_steps",
            "value": context_data[ContextKey.EPISODE_STEPS],
        }


class TDErrorHandler(MetricHandler):
    """Handler for TD error metric."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.TD_ERROR}

    @property
    def metric_name(self) -> str:
        return "td_error"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {"type": "scalar", "tag": "td_error", "value": context_data[ContextKey.TD_ERROR]}


class ValueChangeHandler(MetricHandler):
    """Handler for value change metric."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.VALUE_CHANGE}

    @property
    def metric_name(self) -> str:
        return "value_change"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "value_change",
            "value": context_data[ContextKey.VALUE_CHANGE],
        }


class EpsilonHandler(MetricHandler):
    """Handler for epsilon (exploration rate) metric."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.EPSILON}

    @property
    def metric_name(self) -> str:
        return "epsilon"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {"type": "scalar", "tag": "epsilon", "value": context_data[ContextKey.EPSILON]}


class ImportanceRatioHandler(MetricHandler):
    """Handler for importance sampling ratio metric (off-policy)."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.IMPORTANCE_RATIO}

    @property
    def metric_name(self) -> str:
        return "importance_ratio"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "importance_ratio",
            "value": context_data[ContextKey.IMPORTANCE_RATIO],
        }


class TrainingTimeHandler(MetricHandler):
    """Handler for total training time (computed by the collector at close)."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.TRAINING_TIME}

    @property
    def metric_name(self) -> str:
        return "training_time"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "training_time",
            "value": context_data[ContextKey.TRAINING_TIME],
        }


class EpisodesPerSecondHandler(MetricHandler):
    """Handler for training throughput (computed by the collector at close)."""

    @property
    def required_keys(self) -> Set[ContextKey]:
        return {ContextKey.EPISODES_PER_SECOND}

    @property
    def metric_name(self) -> str:
        return "episodes_per_second"

    def process(self, context_data: dict[ContextKey, Any], episode: int) -> dict[str, Any]:
        return {
            "type": "scalar",
            "tag": "episodes_per_second",
            "value": context_data[ContextKey.EPISODES_PER_SECOND],
        }
