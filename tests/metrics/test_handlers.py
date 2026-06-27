"""Unit tests for metric handlers."""

import pytest
import numpy as np

from my_rl_lib.metrics.context_key import ContextKey
from my_rl_lib.metrics.handlers.scalars import (
    EpisodeRewardHandler,
    EpisodeStepsHandler,
    TDErrorHandler,
    ValueChangeHandler,
    EpsilonHandler,
    ImportanceRatioHandler,
)
from my_rl_lib.metrics.handlers.media import (
    StateVisitationHeatmapHandler,
    ValueFunctionHeatmapHandler,
)


class TestScalarHandlers:
    @pytest.mark.parametrize(
        "handler, key, value",
        [
            (EpisodeRewardHandler(), ContextKey.EPISODE_REWARD, 42.0),
            (EpisodeStepsHandler(), ContextKey.EPISODE_STEPS, 100),
            (TDErrorHandler(), ContextKey.TD_ERROR, 0.5),
            (ValueChangeHandler(), ContextKey.VALUE_CHANGE, 0.01),
            (EpsilonHandler(), ContextKey.EPSILON, 0.1),
            (ImportanceRatioHandler(), ContextKey.IMPORTANCE_RATIO, 2.0),
        ],
    )
    def test_process_returns_scalar_format(self, handler, key, value):
        result = handler.process({key: value}, episode=0)
        assert result["type"] == "scalar"
        assert "tag" in result
        assert "value" in result

    @pytest.mark.parametrize(
        "handler, key",
        [
            (EpisodeRewardHandler(), ContextKey.EPISODE_REWARD),
            (EpisodeStepsHandler(), ContextKey.EPISODE_STEPS),
            (TDErrorHandler(), ContextKey.TD_ERROR),
            (ValueChangeHandler(), ContextKey.VALUE_CHANGE),
            (EpsilonHandler(), ContextKey.EPSILON),
            (ImportanceRatioHandler(), ContextKey.IMPORTANCE_RATIO),
        ],
    )
    def test_required_keys_contains_expected_key(self, handler, key):
        assert key in handler.required_keys

    @pytest.mark.parametrize(
        "handler, key, value",
        [
            (EpisodeRewardHandler(), ContextKey.EPISODE_REWARD, 7.5),
            (EpisodeStepsHandler(), ContextKey.EPISODE_STEPS, 50),
            (TDErrorHandler(), ContextKey.TD_ERROR, 0.123),
            (EpsilonHandler(), ContextKey.EPSILON, 0.05),
        ],
    )
    def test_process_value_matches_context_input(self, handler, key, value):
        result = handler.process({key: value}, episode=0)
        assert result["value"] == value

    @pytest.mark.parametrize(
        "handler",
        [
            EpisodeRewardHandler(),
            EpisodeStepsHandler(),
            TDErrorHandler(),
            EpsilonHandler(),
        ],
    )
    def test_metric_name_is_a_non_empty_string(self, handler):
        assert isinstance(handler.metric_name, str)
        assert len(handler.metric_name) > 0


class TestMediaHandlers:
    def test_state_visitation_required_key(self):
        handler = StateVisitationHeatmapHandler()
        assert ContextKey.STATE_VISITS in handler.required_keys

    def test_state_visitation_returns_image_format(self):
        handler = StateVisitationHeatmapHandler()
        context = {ContextKey.STATE_VISITS: [(0, 0), (0, 1), (1, 0), (0, 0)]}
        result = handler.process(context, episode=0)
        assert result["type"] == "image"
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)

    def test_state_visitation_heatmap_shape_matches_states(self):
        handler = StateVisitationHeatmapHandler()
        context = {ContextKey.STATE_VISITS: [(0, 0), (1, 2), (2, 3)]}
        result = handler.process(context, episode=0)
        # max_x=2+1=3, max_y=3+1=4 → heatmap shape = (4, 3)
        assert result["data"].shape == (4, 3)

    def test_state_visitation_empty_visits_returns_zeros(self):
        handler = StateVisitationHeatmapHandler()
        context = {ContextKey.STATE_VISITS: []}
        result = handler.process(context, episode=0)
        assert result["type"] == "image"

    def test_value_function_required_key(self):
        handler = ValueFunctionHeatmapHandler()
        assert ContextKey.VALUE_FUNCTION in handler.required_keys

    def test_value_function_returns_image_format(self):
        handler = ValueFunctionHeatmapHandler()
        context = {ContextKey.VALUE_FUNCTION: {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}}
        result = handler.process(context, episode=0)
        assert result["type"] == "image"
        assert isinstance(result["data"], np.ndarray)

    def test_value_function_empty_dict_returns_zeros(self):
        handler = ValueFunctionHeatmapHandler()
        context = {ContextKey.VALUE_FUNCTION: {}}
        result = handler.process(context, episode=0)
        assert result["type"] == "image"
