"""Unit tests for WindyGridWorld environment."""

import pytest

from my_rl_lib.environments.windy_grid_world import WindyGridWorld, WindyGridWorlType


def _make_grid(**kwargs) -> WindyGridWorld:
    defaults = dict(
        grid_height=5,
        grid_width=5,
        start_cell=(0, 0),
        goal_cell=(4, 4),
        wind_strength_by_column={},
    )
    defaults.update(kwargs)
    return WindyGridWorld(**defaults)


class TestGetStates:
    def test_state_count_matches_grid_size(self):
        env = _make_grid(grid_height=4, grid_width=6, goal_cell=(3, 5))
        assert len(env.get_states()) == 24

    def test_contains_start_and_goal(self):
        env = _make_grid()
        states = env.get_states()
        assert (0, 0) in states
        assert (4, 4) in states

    def test_all_cells_present(self):
        env = _make_grid(grid_height=3, grid_width=3, goal_cell=(2, 2))
        states = set(env.get_states())
        for r in range(3):
            for c in range(3):
                assert (r, c) in states


class TestGetTerminalStates:
    def test_contains_goal_cell(self):
        env = _make_grid(goal_cell=(3, 3))
        assert (3, 3) in env.get_terminal_states()

    def test_only_goal_is_terminal(self):
        env = _make_grid(goal_cell=(2, 2))
        assert env.get_terminal_states() == [(2, 2)]


class TestReset:
    def test_sets_current_state_to_start(self):
        env = _make_grid(start_cell=(1, 2))
        env.reset()
        assert env.current_state == (1, 2)

    def test_reset_returns_none(self):
        env = _make_grid()
        result = env.reset()
        assert result is None

    def test_reset_sets_timestep_to_zero(self):
        env = _make_grid()
        env.reset()
        assert env.current_timestep == 0


class TestIsCurrentStateTerminal:
    def test_not_terminal_at_start(self):
        env = _make_grid(start_cell=(0, 0), goal_cell=(4, 4))
        env.reset()
        assert env.is_current_state_terminal() is False

    def test_terminal_at_goal(self):
        env = _make_grid(goal_cell=(4, 4))
        env.reset()
        env.current_state = (4, 4)
        assert env.is_current_state_terminal() is True

    def test_not_terminal_at_arbitrary_non_goal_cell(self):
        env = _make_grid(goal_cell=(4, 4))
        env.reset()
        env.current_state = (2, 3)
        assert env.is_current_state_terminal() is False


class TestStep:
    def test_move_right_no_wind(self):
        env = _make_grid(start_cell=(2, 0))
        env.reset()
        result = env.step((0, 1))  # right
        assert result.next_state == (2, 1)
        assert result.reward == pytest.approx(-1.0)

    def test_move_down_no_wind(self):
        env = _make_grid(start_cell=(0, 2))
        env.reset()
        result = env.step((1, 0))  # down
        assert result.next_state == (1, 2)

    def test_move_up_no_wind(self):
        env = _make_grid(start_cell=(3, 2))
        env.reset()
        result = env.step((-1, 0))  # up
        assert result.next_state == (2, 2)

    def test_wind_pushes_agent_upward(self):
        # Wind of 1 in column 1: moving right into column 1 pushes row down by 1 (row - 1)
        env = _make_grid(start_cell=(2, 0), wind_strength_by_column={1: 1})
        env.reset()
        result = env.step((0, 1))  # move right into column 1
        # row = 2 + 0 (delta) - 1 (wind) = 1, col = 0 + 1 = 1
        assert result.next_state == (1, 1)

    def test_wind_does_not_push_outside_grid(self):
        # Agent at row 0, column 1 has wind 3 → would go to row -3, clamped to 0
        env = _make_grid(start_cell=(0, 0), wind_strength_by_column={1: 3})
        env.reset()
        result = env.step((0, 1))  # move right into column 1
        assert result.next_state[0] >= 0

    def test_boundary_clamped_by_wind(self):
        # Wind of 3 in column 1: moving right from (1,0) → row 1+0-3 = -2, clamped to 0
        env = _make_grid(start_cell=(1, 0), wind_strength_by_column={1: 3})
        env.reset()
        result = env.step((0, 1))  # right into column 1
        assert result.next_state[0] == 0

    def test_goal_reward_on_reaching_goal(self):
        env = _make_grid(
            start_cell=(4, 3),
            goal_cell=(4, 4),
            goal_reward=10.0,
        )
        env.reset()
        result = env.step((0, 1))
        assert result.reward == pytest.approx(10.0)
        assert result.next_state == (4, 4)

    def test_step_updates_current_state(self):
        env = _make_grid(start_cell=(2, 0))
        env.reset()
        env.step((0, 1))
        assert env.current_state == (2, 1)

    def test_step_before_reset_raises(self):
        env = _make_grid()
        with pytest.raises(ValueError):
            env.step((0, 1))

    def test_invalid_action_raises(self):
        # Action that would go out of bounds is not valid
        env = _make_grid(start_cell=(0, 0))
        env.reset()
        with pytest.raises(ValueError, match="not valid"):
            env.step((-1, 0))  # up from top-left — out of bounds


class TestActionModes:
    def test_simple_mode_center_cell_has_4_actions(self):
        env = WindyGridWorld(
            grid_height=5,
            grid_width=5,
            start_cell=(2, 2),
            goal_cell=(4, 4),
            wind_strength_by_column={},
            windy_grid_world_type=WindyGridWorlType.SIMPLE,
        )
        env.reset()
        actions = env.get_current_possible_actions()
        assert len(actions) == 4

    def test_kings_move_center_cell_has_8_actions(self):
        env = WindyGridWorld(
            grid_height=5,
            grid_width=5,
            start_cell=(2, 2),
            goal_cell=(4, 4),
            wind_strength_by_column={},
            windy_grid_world_type=WindyGridWorlType.KINGS_MOVE,
        )
        env.reset()
        actions = env.get_current_possible_actions()
        assert len(actions) == 8

    def test_corner_cell_has_fewer_actions(self):
        # Top-left corner has only 2 valid actions (right, down) in SIMPLE mode
        env = WindyGridWorld(
            grid_height=5,
            grid_width=5,
            start_cell=(0, 0),
            goal_cell=(4, 4),
            wind_strength_by_column={},
            windy_grid_world_type=WindyGridWorlType.SIMPLE,
        )
        env.reset()
        actions = env.get_current_possible_actions()
        assert len(actions) == 2


class TestValidation:
    def test_out_of_bounds_start_raises(self):
        with pytest.raises(ValueError):
            _make_grid(start_cell=(10, 0))

    def test_out_of_bounds_goal_raises(self):
        with pytest.raises(ValueError):
            _make_grid(goal_cell=(0, 99))

    def test_negative_wind_strength_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_grid(wind_strength_by_column={1: -1})

    def test_wind_column_outside_grid_raises(self):
        with pytest.raises(ValueError):
            _make_grid(wind_strength_by_column={99: 1})
