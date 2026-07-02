import random
import tempfile
from enum import Enum
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from pydantic import Field, PositiveInt, PrivateAttr, field_validator, model_validator

from my_rl_lib.policies.greedy import Greedy
from my_rl_lib.values.abstract import Values, ValuesType

from .abstract import Environment, EnvironmentType, RenderMode, RenderPolicyOptions, StepResult

# Grid world states and actions are (row, col) integer tuples.
GridState = tuple[int, int]
GridAction = tuple[int, int]

DEFAULT_HEIGHT = 10
DEFAULT_WIDTH = 10
DEFAULT_WIND_STRENGTH_BY_COLUMN = {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1}


class WindyGridWorlType(str, Enum):
    """Type of Windy Grid World environment."""

    SIMPLE = "simple"
    KINGS_MOVE = "kings_move"


class WindyGridWorld(Environment[GridState, GridAction]):
    """Windy Grid World environment.

    A simple grid world environment with wind effects.
    From Sutton & Barto's Reinforcement Learning book.
    """

    type: EnvironmentType = EnvironmentType.SIMULATOR
    current_timestep: int | None = None
    current_state: tuple[int, int] | None = None  # (row, column)

    # possible actions are represented as vector displacements
    windy_grid_world_type: WindyGridWorlType = WindyGridWorlType.SIMPLE
    _possible_actions: list[tuple[int, int]] = PrivateAttr()

    # Grid world parameters
    grid_height: PositiveInt = DEFAULT_HEIGHT
    grid_width: PositiveInt = DEFAULT_WIDTH

    start_cell: tuple[int, int] = (3, 0)  # (row, column)
    goal_cell: tuple[int, int] = (3, 7)  # (row, column)

    # Wind parameters
    wind_strength_by_column: dict[int, int] = DEFAULT_WIND_STRENGTH_BY_COLUMN
    wind_variance: int = 0  # No variance by default
    wind_variance_probability: float = Field(
        default=0, ge=0.0, le=1.0
    )  # Probability of wind variance occurring

    # Reward parameters
    step_reward: float = -1.0
    goal_reward: float = 0.0

    @field_validator("wind_strength_by_column")
    @classmethod
    def validate_wind_strength(cls, v: dict[int, int]) -> dict[int, int]:
        """Validate that wind strength values are non-negative."""
        for col, strength in v.items():
            if strength < 0:
                raise ValueError(
                    f"Wind strength must be non-negative, got {strength} for column {col}"
                )
        return v

    @model_validator(mode="after")
    def validate_cells_and_wind_within_grid(self) -> Self:
        """Validate that start, goal cells, and wind columns are within the grid bounds."""
        start_row, start_col = self.start_cell
        goal_row, goal_col = self.goal_cell

        if not (0 <= start_row < self.grid_height and 0 <= start_col < self.grid_width):
            raise ValueError(
                f"Start cell {self.start_cell} is outside grid bounds "
                f"(0, 0) to ({self.grid_height - 1}, {self.grid_width - 1})"
            )

        if not (0 <= goal_row < self.grid_height and 0 <= goal_col < self.grid_width):
            raise ValueError(
                f"Goal cell {self.goal_cell} is outside grid bounds "
                f"(0, 0) to ({self.grid_height - 1}, {self.grid_width - 1})"
            )

        # Validate wind columns are within grid bounds
        for col in self.wind_strength_by_column.keys():
            if not (0 <= col < self.grid_width):
                raise ValueError(
                    f"Wind column {col} is outside grid bounds (0 to {self.grid_width - 1})"
                )

        # Set possible actions based on environment type
        if self.windy_grid_world_type == WindyGridWorlType.SIMPLE:
            self._possible_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        elif self.windy_grid_world_type == WindyGridWorlType.KINGS_MOVE:
            self._possible_actions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),  # Up, Down, Left, Right
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),  # Diagonals
            ]
        else:
            raise ValueError(f"Unsupported windy_grid_world_type: {self.windy_grid_world_type}")

        return self

    ###############
    def get_states(self) -> list[tuple[int, int]]:
        states = []
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                states.append((row, col))
        return states

    def get_actions_per_state(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        actions_per_state = {}
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                state = (row, col)
                actions_per_state[state] = self._get_possible_action_from_state(state)
        return actions_per_state

    def get_terminal_states(self) -> list[tuple[int, int]]:
        return [self.goal_cell]

    ########
    def reset(self) -> None:
        self.current_state = self.start_cell
        self.current_timestep = 0

    def _get_possible_action_from_state(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        valid_actions = []
        current_row, current_col = state

        for action in self._possible_actions:
            delta_row, delta_col = action
            new_row = current_row + delta_row
            new_col = current_col + delta_col

            # Check if the action keeps the agent within bounds
            if 0 <= new_row < self.grid_height and 0 <= new_col < self.grid_width:
                valid_actions.append(action)

        return valid_actions

    def get_current_possible_actions(self) -> list[tuple[int, int]]:
        """Get the list of possible actions from the current state.

        Only returns actions that keep the agent within the grid bounds.

        Returns:
            List of valid actions as (row_delta, col_delta) tuples.
        """
        if self.current_state is None:
            raise ValueError("Current state is None. Call reset() first.")

        return self._get_possible_action_from_state(self.current_state)

    def is_current_state_terminal(self) -> bool:
        """Check if the current state is terminal.

        Returns:
            True if the current state is terminal, False otherwise.
        """
        return self.current_state == self.goal_cell

    def step(self, action: GridAction) -> StepResult[GridState]:
        """Take an action in the environment.

        Args:
            action: The action to take as (row_delta, col_delta).

        Returns:
            A StepResult containing the next state and reward.

        Raises:
            ValueError: If the action is not valid from the current state.
        """
        if self.current_state is None or self.current_timestep is None:
            raise ValueError("Current state and/or timestamp are None. Call reset() first.")

        # Check if the current state is terminal
        if self.is_current_state_terminal():
            raise ValueError(
                f"Cannot take action from terminal state {self.current_state}. "
                "Call reset() to start a new episode."
            )

        # Validate that the action is possible
        valid_actions = self.get_current_possible_actions()
        if action not in valid_actions:
            raise ValueError(
                f"Action {action} is not valid from state {self.current_state}. "
                f"Valid actions: {valid_actions}"
            )

        # Apply the action
        current_row, current_col = self.current_state
        delta_row, delta_col = action
        new_row = current_row + delta_row
        new_col = current_col + delta_col

        # Apply wind effect (wind pushes upward, which decreases row index)
        wind_strength = self.wind_strength_by_column.get(new_col, 0)

        # Apply stochastic wind variance if probability check passes
        if wind_strength != 0 and random.random() < self.wind_variance_probability:
            wind_strength += self.wind_variance

        new_row = new_row - wind_strength

        # Clamp to grid boundaries
        new_row = max(0, min(new_row, self.grid_height - 1))
        new_col = max(0, min(new_col, self.grid_width - 1))

        # Update current state
        self.current_state = (new_row, new_col)
        self.current_timestep += 1

        # Determine reward
        if self.current_state == self.goal_cell:
            reward = self.goal_reward
        else:
            reward = self.step_reward

        return StepResult[GridState](next_state=self.current_state, reward=reward)

    def render(self, mode: RenderMode = RenderMode.HUMAN) -> None:
        """Render the environment.

        Args:
            mode: The mode to render with.
        """
        if mode != RenderMode.HUMAN:
            raise NotImplementedError(f"Render mode {mode} is not implemented yet.")

        # Create figure and axis with higher DPI for better resolution
        fig, ax = plt.subplots(figsize=(self.grid_width * 0.5, self.grid_height * 0.5), dpi=150)

        # Get maximum wind strength for color normalization
        max_wind = max(self.wind_strength_by_column.values()) if self.wind_strength_by_column else 1

        # Draw grid cells
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # Determine cell color based on wind strength
                wind_strength = self.wind_strength_by_column.get(col, 0)

                # Create blue gradient based on wind strength
                # No wind = white, max wind = dark blue
                blue_intensity = wind_strength / max_wind if max_wind > 0 else 0
                cell_color = (1 - 0.7 * blue_intensity, 1 - 0.5 * blue_intensity, 1.0)

                # Special colors for start and goal cells
                if (row, col) == self.start_cell:
                    cell_color = (0.8, 1.0, 0.8)  # Light green for start
                elif (row, col) == self.goal_cell:
                    cell_color = (1.0, 0.8, 0.8)  # Light red for goal

                # Draw cell rectangle
                # Note: matplotlib uses bottom-left origin, so we flip the row
                rect = Rectangle(
                    (col, self.grid_height - row - 1),
                    1,
                    1,
                    facecolor=cell_color,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # Add wind strength text if there's wind
                if wind_strength > 0:
                    ax.text(
                        col + 0.5,
                        self.grid_height - row - 0.2,
                        f"↑{wind_strength}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="darkblue",
                    )

        # Draw the agent as a black dot if current_state is set
        if self.current_state is not None:
            agent_row, agent_col = self.current_state
            # Draw agent in the center of its cell
            circle = Circle(
                (agent_col + 0.5, self.grid_height - agent_row - 0.5),
                0.15,
                color="black",
                zorder=10,
            )
            ax.add_patch(circle)

        # Create legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=(0.8, 1.0, 0.8), edgecolor="black", label="Start"),
            Patch(facecolor=(1.0, 0.8, 0.8), edgecolor="black", label="Goal"),
            Patch(facecolor=(0.55, 0.75, 1.0), edgecolor="black", label="Wind"),
            Patch(facecolor="white", edgecolor="black", label="Normal"),
        ]
        if self.current_state is not None:
            legend_elements.insert(0, Patch(facecolor="black", label="Agent"))

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            frameon=True,
        )

        # Set axis properties
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

        # Add title
        title = "Windy Grid World"
        if self.current_state is not None and self.current_timestep is not None:
            title += f" (Step: {self.current_timestep}, State: {self.current_state})"
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def __draw_base_grid(self, ax: Any) -> None:
        """Draw the base grid with white cells and borders for start/goal.

        Args:
            ax: The matplotlib axis to draw on.
        """
        # Draw grid cells with white background
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                rect = Rectangle(
                    (col, self.grid_height - row - 1),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

        # Draw colored borders for start and goal cells on top
        start_row, start_col = self.start_cell
        goal_row, goal_col = self.goal_cell

        start_rect = Rectangle(
            (start_col, self.grid_height - start_row - 1),
            1,
            1,
            facecolor="none",
            edgecolor="green",
            linewidth=1.5,
            zorder=100,
        )
        ax.add_patch(start_rect)

        goal_rect = Rectangle(
            (goal_col, self.grid_height - goal_row - 1),
            1,
            1,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            zorder=100,
        )
        ax.add_patch(goal_rect)

    def __render_wind(self, ax: Any) -> None:
        """Draw wind strength indicators on the grid with gradient fill.

        Args:
            ax: The matplotlib axis to draw on.
        """
        # Get maximum wind strength for color normalization
        max_wind = max(self.wind_strength_by_column.values()) if self.wind_strength_by_column else 1

        for col, wind_strength in self.wind_strength_by_column.items():
            if wind_strength > 0:
                # Calculate blue gradient intensity based on wind strength
                blue_intensity = wind_strength / max_wind if max_wind > 0 else 0
                cell_color = (1 - 0.7 * blue_intensity, 1 - 0.5 * blue_intensity, 1.0)

                # Draw gradient-filled rectangles for all rows in this column
                for row in range(self.grid_height):
                    cell_x = col
                    cell_y = self.grid_height - row - 1

                    # Draw filled rectangle with gradient color
                    wind_rect = Rectangle(
                        (cell_x, cell_y),
                        1,
                        1,
                        facecolor=cell_color,
                        edgecolor="none",
                        alpha=0.6,
                        zorder=1,
                    )
                    ax.add_patch(wind_rect)

                    # Add wind strength text
                    ax.text(
                        cell_x + 0.5,
                        cell_y + 0.2,
                        f"↑{wind_strength}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="darkblue",
                        zorder=2,
                    )

    def __render_values(self, ax: Any, values: "Values[GridState, GridAction]") -> None:
        """Render policy values on the grid.

        Args:
            ax: The matplotlib axis to draw on.
            values: The values to render.
        """

        if values.type == ValuesType.STATE_VALUES:
            self.__render_state_values(ax, values)
        elif values.type == ValuesType.ACTION_STATE_VALUES:
            self.__render_action_state_values(ax, values)

    def __render_state_values(self, ax: Any, values: Values[GridState, GridAction]) -> None:
        """Render state values in the center of cells.

        Args:
            ax: The matplotlib axis to draw on.
            values: The values object to render.
        """
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                state = (row, col)
                if values.values is not None and state in values.values:
                    value = values.values[state]
                    ax.text(
                        col + 0.5,
                        self.grid_height - row - 0.5,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        fontweight="bold",
                        color="black",
                    )

    def __render_action_state_values(self, ax: Any, values: Values[GridState, GridAction]) -> None:
        """Render action-state values in wedges dividing each cell.

        Args:
            ax: The matplotlib axis to draw on.
            values: The values object to render.
        """
        from matplotlib.patches import Polygon

        if values.values is None:
            return

        # Compute global min/max for gradient scaling
        # Flatten nested dict to get all values
        all_values = []
        for state_actions in values.values.values():
            all_values.extend(state_actions.values())
        if not all_values:
            return

        global_min_value = min(all_values)
        global_max_value = max(all_values)

        # Get wedge definitions based on environment type
        action_wedges, action_text_positions, font_size = self.__get_action_wedges()

        # Draw wedges for each cell
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                state = (row, col)
                cell_x = col
                cell_y = self.grid_height - row - 1

                # Collect values for this state
                if state not in values.values:
                    continue

                state_action_values = values.values[state]
                if not state_action_values:
                    continue

                # Find the best action
                best_action = max(state_action_values.items(), key=lambda x: x[1])[0]

                for action, value in state_action_values.items():
                    # Normalize value for color gradient
                    value_range = global_max_value - global_min_value
                    if value_range > 0:
                        normalized_value = (value - global_min_value) / value_range
                    else:
                        normalized_value = 0.5

                    # Create color gradient from light to dark blue
                    color_intensity = normalized_value
                    triangle_color = (1 - 0.9 * color_intensity, 1 - 0.7 * color_intensity, 1.0)

                    # Draw wedge
                    if action in action_wedges:
                        is_best = action == best_action

                        # Translate wedge coordinates to current cell
                        wedge_coords = [(x + cell_x, y + cell_y) for x, y in action_wedges[action]]
                        wedge = Polygon(
                            wedge_coords,
                            facecolor=triangle_color,
                            edgecolor="gray",
                            linestyle="-",
                            alpha=0.6,
                            linewidth=0.5,
                        )
                        ax.add_patch(wedge)

                        # Add value text
                        if action in action_text_positions:
                            text_x, text_y = action_text_positions[action]
                            text_color = "red" if is_best else "darkblue"
                            text_weight = "bold" if is_best else "normal"
                            ax.text(
                                text_x + cell_x,
                                text_y + cell_y,
                                f"{value:.1f}",
                                ha="center",
                                va="center",
                                fontsize=font_size,
                                color=text_color,
                                fontweight=text_weight,
                            )

    def __get_action_wedges(self) -> tuple[dict[Any, Any], dict[Any, Any], int]:
        """Get wedge definitions based on environment type.

        Returns:
            Tuple of (action_wedges, action_text_positions, font_size).
            Coordinates are relative to cell at (0, 0).
        """
        center_x = 0.5
        center_y = 0.5

        if self.windy_grid_world_type == WindyGridWorlType.SIMPLE:
            # 4 triangular wedges for simple 4-directional moves
            action_wedges = {
                (-1, 0): [(center_x, center_y), (0, 1), (1, 1)],  # Up
                (1, 0): [(center_x, center_y), (0, 0), (1, 0)],  # Down
                (0, -1): [(center_x, center_y), (0, 0), (0, 1)],  # Left
                (0, 1): [(center_x, center_y), (1, 0), (1, 1)],  # Right
            }

            action_text_positions = {
                (-1, 0): (center_x, 0.75),  # Up
                (1, 0): (center_x, 0.25),  # Down
                (0, -1): (0.25, center_y),  # Left
                (0, 1): (0.75, center_y),  # Right
            }
            font_size = 5
        else:
            # 8 wedges for king's moves - divide perimeter into 8 equal parts
            top_1_4 = (0.25, 1)
            top_3_4 = (0.75, 1)
            right_1_4 = (1, 0.75)
            right_3_4 = (1, 0.25)
            bottom_3_4 = (0.75, 0)
            bottom_1_4 = (0.25, 0)
            left_3_4 = (0, 0.25)
            left_1_4 = (0, 0.75)

            action_wedges = {
                (-1, 0): [(center_x, center_y), top_1_4, top_3_4],  # Up
                (-1, 1): [(center_x, center_y), top_3_4, right_1_4],  # Up-Right
                (0, 1): [(center_x, center_y), right_1_4, right_3_4],  # Right
                (1, 1): [(center_x, center_y), right_3_4, bottom_3_4],  # Down-Right
                (1, 0): [(center_x, center_y), bottom_3_4, bottom_1_4],  # Down
                (1, -1): [(center_x, center_y), bottom_1_4, left_3_4],  # Down-Left
                (0, -1): [(center_x, center_y), left_3_4, left_1_4],  # Left
                (-1, -1): [(center_x, center_y), left_1_4, top_1_4],  # Up-Left
            }

            action_text_positions = {
                (-1, 0): (center_x, 0.8),  # Up
                (-1, 1): (0.75, 0.75),  # Up-Right
                (0, 1): (0.8, center_y),  # Right
                (1, 1): (0.75, 0.25),  # Down-Right
                (1, 0): (center_x, 0.2),  # Down
                (1, -1): (0.25, 0.25),  # Down-Left
                (0, -1): (0.2, center_y),  # Left
                (-1, -1): (0.25, 0.75),  # Up-Left
            }
            font_size = 4

        return action_wedges, action_text_positions, font_size

    def __render_greedy_agent(
        self, ax: Any, values: Values[GridState, GridAction], max_steps: int = 1000
    ) -> tuple[int, float]:
        """Render greedy agent trajectory by simulating an episode.

        Args:
            ax: The matplotlib axis to draw on.
            values: The values object to use for creating a greedy policy.
            max_steps: Cap on the simulated greedy rollout length.

        Returns:
            Tuple of (number of steps, total reward).
        """
        # Create a greedy policy from the values
        greedy_policy: Greedy[GridState, GridAction] = Greedy()
        greedy_policy.init_from_environment_and_values(self, values)

        # Simulate a greedy episode
        history = self.simulate_greedy_agent(greedy_policy, max_steps=max_steps)

        # Calculate total reward
        total_reward = sum(reward for _, _, reward in history)
        num_steps = len(history) - 1  # Subtract initial state

        # Draw arrows for each step
        for i in range(len(history) - 1):
            current_state, action, _ = history[i + 1]
            previous_state = history[i][0]

            if action is not None and previous_state is not None and current_state is not None:
                # Convert state to grid coordinates
                prev_row, prev_col = previous_state
                curr_row, curr_col = current_state

                # Calculate positions in matplotlib coordinates
                start_x = prev_col + 0.5
                start_y = self.grid_height - prev_row - 0.5
                end_x = curr_col + 0.5
                end_y = self.grid_height - curr_row - 0.5

                # Calculate state-to-state displacement (red arrow)
                dx_result = end_x - start_x
                dy_result = end_y - start_y

                # Calculate action displacement (green arrow)
                action_delta_row, action_delta_col = action
                # In matplotlib coordinates: y increases upward, so negate row delta
                dx_action = action_delta_col * 0.3  # Scale down for visibility
                dy_action = -action_delta_row * 0.3

                # Draw state-to-state arrow (red) - shows actual movement
                ax.arrow(
                    start_x,
                    start_y,
                    dx_result,
                    dy_result,
                    head_width=0.15,
                    head_length=0.1,
                    fc="red",
                    ec="red",
                    linewidth=2,
                    length_includes_head=True,
                    zorder=10,
                    alpha=0.7,
                )

                # Draw action arrow (green) - shows intended action
                ax.arrow(
                    start_x,
                    start_y,
                    dx_action,
                    dy_action,
                    head_width=0.12,
                    head_length=0.08,
                    fc="green",
                    ec="green",
                    linewidth=1.5,
                    length_includes_head=True,
                    zorder=11,
                    alpha=0.8,
                )

        return num_steps, total_reward

    def __setup_axis(self, ax: Any, title: str, show_trajectory_legend: bool = False) -> None:
        """Setup axis properties and add legend.

        Args:
            ax: The matplotlib axis to setup.
            title: The title for the subplot.
            show_trajectory_legend: Whether to show trajectory arrow legend.
        """
        from matplotlib.patches import FancyArrow, Patch

        legend_elements = [
            Patch(facecolor="white", edgecolor="green", linewidth=1.5, label="Start"),
            Patch(facecolor="white", edgecolor="red", linewidth=1.5, label="Goal"),
        ]

        if show_trajectory_legend:
            legend_elements.extend(
                [
                    FancyArrow(
                        0,
                        0,
                        0.1,
                        0.1,
                        width=0.02,
                        head_width=0.05,
                        head_length=0.03,
                        fc="green",
                        ec="green",
                        label="Action",
                    ),
                    FancyArrow(
                        0,
                        0,
                        0.1,
                        0.1,
                        width=0.03,
                        head_width=0.06,
                        head_length=0.04,
                        fc="red",
                        ec="red",
                        label="Actual Movement",
                    ),
                ]
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            frameon=True,
        )

        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    def _build_policy_figure(
        self,
        values: Values[GridState, GridAction],
        options: list[RenderPolicyOptions],
        max_steps: int = 1000,
    ) -> Any:
        """Build (but do not save/show) the policy figure for the given options.

        Returns:
            The matplotlib Figure. Caller is responsible for saving/showing/closing.
        """
        # Create subplots - one per option
        num_plots = len(options)
        fig, axes = plt.subplots(
            1,
            num_plots,
            figsize=(self.grid_width * 0.8 * num_plots, self.grid_height * 0.8),
            dpi=150,
        )

        # Handle single subplot case
        if num_plots == 1:
            axes = [axes]

        # Render each option in its own subplot
        for idx, option in enumerate(options):
            ax = axes[idx]

            # Draw base grid for all options
            self.__draw_base_grid(ax)

            # Render based on option type
            if option == RenderPolicyOptions.VALUES:
                self.__render_values(ax, values)

                # Set title based on value type
                if values.type == ValuesType.STATE_VALUES:
                    title = "State Values"
                elif values.type == ValuesType.ACTION_STATE_VALUES:
                    title = "Action-State Values"
                else:
                    title = "Values"

                self.__setup_axis(ax, title)

            elif option == RenderPolicyOptions.GREEDY_AGENT:
                self.__render_wind(ax)
                num_steps, total_reward = self.__render_greedy_agent(ax, values, max_steps)
                title = f"Greedy Agent (Steps: {num_steps}, Reward: {total_reward:.1f})"
                self.__setup_axis(ax, title, show_trajectory_legend=True)

            else:
                title = str(option)
                self.__setup_axis(ax, title)

        plt.tight_layout()
        return fig

    def render_policy(
        self,
        values: Values[GridState, GridAction],
        options: list[RenderPolicyOptions],
        file_path: str | None = None,
        log_to_mlflow: bool = False,
        max_steps: int = 1000,
    ) -> None:
        """Render the given values in the environment.

        Args:
            values: The values to render.
            options: List of rendering options (VALUES, GREEDY_ACTIONS).
            file_path: Optional path to save the figure.
            max_steps: Cap on the greedy-trajectory rollout length.
        """
        fig = self._build_policy_figure(values, options, max_steps=max_steps)

        # Save or show the figure
        if file_path is not None:
            fig.savefig(file_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            if log_to_mlflow:
                import mlflow

                mlflow.log_artifact(file_path, artifact_path="plots")
        elif log_to_mlflow:
            import os

            import mlflow

            with tempfile.NamedTemporaryFile(
                suffix=".png", prefix="policy_plot_", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
            fig.savefig(tmp_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            try:
                mlflow.log_artifact(tmp_path, artifact_path="plots")
            finally:
                os.remove(tmp_path)
        else:
            plt.show()

    def render_policy_array(
        self,
        values: Values[GridState, GridAction],
        options: list[RenderPolicyOptions] | None = None,
        max_steps: int = 1000,
    ) -> np.ndarray:
        """Render the policy figure to an RGB numpy array (no file I/O).

        Uses the Agg canvas directly so it is independent of the active
        matplotlib backend — safe to call from a worker process.

        Args:
            values: The values to render.
            options: Rendering options; defaults to values + greedy trajectory.
            max_steps: Cap on the greedy-trajectory rollout length.

        Returns:
            A ``(H, W, 3)`` uint8 RGB image array.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if options is None:
            options = [RenderPolicyOptions.VALUES, RenderPolicyOptions.GREEDY_AGENT]

        fig = self._build_policy_figure(values, options, max_steps=max_steps)
        canvas: Any = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image

    def _build_greedy_episode_frames(
        self,
        values: Values[GridState, GridAction],
        number_episodes: int,
        max_steps: int = 1000,
    ) -> tuple[list[Any], list[Any]]:
        """Simulate greedy episodes and flatten them into per-step frame specs.

        Returns:
            Tuple ``(all_episodes, frames)`` where each frame is
            ``(episode_idx, step_idx, state, next_action, reward, is_terminal)``.
        """
        greedy_policy: Greedy[GridState, GridAction] = Greedy()
        greedy_policy.init_from_environment_and_values(self, values)

        all_episodes = []
        for _ in range(number_episodes):
            history = self.simulate_greedy_agent(greedy_policy, max_steps=max_steps)
            all_episodes.append(history)

        frames = []
        for episode_idx, history in enumerate(all_episodes):
            for step_idx, (state, action, reward) in enumerate(history):
                is_terminal = state == self.goal_cell
                # Get the next action (action at t+1) if available
                next_action = None
                if step_idx + 1 < len(history):
                    next_action = history[step_idx + 1][1]
                frames.append((episode_idx, step_idx, state, next_action, reward, is_terminal))

        return all_episodes, frames

    def _draw_animation_frame(self, ax: Any, frame: Any, all_episodes: list[Any]) -> None:
        """Draw a single animation frame onto the given axis."""
        ax.clear()

        episode_num, step_num, state, action, reward, is_terminal = frame

        # Draw base grid
        self.__draw_base_grid(ax)

        # Draw wind indicators
        self.__render_wind(ax)

        # Draw the agent
        if state is not None:
            agent_row, agent_col = state
            circle = Circle(
                (agent_col + 0.5, self.grid_height - agent_row - 0.5),
                0.2,
                color="black",
                zorder=10,
            )
            ax.add_patch(circle)

        # Draw action arrow (red) if action is available
        if action is not None and state is not None:
            agent_row, agent_col = state
            action_delta_row, action_delta_col = action

            # Calculate arrow position in matplotlib coordinates
            start_x = agent_col + 0.5
            start_y = self.grid_height - agent_row - 0.5

            # Scale arrow to be visible but not too large
            dx = action_delta_col * 0.35
            dy = -action_delta_row * 0.35  # Negate because matplotlib y increases upward

            # Draw action arrow
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                head_width=0.15,
                head_length=0.1,
                fc="red",
                ec="red",
                linewidth=2,
                length_includes_head=True,
                zorder=11,
                alpha=0.8,
            )

        # Setup axis
        total_episodes = len(all_episodes)
        total_steps = len(all_episodes[episode_num]) - 1  # Subtract initial state
        title = f"Episode {episode_num + 1}/{total_episodes} | Step {step_num}/{total_steps}"
        if is_terminal:
            title += " | GOAL REACHED"

        self.__setup_axis(ax, title, show_trajectory_legend=False)

    def generate_agent_animation_greedy_policy_from_values(
        self,
        values: Values[GridState, GridAction],
        number_episodes: int,
        fps: int = 10,
        file_path: str | None = None,
        log_to_mlflow: bool = False,
        max_steps: int = 1000,
    ) -> None:
        """Generate an animation of a greedy agent acting in the environment.

        All episodes are rendered sequentially in a single video file.

        Args:
            values: The values to use for creating a greedy policy.
            number_episodes: Number of episodes to generate and animate.
            fps: Frames per second for the animation.
            file_path: The path to save the animation file. If None, defaults to 'agent_animation.mp4'.
            max_steps: Cap on the greedy rollout length per episode.
        """
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        if file_path is None:
            file_path = "agent_animation.mp4"

        all_episodes, frames = self._build_greedy_episode_frames(
            values, number_episodes, max_steps=max_steps
        )

        # Create figure and axis for animation
        fig, ax = plt.subplots(figsize=(self.grid_width * 0.8, self.grid_height * 0.8), dpi=150)

        def init() -> list[Any]:
            """Initialize animation."""
            ax.clear()
            return []

        def animate(frame_idx: int) -> list[Any]:
            """Animate a single frame."""
            self._draw_animation_frame(ax, frames[frame_idx], all_episodes)
            return []

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frames),
            interval=1000 / fps,
            blit=True,
        )

        # Save animation
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(file_path, writer=writer)
        plt.close(fig)

        print(f"Animation saved to {file_path}")

        if log_to_mlflow:
            import mlflow

            mlflow.log_artifact(file_path, artifact_path="animations")

    def render_greedy_agent_frames(
        self,
        values: Values[GridState, GridAction],
        number_episodes: int = 3,
        max_steps: int = 1000,
    ) -> list[np.ndarray]:
        """Render greedy-agent episodes into a list of RGB numpy frames.

        Uses the Agg canvas directly (backend-independent), so it is safe to
        call from a worker process. Frames share a fixed size (constant figsize
        and dpi), suitable for encoding to a GIF/video.

        Args:
            values: The values to use for creating a greedy policy.
            number_episodes: Number of episodes to simulate and animate.
            max_steps: Cap on the greedy rollout length per episode.

        Returns:
            List of ``(H, W, 3)`` uint8 RGB frame arrays.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        all_episodes, frames = self._build_greedy_episode_frames(
            values, number_episodes, max_steps=max_steps
        )

        fig, ax = plt.subplots(figsize=(self.grid_width * 0.8, self.grid_height * 0.8), dpi=150)
        canvas: Any = FigureCanvasAgg(fig)

        rendered_frames: list[np.ndarray] = []
        for frame in frames:
            self._draw_animation_frame(ax, frame, all_episodes)
            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
            rendered_frames.append(image)

        plt.close(fig)
        return rendered_frames
