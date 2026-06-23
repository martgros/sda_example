"""Core state, dynamics, and result models for hierarchical production control."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class CostConfig:
    """Cost parameters used in the objective.

    Args:
        shortage_cost: End-of-day shortage penalty per unit.
        holding_cost: End-of-day inventory holding cost per unit.
        machine_cost: Machine operating cost per active machine per step.
    """

    shortage_cost: float
    holding_cost: float
    machine_cost: float = 0.0


@dataclass(frozen=True)
class ProductionConfig:
    """Production and horizon parameters for the environment.

    Args:
        total_machines: Total number of available machines.
        steps_per_day: Number of control steps in one day.
        mean_output_per_machine: Mean expected output per machine per step.
        max_output_per_machine: Maximum output cap per machine per step.
        beta_shape_a: Alpha shape for beta-distributed performance factors.
    """

    total_machines: int
    steps_per_day: int
    mean_output_per_machine: float
    max_output_per_machine: float
    beta_shape_a: float = 50.0


@dataclass(frozen=True)
class PlanningState:
    """State visible to the day-level planning policy.

    Args:
        inventory: Inventory at start of day.
        current_day_index: Zero-based day index.
        demand_today: Deterministic demand for current day.
        forecast_demand: Remaining deterministic demand forecast.
    """

    inventory: int
    current_day_index: int
    demand_today: int
    forecast_demand: Sequence[int]


@dataclass(frozen=True)
class ExecutionState:
    """State visible to the intra-day control policy.

    Args:
        inventory: Current inventory.
        remaining_demand: Remaining demand for current day.
        remaining_time_steps: Remaining control steps in current day.
        cumulative_production: Production accumulated so far in current day.
        production_target: Day-level production target.
    """

    inventory: int
    remaining_demand: int
    remaining_time_steps: int
    cumulative_production: int
    production_target: int


@dataclass(frozen=True)
class StepResult:
    """Result of one intra-day execution step.

    Args:
        active_machines: Active machines chosen by control policy.
        produced_units: Realized produced units in this step.
        shipped_units: Units shipped to demand in this step.
        ending_inventory: Inventory after production and shipment.
        remaining_demand: Remaining demand after shipment.
    """

    active_machines: int
    produced_units: int
    shipped_units: int
    ending_inventory: int
    remaining_demand: int


@dataclass(frozen=True)
class DayResult:
    """Aggregated result of one day.

    Args:
        day_index: Zero-based day index.
        production_target: Planned daily target.
        total_production: Realized total production over the day.
        end_inventory: End-of-day inventory.
        shortage: End-of-day shortage.
        total_machine_usage: Sum of active machines across steps.
        inventory_cost: End-of-day holding cost.
        shortage_cost: End-of-day shortage penalty.
        machine_cost: Accumulated machine operating cost.
        total_cost: Total daily cost.
        step_results: Step-level execution details.
    """

    day_index: int
    production_target: int
    total_production: int
    end_inventory: int
    shortage: int
    total_machine_usage: int
    inventory_cost: float
    shortage_cost: float
    machine_cost: float
    total_cost: float
    step_results: list[StepResult]


class HierarchicalProductionModel:
    """Single-step execution model with machine underperformance uncertainty.

    Encapsulates both the physical production dynamics and the cost objective,
    making the model self-contained per the Powell SDA framework.

    Args:
        production_config: Production system parameters.
        cost_config: Cost model for the objective function.
    """

    def __init__(self, production_config: ProductionConfig, cost_config: CostConfig) -> None:
        if production_config.total_machines <= 0:
            raise ValueError("total_machines must be positive")
        if production_config.steps_per_day <= 0:
            raise ValueError("steps_per_day must be positive")
        if production_config.mean_output_per_machine <= 0:
            raise ValueError("mean_output_per_machine must be positive")
        if production_config.max_output_per_machine <= 0:
            raise ValueError("max_output_per_machine must be positive")

        self.config = production_config
        self.cost_config = cost_config

    def clamp_active_machines(self, active_machines: int) -> int:
        """Clamp machine usage to feasible bounds.

        Args:
            active_machines: Requested machine count.

        Returns:
            Feasible machine count.
        """

        return max(0, min(active_machines, self.config.total_machines))

    def sample_machine_output(self, rng: random.Random) -> int:
        """Sample output of one machine for one step.

        Args:
            rng: Random generator.

        Returns:
            Integer machine output.
        """
        # control sharpness (higher = tighter peak near max)
        a = self.config.beta_shape_a   # 50
        mu = self.config.mean_output_per_machine / self.config.max_output_per_machine  # ≈ 0.923
        b = a * (1 - mu) / mu
        factor = rng.betavariate(a, b)
        return int(round(factor * self.config.max_output_per_machine))

    def sample_step_production(self, active_machines: int, rng: random.Random) -> int:
        """Sample realized step production across active machines.

        Args:
            active_machines: Requested machine count.
            rng: Random generator.

        Returns:
            Realized production units in one step.
        """

        machines = self.clamp_active_machines(active_machines)
        return sum(self.sample_machine_output(rng) for _ in range(machines))

    def execute_step(
        self,
        state: ExecutionState,
        active_machines: int,
        rng: random.Random,
    ) -> StepResult:
        """Apply one intra-day control action to the execution state.

        Args:
            state: Current execution state.
            active_machines: Machines requested by control policy.
            rng: Random generator.

        Returns:
            Step-level transition result.
        """

        machines = self.clamp_active_machines(active_machines)
        produced = self.sample_step_production(machines, rng)
        inventory_after_production = state.inventory + produced
        shipped = min(inventory_after_production, state.remaining_demand)
        remaining_demand = state.remaining_demand - shipped
        ending_inventory = inventory_after_production - shipped

        return StepResult(
            active_machines=machines,
            produced_units=produced,
            shipped_units=shipped,
            ending_inventory=ending_inventory,
            remaining_demand=remaining_demand,
        )
