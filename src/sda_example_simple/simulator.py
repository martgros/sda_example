"""Simulator that orchestrates planning and intra-day control over a horizon."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .models import (
    DayResult,
    ExecutionState,
    HierarchicalProductionModel,
    PlanningState,
    StepResult,
)
from .policies import HierarchicalPolicy


@dataclass(frozen=True)
class EpisodeResult:
    """Container for full-horizon simulation outputs.

    Args:
        day_results: List of per-day simulation results.
        total_cost: Total cost over all days.
    """

    day_results: list[DayResult]
    total_cost: float


class HierarchicalSimulator:
    """Runs hierarchical planning/control simulation over multiple days.

    Args:
        model: Production model (includes cost config).
    """

    def __init__(self, model: HierarchicalProductionModel) -> None:
        self.model = model

    def run_episode(
        self,
        initial_inventory: int,
        demand_forecast: Sequence[int],
        policy: HierarchicalPolicy,
        seed: int,
    ) -> EpisodeResult:
        """Run one horizon episode using a hierarchical policy.

        Args:
            initial_inventory: Inventory at day 0.
            demand_forecast: Deterministic demand list across horizon.
            policy: Hierarchical policy with plan and control methods.
            seed: Seed for reproducibility.

        Returns:
            Full episode result.
        """

        rng = random.Random(seed)
        inventory = initial_inventory
        day_results: list[DayResult] = []

        for day_idx, demand_today in enumerate(demand_forecast):
            planning_state = PlanningState(
                inventory=inventory,
                current_day_index=day_idx,
                demand_today=demand_today,
                forecast_demand=demand_forecast[day_idx:],
            )
            target = max(0, policy.plan_production(planning_state, self.model))

            execution_state = ExecutionState(
                inventory=inventory,
                remaining_demand=demand_today,
                remaining_time_steps=self.model.config.steps_per_day,
                cumulative_production=0,
                production_target=target,
            )
            step_results: list[StepResult] = []
            total_machine_usage = 0

            for step_idx in range(self.model.config.steps_per_day):
                active_machines = policy.control_machines(execution_state, self.model, rng)
                step = self.model.execute_step(execution_state, active_machines, rng)
                step_results.append(step)
                total_machine_usage += step.active_machines

                execution_state = ExecutionState(
                    inventory=step.ending_inventory,
                    remaining_demand=step.remaining_demand,
                    remaining_time_steps=self.model.config.steps_per_day - step_idx - 1,
                    cumulative_production=execution_state.cumulative_production + step.produced_units,
                    production_target=target,
                )

            shortage = execution_state.remaining_demand
            inventory = execution_state.inventory
            cost_cfg = self.model.cost_config
            inventory_cost = inventory * cost_cfg.holding_cost
            shortage_cost = shortage * cost_cfg.shortage_cost
            machine_cost = total_machine_usage * cost_cfg.machine_cost
            total_cost = inventory_cost + shortage_cost + machine_cost

            day_results.append(
                DayResult(
                    day_index=day_idx,
                    production_target=target,
                    total_production=execution_state.cumulative_production,
                    end_inventory=inventory,
                    shortage=shortage,
                    total_machine_usage=total_machine_usage,
                    inventory_cost=inventory_cost,
                    shortage_cost=shortage_cost,
                    machine_cost=machine_cost,
                    total_cost=total_cost,
                    step_results=step_results,
                )
            )

        return EpisodeResult(
            day_results=day_results,
            total_cost=sum(day.total_cost for day in day_results),
        )
