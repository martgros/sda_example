"""Planning and control policies for hierarchical SDA production control."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from .models import ExecutionState, HierarchicalProductionModel, PlanningState


class HierarchicalPolicy(ABC):
    """Interface for hierarchical planning and control policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a display name."""

    @abstractmethod
    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        """Return daily production target."""

    @abstractmethod
    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        """Return active machines for the current execution step."""


class PFAPolicy(HierarchicalPolicy):
    """Rule-based PFA policy using safety stock buffer.

    Planning: target = demand_today + safety_stock - inventory.
    Control: urgency-rate rule distributing remaining gap over remaining steps.

    Args:
        safety_stock: Additional production target buffer.
    """

    def __init__(self, safety_stock: int = 0) -> None:
        self.safety_stock = safety_stock

    @property
    def name(self) -> str:
        return "PFA"

    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        del model
        return max(0, planning_state.demand_today + self.safety_stock - planning_state.inventory)

    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        del rng
        remaining_target_gap = max(
            0,
            execution_state.production_target - execution_state.cumulative_production,
        )
        urgency_units = max(remaining_target_gap, execution_state.remaining_demand)
        per_machine = max(1.0, model.config.mean_output_per_machine)
        required = urgency_units / max(1, execution_state.remaining_time_steps)
        machines = math.ceil(required / per_machine)
        return model.clamp_active_machines(machines)


class PFACapacityPolicy(HierarchicalPolicy):
    """PFA variant using deterministic capacity-based planning.

    Planning: target = min(mean_daily_capacity, demand - inventory + buffer).
    Control: spread remaining gap evenly using mean capacity per step.

    This is a pure state-to-decision mapping with no optimization involved.

    Args:
        planning_buffer: Additional planning buffer over deterministic target.
    """

    def __init__(self, planning_buffer: int = 0) -> None:
        self.planning_buffer = planning_buffer

    @property
    def name(self) -> str:
        return "PFA-Capacity"

    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        base = max(0, planning_state.demand_today - planning_state.inventory)
        mean_daily_capacity = (
            model.config.total_machines
            * model.config.mean_output_per_machine
            * model.config.steps_per_day
        )
        target = min(mean_daily_capacity, base + self.planning_buffer)
        return int(math.ceil(max(0.0, target)))

    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        del rng
        remaining_target_gap = max(
            0,
            execution_state.production_target - execution_state.cumulative_production,
        )
        per_step_need = remaining_target_gap / max(1, execution_state.remaining_time_steps)
        machines = math.ceil(per_step_need / max(1.0, model.config.mean_output_per_machine))
        return model.clamp_active_machines(machines)


class CFAPolicy(HierarchicalPolicy):
    """Pure CFA: single-step optimization with parameterized cost modification.

    The CFA does not look ahead over a horizon. Instead, it solves a
    single-step optimization where the machine output parameter is *modified*
    (discounted) to hedge against stochastic underperformance.

    Key idea: instead of assuming machines produce at their true mean μ,
    the optimizer uses an effective output θ_eff × μ where θ_eff ≤ 1.
    This makes the single-step optimizer activate more machines than needed
    under mean assumptions, hedging against production variance — without
    explicitly simulating or looking ahead.

    The parameters θ_eff and θ_b are tuned offline (e.g., grid search over
    simulation episodes) so that the greedy single-step decisions yield good
    multi-period performance.

    Args:
        planning_buffer: Buffer added to demand gap for planning target.
        efficiency_discount: θ_eff ∈ (0, 1] — discounts assumed per-machine
            output. Lower values → more conservative (more machines activated).
    """

    def __init__(self, planning_buffer: int = 0, efficiency_discount: float = 0.85) -> None:
        if not 0.0 < efficiency_discount <= 1.0:
            raise ValueError("efficiency_discount must be in (0, 1]")
        self.planning_buffer = planning_buffer
        self.efficiency_discount = efficiency_discount

    @property
    def name(self) -> str:
        return "CFA"

    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        del model
        base = max(0, planning_state.demand_today - planning_state.inventory)
        return base + self.planning_buffer

    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        del rng
        # Remaining production gap
        gap = max(
            0,
            execution_state.production_target - execution_state.cumulative_production,
        )
        per_step_need = gap / max(1, execution_state.remaining_time_steps)

        # Single-step modified-cost optimization:
        #   min u  s.t.  u * (θ_eff * μ) >= per_step_need,  0 <= u <= N
        # The efficiency discount θ_eff < 1 inflates the required machines,
        # implicitly penalizing machine underperformance without lookahead.
        effective_output = self.efficiency_discount * model.config.mean_output_per_machine
        machines = math.ceil(per_step_need / max(0.01, effective_output))
        return model.clamp_active_machines(machines)


class CFALookaheadPolicy(HierarchicalPolicy):
    """Hybrid CFA / deterministic DLA using receding-horizon optimization.

    This policy explicitly looks ahead over a configurable horizon H and
    optimizes machine activation to cover the lookahead window. It replaces
    stochastic machine output with its deterministic mean μ.

    Strictly speaking, this is a **deterministic DLA** (lookahead without
    sampling) rather than a pure CFA, because it explicitly optimizes over
    future time steps. It is labeled as a hybrid: the horizon H and buffer
    θ_b parameterize the optimization structure (CFA spirit), while the
    multi-step lookahead is DLA-like.

    Args:
        planning_buffer: Additional planning buffer over deterministic target.
        optimization_horizon_steps: Deterministic optimization horizon in
            intra-day time units.
    """

    def __init__(self, planning_buffer: int = 0, optimization_horizon_steps: int = 24) -> None:
        self.planning_buffer = planning_buffer
        self.optimization_horizon_steps = optimization_horizon_steps

    @property
    def name(self) -> str:
        return "CFA-Lookahead"

    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        steps_per_day = model.config.steps_per_day
        days_in_horizon = max(1, math.ceil(self.optimization_horizon_steps / steps_per_day))
        demand_in_horizon = sum(planning_state.forecast_demand[:days_in_horizon])

        required_over_horizon = max(0, demand_in_horizon - planning_state.inventory)
        mean_daily_capacity = (
            model.config.total_machines
            * model.config.mean_output_per_machine
            * model.config.steps_per_day
        )

        target = min(mean_daily_capacity, required_over_horizon + self.planning_buffer)
        return int(math.ceil(max(0.0, target)))

    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        del rng
        remaining_target_gap = max(
            0,
            execution_state.production_target - execution_state.cumulative_production,
        )
        required_units = max(remaining_target_gap, execution_state.remaining_demand)
        net_production_needed = max(0, required_units - execution_state.inventory)

        horizon_steps = max(
            1,
            min(self.optimization_horizon_steps, execution_state.remaining_time_steps),
        )
        mean_capacity_per_machine = max(1.0, model.config.mean_output_per_machine)

        # Deterministic optimizer (closed-form): minimal constant machines per
        # step that satisfy demand within the chosen horizon.
        machines = math.ceil(
            net_production_needed / (horizon_steps * mean_capacity_per_machine)
        )
        return model.clamp_active_machines(machines)


class DLAPolicy(HierarchicalPolicy):
    """Direct lookahead policy using multi-step Monte Carlo rollouts.

    Enumerates candidate actions for the current step, then simulates H steps
    forward under a greedy inner policy to estimate multi-step cost. The action
    minimizing expected H-step cost is selected.

    For steps beyond the first in each rollout, a simple rate-spread heuristic
    is used as the inner policy (ceil of remaining need per remaining steps
    divided by mean output).

    Args:
        lookahead_samples: Number of Monte Carlo rollouts per candidate action.
        lookahead_horizon: Number of future steps to simulate (H). H=1 gives
            the classic myopic one-step DLA.
        planning_buffer: Buffer added to plan target.
    """

    def __init__(
        self,
        lookahead_samples: int = 15,
        lookahead_horizon: int = 1,
        planning_buffer: int = 0,
    ) -> None:
        self.lookahead_samples = lookahead_samples
        self.lookahead_horizon = max(1, lookahead_horizon)
        self.planning_buffer = planning_buffer

    @property
    def name(self) -> str:
        return f"DLA(H={self.lookahead_horizon})"

    def plan_production(
        self,
        planning_state: PlanningState,
        model: HierarchicalProductionModel,
    ) -> int:
        del model
        return max(0, planning_state.demand_today - planning_state.inventory + self.planning_buffer)

    def _inner_policy_action(
        self,
        inventory: int,
        remaining_demand: int,
        remaining_steps: int,
        cumulative_production: int,
        production_target: int,
        model: HierarchicalProductionModel,
    ) -> int:
        """Greedy rate-spread heuristic used for future steps in rollouts."""
        gap = max(0, production_target - cumulative_production)
        per_step_need = gap / max(1, remaining_steps)
        machines = math.ceil(per_step_need / max(1.0, model.config.mean_output_per_machine))
        return model.clamp_active_machines(machines)

    def control_machines(
        self,
        execution_state: ExecutionState,
        model: HierarchicalProductionModel,
        rng: random.Random,
    ) -> int:
        best_action = 0
        best_score = float("inf")
        cost_cfg = model.cost_config

        # Effective horizon: don't look beyond remaining steps in the day
        horizon = min(self.lookahead_horizon, execution_state.remaining_time_steps)

        # Narrow the search band: estimate needed machines, then search ±3
        gap = max(0, execution_state.production_target - execution_state.cumulative_production)
        per_step_need = gap / max(1, execution_state.remaining_time_steps)
        center = math.ceil(per_step_need / max(1.0, model.config.mean_output_per_machine))
        lo = max(0, center - 3)
        hi = min(model.config.total_machines, center + 3)
        candidates = range(lo, hi + 1)

        for candidate in candidates:
            score = 0.0
            for _ in range(self.lookahead_samples):
                local_rng = random.Random(rng.randint(0, 10_000_000))

                # Simulate H steps forward
                inv = execution_state.inventory
                rem_demand = execution_state.remaining_demand
                cum_prod = execution_state.cumulative_production
                target = execution_state.production_target
                rem_steps = execution_state.remaining_time_steps

                for step in range(horizon):
                    # First step uses candidate action; subsequent use inner policy
                    if step == 0:
                        action = candidate
                    else:
                        action = self._inner_policy_action(
                            inv, rem_demand, rem_steps, cum_prod, target, model
                        )

                    produced = model.sample_step_production(action, local_rng)
                    next_inv = inv + produced
                    shipped = min(next_inv, rem_demand)
                    remaining = rem_demand - shipped
                    leftover = next_inv - shipped

                    score += (
                        remaining * cost_cfg.shortage_cost
                        + leftover * cost_cfg.holding_cost
                        + action * cost_cfg.machine_cost
                    )

                    # Advance state for next step in rollout
                    inv = leftover
                    rem_demand = remaining
                    cum_prod += produced
                    rem_steps -= 1

            expected_score = score / self.lookahead_samples
            if expected_score < best_score:
                best_score = expected_score
                best_action = candidate

        return best_action
