"""Grid-search tuning for hierarchical SDA policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .evaluation import HierarchicalEvaluator
from .policies import (
    CFALookaheadPolicy,
    CFAPolicy,
    DLAPolicy,
    HierarchicalPolicy,
    PFACapacityPolicy,
    PFAPolicy,
)


@dataclass(frozen=True)
class TuningResult:
    """Result of a policy tuning run.

    Args:
        best_policy: Policy instance with the best-performing parameters.
        best_cost: Mean total cost achieved by best policy.
        best_service_level: Mean service level achieved by best policy.
        all_results: List of (policy, mean_cost, mean_service_level) tuples
            for each candidate evaluated, ordered by mean_cost ascending.
    """

    best_policy: HierarchicalPolicy
    best_cost: float
    best_service_level: float
    all_results: list[tuple[HierarchicalPolicy, float, float]]


class PolicyTuner:
    """Grid-search tuner for hierarchical policies.

    Evaluates candidate policies over repeated episodes and returns the
    best-performing configuration.

    Args:
        evaluator: Hierarchical evaluator for multi-episode scoring.
        initial_inventory: Starting inventory for each episode.
        demand_forecast: Deterministic demand profile over horizon.
        episodes: Number of Monte Carlo episodes per candidate.
        seed: Base random seed.
    """

    def __init__(
        self,
        evaluator: HierarchicalEvaluator,
        initial_inventory: int,
        demand_forecast: Sequence[int],
        episodes: int = 30,
        seed: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.initial_inventory = initial_inventory
        self.demand_forecast = demand_forecast
        self.episodes = episodes
        self.seed = seed

    def tune(self, candidates: Sequence[HierarchicalPolicy]) -> TuningResult:
        """Evaluate all candidate policies and return the best.

        Args:
            candidates: Policy instances to evaluate.

        Returns:
            Tuning result with ranked candidates.

        Raises:
            ValueError: If candidates is empty.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")

        results: list[tuple[HierarchicalPolicy, float, float]] = []
        for policy in candidates:
            summary = self.evaluator.evaluate_policy(
                policy=policy,
                initial_inventory=self.initial_inventory,
                demand_forecast=self.demand_forecast,
                episodes=self.episodes,
                seed=self.seed,
            )
            results.append((policy, summary.mean_total_cost, summary.mean_service_level))

        results.sort(key=lambda x: x[1])
        best_policy, best_cost, best_service = results[0]

        return TuningResult(
            best_policy=best_policy,
            best_cost=best_cost,
            best_service_level=best_service,
            all_results=results,
        )

    def tune_pfa(
        self,
        safety_stock_candidates: Sequence[int],
    ) -> TuningResult:
        """Tune PFA safety stock parameter.

        Args:
            safety_stock_candidates: Values to search over.

        Returns:
            Tuning result.
        """
        candidates = [PFAPolicy(safety_stock=ss) for ss in safety_stock_candidates]
        return self.tune(candidates)

    def tune_pfa_capacity(
        self,
        planning_buffer_candidates: Sequence[int],
    ) -> TuningResult:
        """Tune PFA-Capacity planning buffer.

        Args:
            planning_buffer_candidates: Values to search over.

        Returns:
            Tuning result.
        """
        candidates = [PFACapacityPolicy(planning_buffer=pb) for pb in planning_buffer_candidates]
        return self.tune(candidates)

    def tune_cfa(
        self,
        planning_buffer_candidates: Sequence[int],
        efficiency_discount_candidates: Sequence[float],
    ) -> TuningResult:
        """Tune CFA planning buffer and efficiency discount jointly.

        Args:
            planning_buffer_candidates: Buffer values.
            efficiency_discount_candidates: Efficiency discount values in (0,1].

        Returns:
            Tuning result.
        """
        candidates = [
            CFAPolicy(planning_buffer=pb, efficiency_discount=ed)
            for pb in planning_buffer_candidates
            for ed in efficiency_discount_candidates
        ]
        return self.tune(candidates)

    def tune_cfa_lookahead(
        self,
        planning_buffer_candidates: Sequence[int],
        horizon_candidates: Sequence[int],
    ) -> TuningResult:
        """Tune CFA-Lookahead planning buffer and optimization horizon.

        Args:
            planning_buffer_candidates: Buffer values.
            horizon_candidates: Optimization horizon values (in steps).

        Returns:
            Tuning result.
        """
        candidates = [
            CFALookaheadPolicy(planning_buffer=pb, optimization_horizon_steps=h)
            for pb in planning_buffer_candidates
            for h in horizon_candidates
        ]
        return self.tune(candidates)

    def tune_dla(
        self,
        lookahead_samples_candidates: Sequence[int],
        lookahead_horizon_candidates: Sequence[int],
        planning_buffer_candidates: Sequence[int],
    ) -> TuningResult:
        """Tune DLA samples, horizon, and planning buffer jointly.

        Args:
            lookahead_samples_candidates: Sample counts per candidate action.
            lookahead_horizon_candidates: Lookahead horizon values.
            planning_buffer_candidates: Buffer values.

        Returns:
            Tuning result.
        """
        candidates = [
            DLAPolicy(
                lookahead_samples=k,
                lookahead_horizon=h,
                planning_buffer=pb,
            )
            for k in lookahead_samples_candidates
            for h in lookahead_horizon_candidates
            for pb in planning_buffer_candidates
        ]
        return self.tune(candidates)
