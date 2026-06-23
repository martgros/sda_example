"""Tests for policy tuning in sda_example_simple."""

import pytest

from sda_example_simple.models import CostConfig, HierarchicalProductionModel, ProductionConfig
from sda_example_simple.evaluation import HierarchicalEvaluator
from sda_example_simple.policies import CFAPolicy, DLAPolicy, PFAPolicy
from sda_example_simple.simulator import HierarchicalSimulator
from sda_example_simple.tuning import PolicyTuner, TuningResult


def _build_tuner() -> PolicyTuner:
    model = HierarchicalProductionModel(
        ProductionConfig(
            total_machines=8,
            steps_per_day=6,
            mean_output_per_machine=8.0,
            max_output_per_machine=14.0,
        ),
        cost_config=CostConfig(shortage_cost=50.0, holding_cost=1.0, machine_cost=0.1),
    )
    simulator = HierarchicalSimulator(model=model)
    evaluator = HierarchicalEvaluator(simulator=simulator)
    return PolicyTuner(
        evaluator=evaluator,
        initial_inventory=20,
        demand_forecast=[90, 95, 100],
        episodes=5,
        seed=42,
    )


def test_tune_pfa_returns_best() -> None:
    tuner = _build_tuner()
    result = tuner.tune_pfa(safety_stock_candidates=[0, 10, 20, 50])

    assert isinstance(result, TuningResult)
    assert isinstance(result.best_policy, PFAPolicy)
    assert result.best_cost >= 0
    assert 0.0 <= result.best_service_level <= 1.0
    assert len(result.all_results) == 4
    # Results should be sorted by cost ascending
    costs = [r[1] for r in result.all_results]
    assert costs == sorted(costs)


def test_tune_pfa_capacity_returns_best() -> None:
    tuner = _build_tuner()
    result = tuner.tune_pfa_capacity(planning_buffer_candidates=[0, 10, 30])

    assert isinstance(result, TuningResult)
    assert result.best_cost >= 0
    assert len(result.all_results) == 3


def test_tune_cfa_returns_best() -> None:
    tuner = _build_tuner()
    result = tuner.tune_cfa(
        planning_buffer_candidates=[0, 20],
        efficiency_discount_candidates=[0.7, 0.85, 1.0],
    )

    assert isinstance(result, TuningResult)
    assert isinstance(result.best_policy, CFAPolicy)
    assert len(result.all_results) == 6  # 2 * 3


def test_tune_cfa_lookahead_returns_best() -> None:
    tuner = _build_tuner()
    result = tuner.tune_cfa_lookahead(
        planning_buffer_candidates=[0, 15],
        horizon_candidates=[6, 12],
    )

    assert isinstance(result, TuningResult)
    assert len(result.all_results) == 4  # 2 * 2


def test_tune_dla_returns_best() -> None:
    tuner = _build_tuner()
    result = tuner.tune_dla(
        lookahead_samples_candidates=[5],
        lookahead_horizon_candidates=[1, 3],
        planning_buffer_candidates=[0, 10],
    )

    assert isinstance(result, TuningResult)
    assert isinstance(result.best_policy, DLAPolicy)
    assert len(result.all_results) == 4  # 1 * 2 * 2


def test_tune_generic_with_mixed_policies() -> None:
    tuner = _build_tuner()
    result = tuner.tune([
        PFAPolicy(safety_stock=10),
        CFAPolicy(planning_buffer=10, efficiency_discount=0.8),
        DLAPolicy(lookahead_samples=5, lookahead_horizon=2, planning_buffer=5),
    ])

    assert isinstance(result, TuningResult)
    assert len(result.all_results) == 3
    assert result.best_cost == result.all_results[0][1]


def test_tune_empty_candidates_raises() -> None:
    tuner = _build_tuner()
    with pytest.raises(ValueError, match="candidates must not be empty"):
        tuner.tune([])
