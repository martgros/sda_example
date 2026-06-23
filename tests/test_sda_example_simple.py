"""Tests for the hierarchical SDA simple example package."""

from sda_example_simple.evaluation import HierarchicalEvaluator
from sda_example_simple.models import CostConfig, HierarchicalProductionModel, ProductionConfig
from sda_example_simple.policies import CFAPolicy, DLAPolicy, PFAPolicy
from sda_example_simple.simulator import HierarchicalSimulator


def _build_simulator() -> HierarchicalSimulator:
    model = HierarchicalProductionModel(
        ProductionConfig(
            total_machines=8,
            steps_per_day=6,
            mean_output_per_machine=8.0,
            max_output_per_machine=14.0,
        ),
        cost_config=CostConfig(shortage_cost=50.0, holding_cost=1.0, machine_cost=0.1),
    )
    return HierarchicalSimulator(model=model)


def test_hierarchical_simulation_runs() -> None:
    simulator = _build_simulator()
    policy = PFAPolicy(safety_stock=10)
    result = simulator.run_episode(
        initial_inventory=30,
        demand_forecast=[90, 95, 100],
        policy=policy,
        seed=3,
    )

    assert len(result.day_results) == 3
    assert result.total_cost >= 0


def test_multiple_policies_are_evaluable() -> None:
    simulator = _build_simulator()
    evaluator = HierarchicalEvaluator(simulator)

    for policy in [PFAPolicy(5), CFAPolicy(5), DLAPolicy(lookahead_samples=5)]:
        summary = evaluator.evaluate_policy(
            policy=policy,
            initial_inventory=20,
            demand_forecast=[85, 90, 88],
            episodes=3,
            seed=11,
        )
        assert summary.mean_total_cost >= 0
        assert 0.0 <= summary.mean_service_level <= 1.0
