"""Tests for the manufacturing simulator and policy evaluation flows."""

from sda_example.evaluation import PolicyEvaluationEnabler, ScenarioEvaluator
from sda_example.models import CostConfig, ManufacturingModel, ManufacturingState
from sda_example.policies import PFAPolicy, PolicyBlock
from sda_example.simulator import ManufacturingSimulator
from sda_example.tuning import CFATuner, DLATuner, PFATuner


def test_simulator_produces_non_negative_inventory() -> None:
    model = ManufacturingModel(
        total_machines=5,
        avg_machine_output=40.0,
        max_machine_output=80,
        cost_config=CostConfig(inventory_holding_cost=1.0, shortage_penalty_cost=20.0),
    )
    simulator = ManufacturingSimulator(model=model)
    initial_state = ManufacturingState(
        day_index=0,
        inventory=50,
        today_demand=100,
        forecast_demands=[100, 90, 110],
    )

    history = simulator.simulate_horizon(
        initial_state=initial_state,
        decisions=[3, 3, 3],
        seed=42,
    )

    assert all(step.ending_inventory >= 0 for step in history)
    assert all(step.production >= 0 for step in history)


def test_tuner_selects_candidate_from_search_space() -> None:
    model = ManufacturingModel(
        total_machines=6,
        avg_machine_output=45.0,
        max_machine_output=90,
        cost_config=CostConfig(inventory_holding_cost=1.2, shortage_penalty_cost=30.0),
    )
    simulator = ManufacturingSimulator(model=model)
    evaluator = ScenarioEvaluator(simulator=simulator, scenarios=20)
    tuner = PFATuner(evaluator=evaluator, horizon_days=7)

    result = tuner.tune_safety_stock(
        candidates=[0, 20, 40],
        initial_inventory=30,
        oem_today_demand=100,
        forecast_demands=[100, 105, 110, 108, 112, 115, 113],
        seed=7,
    )

    assert result.best_safety_stock in {0, 20, 40}
    assert len(result.score_by_safety_stock) == 3


def test_evaluation_returns_horizon_traces() -> None:
    model = ManufacturingModel(
        total_machines=6,
        avg_machine_output=45.0,
        max_machine_output=90,
        cost_config=CostConfig(inventory_holding_cost=1.2, shortage_penalty_cost=30.0),
    )
    simulator = ManufacturingSimulator(model=model)
    evaluator = ScenarioEvaluator(simulator=simulator, scenarios=10)
    block = PolicyBlock(PFAPolicy(safety_stock=25))

    result = evaluator.evaluate_policy(
        policy_block=block,
        initial_inventory=30,
        oem_today_demand=100,
        forecast_demands=[100, 105, 110, 108, 112, 115, 113],
        horizon_days=7,
        seed=2,
    )

    assert result.expected_total_cost >= 0
    assert len(result.expected_inventory_trace) == 7
    assert len(result.expected_production_trace) == 7
    assert len(result.expected_shortage_trace) == 7


def test_enabler_evaluates_policy_list() -> None:
    model = ManufacturingModel(
        total_machines=6,
        avg_machine_output=45.0,
        max_machine_output=90,
        cost_config=CostConfig(inventory_holding_cost=1.2, shortage_penalty_cost=30.0),
    )
    simulator = ManufacturingSimulator(model=model)
    evaluator = ScenarioEvaluator(simulator=simulator, scenarios=10)
    enabler = PolicyEvaluationEnabler(evaluator=evaluator)

    policies = [PFAPolicy(safety_stock=20), PFAPolicy(safety_stock=40)]
    results = enabler.evaluate_policies(
        policies=policies,
        initial_inventory=30,
        oem_today_demand=100,
        forecast_demands=[100, 105, 110, 108, 112, 115, 113],
        horizon_days=7,
        seed=2,
    )

    assert len(results) == 2
    assert all(value.expected_total_cost >= 0 for value in results.values())


def test_cfa_and_dla_tuners_return_valid_candidates() -> None:
    model = ManufacturingModel(
        total_machines=6,
        avg_machine_output=45.0,
        max_machine_output=90,
        cost_config=CostConfig(inventory_holding_cost=1.2, shortage_penalty_cost=30.0),
    )
    simulator = ManufacturingSimulator(model=model)
    evaluator = ScenarioEvaluator(simulator=simulator, scenarios=10)

    cfa_tuner = CFATuner(evaluator=evaluator, horizon_days=7)
    cfa_result = cfa_tuner.tune_demand_buffer(
        candidates=[0, 5, 10],
        initial_inventory=30,
        oem_today_demand=100,
        forecast_demands=[100, 105, 110, 108, 112, 115, 113],
        seed=2,
    )
    assert cfa_result.best_demand_buffer in {0, 5, 10}

    dla_tuner = DLATuner(evaluator=evaluator, horizon_days=7)
    dla_result = dla_tuner.tune_lookahead(
        candidates=[(2, 10), (3, 20)],
        initial_inventory=30,
        oem_today_demand=100,
        forecast_demands=[100, 105, 110, 108, 112, 115, 113],
        seed=2,
    )
    assert (dla_result.best_lookahead_horizon, dla_result.best_lookahead_scenarios) in {
        (2, 10),
        (3, 20),
    }
