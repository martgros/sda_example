"""Evaluation and plotting helpers for hierarchical SDA production control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from .models import HierarchicalProductionModel
from .policies import HierarchicalPolicy
from .simulator import EpisodeResult, HierarchicalSimulator


@dataclass(frozen=True)
class EpisodeSummary:
    """Aggregate metrics over repeated episodes.

    Args:
        policy_name: Evaluated policy name.
        mean_total_cost: Average total cost across episodes.
        mean_service_level: Average service level across episodes.
    """

    policy_name: str
    mean_total_cost: float
    mean_service_level: float


class HierarchicalEvaluator:
    """Runs multi-episode evaluation and educational visualizations.

    Args:
        simulator: Hierarchical simulator.
    """

    def __init__(self, simulator: HierarchicalSimulator) -> None:
        self.simulator = simulator

    def evaluate_policy(
        self,
        policy: HierarchicalPolicy,
        initial_inventory: int,
        demand_forecast: Sequence[int],
        episodes: int,
        seed: int = 0,
    ) -> EpisodeSummary:
        """Evaluate a policy over repeated episodes.

        Args:
            policy: Policy to evaluate.
            initial_inventory: Initial inventory for each episode.
            demand_forecast: Deterministic daily demand profile.
            episodes: Number of episodes.
            seed: Base seed.

        Returns:
            Aggregate metrics.
        """

        total_cost = 0.0
        total_service_level = 0.0
        total_demand = sum(demand_forecast)

        for idx in range(episodes):
            result = self.simulator.run_episode(
                initial_inventory=initial_inventory,
                demand_forecast=demand_forecast,
                policy=policy,
                seed=seed + idx,
            )
            total_cost += result.total_cost
            unmet = sum(day.shortage for day in result.day_results)
            service_level = 1.0 if total_demand == 0 else 1.0 - unmet / total_demand
            total_service_level += service_level

        return EpisodeSummary(
            policy_name=policy.name,
            mean_total_cost=total_cost / episodes,
            mean_service_level=total_service_level / episodes,
        )

    def plot_machine_randomness(
        self,
        model: HierarchicalProductionModel,
        samples: int = 400,
        seed: int = 0,
    ) -> None:
        """Visualize single-machine production randomness.

        Args:
            model: Production model providing machine output sampling.
            samples: Number of random draws.
            seed: Seed for reproducibility.
        """

        import random

        rng = random.Random(seed)
        outputs = [model.sample_machine_output(rng) for _ in range(samples)]

        plt.figure(figsize=(9, 4))
        plt.hist(outputs, bins=100, color="#4C78A8", alpha=0.85, edgecolor="white")
        plt.axvline(
            model.config.mean_output_per_machine,
            color="#F58518",
            linestyle="--",
            linewidth=2,
            label="Configured mean",
        )
        plt.title("Randomness of Single-Machine Production per Step")
        plt.xlabel("Produced units")
        plt.ylabel("Frequency")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_target_vs_realized(self, simulator_result: EpisodeResult) -> None:
        """Plot daily and intra-day dynamics with interactive Plotly subplots.

        Args:
            simulator_result: One full episode result.
        """

        days = [day.day_index + 1 for day in simulator_result.day_results]
        targets = [day.production_target for day in simulator_result.day_results]
        realized = [day.total_production for day in simulator_result.day_results]

        step_positions: list[int] = []
        per_step_production: list[int] = []
        per_step_inventory: list[int] = []
        per_step_active_machines: list[int] = []
        day_boundaries: list[int] = []
        step_counter = 0

        for day in simulator_result.day_results:
            for step in day.step_results:
                step_counter += 1
                step_positions.append(step_counter)
                per_step_production.append(step.produced_units)
                per_step_inventory.append(step.ending_inventory)
                per_step_active_machines.append(step.active_machines)
            day_boundaries.append(step_counter)

        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=(
                "Target vs Realized Daily Production",
                "Intra-Day Production and Inventory Evolution",
                "Intra-Day Decision Evolution (Active Machines)",
            ),
        )

        figure.add_trace(
            go.Scatter(
                x=days,
                y=targets,
                mode="lines+markers",
                name="Production target",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=days,
                y=realized,
                mode="lines+markers",
                name="Realized production",
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=step_positions,
                y=per_step_production,
                mode="lines+markers",
                name="Produced units per step",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=step_positions,
                y=per_step_inventory,
                mode="lines+markers",
                name="Ending inventory per step",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=step_positions,
                y=per_step_active_machines,
                mode="lines+markers",
                name="Active machines per step",
                line={"color": "#54A24B"},
            ),
            row=3,
            col=1,
        )

        for boundary in day_boundaries[:-1]:
            figure.add_vline(
                x=boundary + 0.5,
                line_dash="dash",
                line_color="#BBBBBB",
                line_width=1,
                row=2,
                col=1,
            )
            figure.add_vline(
                x=boundary + 0.5,
                line_dash="dash",
                line_color="#BBBBBB",
                line_width=1,
                row=3,
                col=1,
            )

        figure.update_xaxes(title_text="Day", row=1, col=1)
        figure.update_yaxes(title_text="Units", row=1, col=1)
        figure.update_xaxes(title_text="Global step index", row=2, col=1)
        figure.update_yaxes(title_text="Units", row=2, col=1)
        figure.update_xaxes(title_text="Global step index", row=3, col=1)
        figure.update_yaxes(title_text="Active machines", row=3, col=1)

        figure.update_layout(
            height=950,
            width=1100,
            title_text="Hierarchical Production Dynamics",
            legend={"orientation": "h", "y": -0.08},
            template="plotly_white",
        )
        figure.show()
