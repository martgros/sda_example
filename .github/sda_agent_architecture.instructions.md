---
applyTo: "src/*.py", "src/*/*.py"
description: "Architecture guardrails for Sequential Decision Analytics agents"
---

# Sequential Decision Analytics Agent Instructions

Use this file to guide any coding agent implementing or extending an SDA problem in this codebase.

## Purpose

Enforce a clean SDA architecture with strict separation of concerns:

- Model: one-step (possibly stochastic) system dynamics.
- Policy: decision logic only.
- Simulator: horizon rollout orchestration.
- Evaluator: scenario generation and policy scoring.
- Tuner: parameter search over policy families (brute-force, knowledge gradient, ...).

## Architecture Contracts

### 1) Model Layer (`models.py`)

The model layer owns the single-step transition and state representation.

Required responsibilities:

- Define immutable data classes for state, decisions, and outputs.
- Implement one-step dynamics in a model class (e.g. `ManufacturingModel.step`).
- Implement deterministic state evolution (`transition`) from one time step to the next.
- Own stochastic sampling primitives used by `step` (if they are present).

Do not:

- Put policy logic in the model.
- Put horizon loops in the model.
- Put Monte Carlo evaluation loops in the model.

### 2) Policy Layer (`policies.py`)

Policies must implement `BasePolicy` and only return a decision for the current state.

Required interface:

- `name` property.
- `decide(state, model, rng) -> int`.

Rules:

- Policies are of one of the 3 types namely Policy Function Approximation (PFA), Cost Function Approximation (CFA) or Direct Lookahead Approximation (DLA)
- Policies can read state and model parameters.
- Policies may use the model itself for online scenario simulations (for DLA-like methods).
- Policies may use `rng` for stochastic lookahead behavior (for DLA-like methods).
- Policies must not mutate state.
- Policies must not run full scenario evaluation loops.

Use `PolicyBlock` as generic wrapper when integrating policies into evaluator or simulator flows.

### 3) Simulator Layer (`simulator.py`)

Simulator orchestrates time progression only.

Required responsibilities:

- Run fixed decision trajectories (`simulate_horizon`).
- Run policy-driven trajectories (`simulate_policy`).
- Apply `model.step` each day and `model.transition` to advance state.

Do not:

- Re-implement model equations in simulator.
- Re-implement scenario sampling in simulator.

### 4) Evaluator Layer (`evaluation.py`)

Evaluator owns uncertainty over scenarios and computes expected metrics.

Required responsibilities:

- Generate demand scenarios from forecast distribution.
- Evaluate policy performance across many scenarios.
- Return aggregated outputs (`EvaluationResult`).

Use `PolicyEvaluationEnabler` for user-facing utilities:

- Evaluate lists of policy instances in one call.
- Plot single policy outcomes.
- Plot cross-policy cost comparisons.
- Plot probabilistic forecast distribution (sample paths + percentile bands).

### 5) Tuning Layer (`tuning.py`)

Tuners own (brute-force, knowledge gradient,...) parameter search for each policy family.

Current pattern:

- `PFATuner`: tunes parameters of the function.
- `CFATuner`: tunes hyper parameters of the optimization model (penalty terms, cost function weights, time horizon...).
- `DLATuner`: tunes hyper parameters of the scenario sampling and Monte Carlo runs (number of runs, way how inputs/scenarios are samples/selected, time horizon).

Rules:

- Tuners call evaluator, never simulator directly.
- Tuners are policy-family specific and return typed result models.
- Tuners must reject empty candidate lists.

## Powell SDA Mapping

When implementing new problems, preserve this mapping explicitly:

- State `S_t`: resource + information state + belief state (estimates, forecasts) (e.g. `inventory`, `today_demand`, remaining forecast, day index).
- Decision `x_t`: `MachineDecision(active_machines=...)` or equivalent typed decision object.
- Exogenous information `W_{t+1}`: sampled demand and sampled production noise.
- Transition `S_{t+1} = M(S_t, x_t, W_{t+1})`: implemented by `step` + `transition`.
- Objective: expected cumulative cost over horizon and scenarios.

## Extension Playbook

When adding a new policy family:

1. Add policy class implementing `BasePolicy` in `policies.py`.
2. Add parameter-aware label support in `PolicyEvaluationEnabler._policy_label`.
3. Add a dedicated tuner class and typed result dataclass if tunable.
4. Add unit tests for:
   - policy output validity,
   - tuner candidate handling,
   - evaluator integration.
5. Update package exports in `__init__.py`.

When adding a new uncertainty model:

1. Add sampling function(s) to evaluator.
2. Keep one-step model stochastic pieces in model layer.
3. Ensure plotting utilities visualize distribution, not just means.

## Guardrails and Anti-Patterns

Never do the following:

- Duplicate one-step equations in multiple modules.
- Let policies call evaluators or tuners.
- Let simulator perform Monte Carlo aggregation.
- Couple notebook/demo code to internal private functions.
- Return untyped dictionaries where typed dataclasses are expected.

## Quality Requirements

For every architecture-level change:

- Keep type hints on all public function signatures.
- Preserve deterministic reproducibility via explicit `seed` usage.
- Run:
  - `uv run ruff check src`
  - `uv run mypy src/`
  - `uv run python -m pytest -v`

## Minimal Definition of Done

A contribution is complete only if:

- Module boundaries above are respected.
- New functionality is covered by tests.
- Evaluator can compare multiple policies through one interface.
- Notebook-level usage remains concise and composes through enabler and tuners.
