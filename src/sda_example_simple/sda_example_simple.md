# Sequential Decision Analytics Example (simple): Hierarchical Production Control

## 1. Problem Description

### Scenario
A Tier-1 automotive supplier operates a production line with a fixed number of identical machines to fulfill daily OEM demand. Each day starts with a known demand for that day and a deterministic forecast for the next several days.

The system is subject to *execution uncertainty*: machines may underperform or fail, causing production shortfalls. This creates a gap between planned production (target) and realized production (reality).

Decisions are made at two levels:

---
### Decision Hierarchy

#### Level 1: Daily Planning (Target Setting)
At the start of each day, decide:
- **Production target for the day**

This decision uses:
- Current inventory
- Known demand today
- Deterministic multi-day demand forecast

---

#### Level 2: Intra-Day Control (Execution Adjustment)
During the day (split into time steps), decide:
- **Number of machines to run at each time step**

Goal:
- Track the daily production target despite execution uncertainty

---

### State

#### Planning State (Start of Day)
- Inventory level
- Day index
- Remaining forecast demand (deterministic)

#### Execution State (During Day)
- Current inventory
- Remaining demand today
- Remaining time steps in the day
- Cumulative production vs. target

---

### Uncertainty

Only one uncertainty source:
- Machines may **underperform or fail** during execution

Effect:
- Realized production is typically below planned output
- Shortfall risk accumulates over time

---

### Constraints

- Number of active machines ≤ total machines available
- Inventory balance:

  inventory_{t+1} = inventory_t + production_t − shipments_t

- Shipments are limited by available inventory and demand

---

### Dynamics (Target vs Reality)

- A daily production target is set
- Actual production evolves stochastically
- Control must adjust to close the gap between target and realized output

---

### Objective

Minimize total cost across the horizon:

- Shortage penalty (very high) at end of each day
- Inventory holding cost
- Optional machine operating cost

---

## 2. Implementation Plan (For Python Agent)

### Step 1: Define Core Parameters

- N_machines
- T_steps_per_day
- Planning_horizon_days
- Mean_output_per_machine
- Underperformance_factor_distribution
- Costs:
  - shortage_cost
  - holding_cost
  - machine_cost

---

### Step 2: Data Structures

#### State Classes

Create two logical states:

1. PlanningState:
```python
inventory
current_day_index
forecast_demand (list)
```

2. ExecutionState:
```python
inventory
remaining_demand
remaining_time_steps
cumulative_production
production_target
```

---

### Step 3: Environment Simulation

#### Day Loop
For each day:
1. Build PlanningState
2. Call planning policy → returns production_target
3. Initialize ExecutionState
4. Run intra-day loop
5. Apply end-of-day costs
6. Update inventory

---

#### Intra-Day Loop

For each time step:

1. Call control policy → returns active machines
2. Simulate production:
   - Each machine produces:
     ```python
     output = mean_output * performance_factor
     ```
   - performance_factor sampled from distribution skewed toward underperformance

3. Aggregate production
4. Update inventory
5. Ship demand:
   ```python
   shipped = min(inventory, remaining_demand)
   ```
6. Update:
   - inventory
   - remaining_demand
   - cumulative production

---

### Step 4: Uncertainty Modeling

Implement machine performance as:

```python
performance_factor = random.betavariate(a, b)
```

Choose parameters such that:
- mean < 1
- skew toward underperformance
- optionally clip maximum at 1.1

---

### Step 5: Cost Calculation

At end of each day:

```python
shortage = max(0, remaining_demand)
holding = max(0, inventory)

cost = shortage * shortage_cost + holding * holding_cost
```

Add machine operating costs during execution if desired.

---

### Step 6: Policy Interfaces

Define two separate policies:

#### Planning Policy
```python
def plan_production(planning_state) -> production_target:
    pass
```

#### Control Policy
```python
def control_machines(execution_state) -> active_machines:
    pass
```

---

### Step 7: SDA Policy Implementations

#### PFA
- Rule-based target
- Rule-based machine adjustments based on backlog and time remaining

#### CFA
- Compute target using deterministic approximation:
  - sum of expected demand − inventory
- Use expected machine output for planning

#### DLA

At decision points:
1. Try candidate targets or machine actions
2. Simulate forward (Monte Carlo)
3. Select action with best expected cost

---

### Step 8: Simulation Driver

- Run multiple episodes
- Track:
  - total cost
  - service level
  - inventory levels

---

### Step 9: Reproducibility

- Set random seed
- Allow configurable parameters

---

### Step 10: Output & Visualization (Optional)

- Plot inventory trajectory
- Plot target vs. actual production
- Compare policies

---

## Summary

This structured problem cleanly separates planning and execution, highlights the gap between target and reality, and provides a strong foundation to implement and compare SDA policies in a realistic manufacturing setting.
