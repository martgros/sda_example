# Hierarchical Production Control — Model and Policy Reference

This module implements a two-level Sequential Decision Analytics problem
for a Tier-1 automotive supplier, following the framework from
*Reinforcement Learning and Stochastic Optimization* (Warren B. Powell, 2022).

A detailed description of the problem setting can be found [here](./sda_example_simple.md).

---

## SDA Framework Mapping

Powell's universal model for sequential decision problems consists of five
core elements. The table below shows how each maps to our implementation:

| Powell Element | Symbol | Our Implementation |
|---|---|---|
| State | $S_t$ | `PlanningState` (inventory, day index, demand, forecast) + `ExecutionState` (inventory, remaining demand, remaining steps, cumulative production, target) |
| Decision | $x_t$ | Two-level: daily production target $Q_d$ + per-step active machines $u_{d,\tau}$ |
| Exogenous information | $W_{t+1}$ | Beta-distributed machine output — realized *after* the decision is made modeling the uncertainty of machine performance (e.g. due to failues, human operator, ...) |
| Transition function | $S^M$ | $I_{d,\tau+1} = I_{d,\tau} + p_{d,\tau} - \text{shipped}_{d,\tau}$ |
| Objective function | $\min \mathbb{E}[\sum C_d]$ | Minimize expected total shortage + holding + machine cost over horizon |

The key insight: demand is deterministic (known in advance, that is why this example is the *simple* one), but **machine
output is stochastic** — creating genuine sequential decision-making under
uncertainty where the policy must react to realized production outcomes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $d$ | Day index, $d = 0, 1, \ldots, D-1$ |
| $\tau$ | Intra-day step index, $\tau = 0, 1, \ldots, T-1$ |
| $I_{d,\tau}$ | Inventory at day $d$, step $\tau$ |
| $R_{d,\tau}$ | Remaining demand at day $d$, step $\tau$ |
| $\tau^{\text{rem}}_{d,\tau}$ | Remaining steps in day $d$ at step $\tau$ |
| $P_{d,\tau}$ | Cumulative production at day $d$, step $\tau$ |
| $Q_d$ | Daily production target set by planning policy |
| $u_{d,\tau} \in \{0,\ldots,N\}$ | Active machines at step $\tau$ of day $d$ |
| $N$ | Total available machines |
| $T$ | Control steps per day |
| $\mu$ | Mean output per machine per step |
| $\bar{q}$ | Maximum output per machine per step |
| $a$ | Beta distribution shape parameter (alpha) |
| $c_s$ | Shortage penalty per unit |
| $c_h$ | Inventory holding cost per unit |
| $c_m$ | Machine operating cost per active machine per step |

---

## Model

### Machine Uncertainty

Each machine $i$ at each step draws an independent performance factor:

$$
\phi_i \sim \mathrm{Beta}(a,\; b), \quad \phi_i \in [0, 1]
$$

where $b$ is derived to satisfy $\mathbb{E}[\phi_i] = \mu / \bar{q}$:

$$
b = \max\!\left(\, a \cdot \left(\frac{\bar{q}}{\mu} - 1\right),\; a + 0.25 \right)
$$

The realized output of machine $i$ is:

$$
q_i = \mathrm{round}\!\left(\phi_i \cdot \bar{q}\right)
$$

Machines are skewed toward underperformance because $b > a$ when $\mu < \bar{q}/2$.

### Step Production

Total realized production at step $\tau$ with $u_{d,\tau}$ active machines:

$$
p_{d,\tau} = \sum_{i=1}^{u_{d,\tau}} q_i
$$

### Inventory Transition

At each step $\tau$, after production the supplier ships as much as possible:

$$
\text{shipped}_{d,\tau} = \min\!\left(I_{d,\tau} + p_{d,\tau},\; R_{d,\tau}\right)
$$

State updates:

$$
I_{d,\tau+1} = I_{d,\tau} + p_{d,\tau} - \text{shipped}_{d,\tau}
$$

$$
R_{d,\tau+1} = R_{d,\tau} - \text{shipped}_{d,\tau}
$$

### End-of-Day Costs

At $\tau = T$ (end of day $d$):

$$
\text{shortage}_d = R_{d,T}
$$

$$
C_d = c_s \cdot \text{shortage}_d \;+\; c_h \cdot I_{d,T} \;+\; c_m \cdot \sum_{\tau=0}^{T-1} u_{d,\tau}
$$

### Horizon Objective

$$
\min \sum_{d=0}^{D-1} C_d
$$

---

## State Representation (Powell SDA Framework)

### Planning State $S^{\text{plan}}_d$

Visible at the start of day $d$, used to set the daily target:

$$
S^{\text{plan}}_d = \left(I_{d,0},\; d,\; \delta_d,\; \boldsymbol{\delta}_{d:D}\right)
$$

where $\delta_d$ is today's deterministic demand and $\boldsymbol{\delta}_{d:D}$ is the remaining forecast.

### Execution State $S^{\text{exec}}_{d,\tau}$

Visible at each intra-day step, used to set active machines:

$$
S^{\text{exec}}_{d,\tau} = \left(I_{d,\tau},\; R_{d,\tau},\; \tau^{\text{rem}}_{d,\tau},\; P_{d,\tau},\; Q_d\right)
$$

Note: $Q_d$ is the planning decision made at the start of the day. From the
control policy's perspective, it is **exogenous information** — known before
control begins, but not under the control policy's authority. The control
policy conditions on both the realized physical state and the prior plan,
making it a hierarchical/rolling-horizon architecture common in practice.

---

## Policies

All policies implement two decisions:

- **Planning decision** $Q_d = \pi^{\text{plan}}(S^{\text{plan}}_d)$
- **Control decision** $u_{d,\tau} = \pi^{\text{ctrl}}(S^{\text{exec}}_{d,\tau})$

Following Powell's four meta-classes for sequential decision policies, we
implement PFA, CFA, and DLA (omitting VFA for simplicity).

---

### Policy Function Approximation — Safety Stock (PFA)

A parametric rule mapping state directly to a decision without solving any
optimization problem.

**Parameter:** $\theta_s \in \mathbb{Z}_{\geq 0}$ — safety stock.

**Planning:**

$$
Q_d^{\text{PFA}} = \max\!\left(0,\; \delta_d + \theta_s - I_{d,0}\right)
$$

**Control:**

The gap between remaining target and remaining demand drives urgency:

$$
\text{gap}_\tau = \max\!\left(Q_d - P_{d,\tau},\; R_{d,\tau}\right)
$$

Required rate per step:

$$
r_\tau = \frac{\text{gap}_\tau}{\max(1, \tau^{\text{rem}}_{d,\tau})}
$$

Active machines:

$$
u_{d,\tau}^{\text{PFA}} = \mathrm{clamp}\!\left(\left\lceil \frac{r_\tau}{\mu} \right\rceil,\; 0,\; N\right)
$$

---

### Policy Function Approximation — Capacity (PFA-Capacity)

A second PFA variant that incorporates capacity awareness into both planning
and control. Still a pure state-to-decision mapping — no optimization is solved.

**Parameter:** $\theta_b \in \mathbb{Z}_{\geq 0}$ — planning buffer.

**Planning:**

Daily mean capacity is $\bar{C} = N \cdot \mu \cdot T$. The target is capped
at capacity:

$$
Q_d^{\text{PFA-Cap}} = \min\!\left(\bar{C},\; \max(0,\; \delta_d - I_{d,0}) + \theta_b\right)
$$

**Control:**

Spread remaining target gap evenly over remaining steps using expected machine
output:

$$
u_{d,\tau}^{\text{PFA-Cap}} = \mathrm{clamp}\!\left(
\left\lceil
\frac{\max(0,\; Q_d - P_{d,\tau})}{\max(1, \tau^{\text{rem}}_{d,\tau}) \cdot \mu}
\right\rceil,\; 0,\; N\right)
$$

---

### Cost Function Approximation (CFA)

The CFA solves a **single-step optimization with parameterized cost
modification**. No lookahead over future steps is performed. Instead, the
optimizer uses a *modified* machine output parameter to hedge against
stochastic underperformance.

Key insight: instead of assuming machines produce at their true mean $\mu$,
the optimizer uses an effective output $\theta_e \cdot \mu$ where
$\theta_e \leq 1$. This makes the single-step optimizer activate more
machines than needed under mean assumptions — implicitly penalizing production
variance without simulating or looking ahead.

This is faithful to Powell's CFA definition: "modify the parameters of the
optimization model to produce better decisions under uncertainty." The
parameters $\theta_b$ and $\theta_e$ are tuned offline (e.g., grid search over
simulation episodes).

**Parameters:**

- $\theta_b \in \mathbb{Z}_{\geq 0}$ — planning buffer.
- $\theta_e \in (0, 1]$ — efficiency discount (lower = more conservative).

**Planning:**

$$
Q_d^{\text{CFA}} = \max\!\left(0,\; \delta_d - I_{d,0}\right) + \theta_b
$$

**Control:**

Solve the single-step modified-cost optimization:

$$
u_{d,\tau}^{\text{CFA}} = \arg\min_{0 \leq u \leq N} u
\quad \text{s.t.} \quad u \cdot \theta_e \cdot \mu \geq
\frac{\max(0,\; Q_d - P_{d,\tau})}{\max(1,\; \tau^{\text{rem}}_{d,\tau})}
$$

Closed-form solution:

$$
u_{d,\tau}^{\text{CFA}} = \mathrm{clamp}\!\left(
\left\lceil
\frac{\max(0,\; Q_d - P_{d,\tau})}{\max(1,\; \tau^{\text{rem}}_{d,\tau}) \cdot \theta_e \cdot \mu}
\right\rceil,\; 0,\; N\right)
$$

**Why this is CFA, not PFA:** Although the closed-form solution *looks* like a
formula, it is derived from solving an optimization problem with a modified
constraint ($\theta_e \cdot \mu$ instead of $\mu$). The parameter $\theta_e$
modifies the optimization's feasibility region — it does not appear in the
true system dynamics. A PFA would use $\mu$ directly without modification.

---

### Cost Function Approximation — Lookahead (CFA-Lookahead)

A **hybrid CFA / deterministic DLA** that explicitly looks ahead over a
configurable horizon $H$ and optimizes machine activation to cover the
lookahead window, replacing stochastic output with its mean $\mu$.

This is labeled as a hybrid because:

- The multi-step lookahead is DLA-like (explicit optimization over future).
- The parameters $H$ and $\theta_b$ parameterize the optimization structure
  (CFA spirit).
- No Monte Carlo sampling is used (deterministic approximation).

**Parameters:**

- $\theta_b \in \mathbb{Z}_{\geq 0}$ — planning buffer.
- $H \in \mathbb{Z}_{>0}$ — optimization horizon in intra-day time steps.

**Planning:**

The planner looks $\lceil H / T \rceil$ days ahead. Let $D_H$ be the total
demand in the horizon window and $\bar{C}$ the daily mean capacity:

$$
Q_d^{\text{CFA-LA}} = \min\!\left(\bar{C},\; \max(0,\; D_H - I_{d,0}) + \theta_b\right)
$$

**Control:**

At step $\tau$, solve the deterministic receding-horizon problem for the
minimal constant machine activation over the remaining horizon:

$$
u_{d,\tau}^{\text{CFA-LA}} = \mathrm{clamp}\!\left(
\left\lceil
\frac{\max\!\left(0,\; \max(Q_d - P_{d,\tau},\; R_{d,\tau}) - I_{d,\tau}\right)}
{\min(H, \tau^{\text{rem}}_{d,\tau}) \cdot \mu}
\right\rceil,\; 0,\; N\right)
$$

Closed-form solution to:

$$
\min_{u} \; u \quad \text{s.t.} \quad u \cdot \tilde{T} \cdot \mu \geq \text{net\_need}, \quad 0 \leq u \leq N
$$

where $\tilde{T} = \min(H, \tau^{\text{rem}})$ is the effective horizon.

---

### Direct Lookahead Approximation (DLA)

The DLA performs **explicit stochastic lookahead**: it enumerates candidate
actions, simulates their consequences over a tunable horizon $H$ using the
true stochastic model, and selects the action minimizing expected multi-step
cost.

For steps beyond the first in each rollout, a greedy rate-spread inner policy
(heuristic) is used to simulate plausible future decisions. This is a standard
"rollout policy" design: optimize the *first* action, simulate the rest with
a cheap heuristic.

**Parameters:**

- $K \in \mathbb{Z}_{>0}$ — Monte Carlo samples per candidate action.
- $H \in \mathbb{Z}_{>0}$ — lookahead horizon in steps (capped at remaining
  steps in the day). $H = 1$ recovers the myopic one-step DLA.
- $\theta_b \in \mathbb{Z}_{\geq 0}$ — planning buffer.

**Planning:**

$$
Q_d^{\text{DLA}} = \max\!\left(0,\; \delta_d - I_{d,0}\right) + \theta_b
$$

**Control:**

For each candidate action $u \in \{0, \ldots, N\}$, simulate $K$ rollouts of
$\tilde{H} = \min(H, \tau^{\text{rem}})$ steps:

- Step 1 uses candidate action $u$.
- Steps $2, \ldots, \tilde{H}$ use the greedy inner policy:
  $u^{\text{inner}} = \lceil (Q_d - P) / (\text{remaining\_steps} \cdot \mu) \rceil$.

Accumulate costs over all $\tilde{H}$ steps in each rollout:

$$
\hat{C}(u) = \frac{1}{K} \sum_{k=1}^{K} \sum_{h=1}^{\tilde{H}}
\left[
c_s \cdot \text{shortage}_h^{(k)}
+ c_h \cdot \text{excess\_inventory}_h^{(k)}
+ c_m \cdot u_h^{(k)}
\right]
$$

Select:

$$
u_{d,\tau}^{\text{DLA}} = \arg\min_{u \in \{0,\ldots,N\}} \hat{C}(u)
$$

**Why multi-step matters:** A myopic ($H=1$) DLA minimizes immediate cost but
cannot trade off current overproduction against future shortage risk. With
$H > 1$, the DLA can recognize that activating fewer machines *now* is
acceptable if the inner policy can recover in subsequent steps — reducing
holding costs while maintaining service level.

**Computational cost:** $O((N+1) \cdot K \cdot H)$ per step. With $N=10$,
$K=15$, $H=4$: 660 stochastic simulations per control decision.

---

## Policy Comparison

| Policy | Meta-class | Planning rule | Control rule | Tunable parameters |
|---|---|---|---|---|
| PFA | PFA | inventory-safety-stock gap | urgency rate rule | $\theta_s$ (safety stock) |
| PFA-Capacity | PFA | capacity-capped demand gap | deterministic rate spread | $\theta_b$ (planning buffer) |
| CFA | CFA | demand gap + buffer | single-step modified-cost optimization | $\theta_b$ (buffer), $\theta_e$ (efficiency discount) |
| CFA-Lookahead | Hybrid CFA/DLA | horizon demand optimization | receding-horizon min-machines | $\theta_b$, $H$ (horizon) |
| DLA | DLA | demand gap + buffer | multi-step Monte Carlo rollout | $K$ (samples), $H$ (horizon), $\theta_b$ |

### Key Distinctions (Powell's Meta-Classes)

- **PFA** maps state to decision through a fixed parametric function — no optimization is solved. The tunable parameters ($\theta_s$, $\theta_b$) are found offline via search/tuning.
- **CFA** solves a **single-period** optimization with modified parameters ($\theta_e$ discounts assumed machine output). The modified parameters are tuned offline to make greedy single-step decisions yield good multi-period performance. No lookahead is performed.
- **CFA-Lookahead** (hybrid): solves a deterministic multi-step optimization. Labeled as a hybrid because it uses explicit horizon lookahead (DLA-like) with parameterized structure (CFA-like). Not a pure CFA per Powell's definition.
- **DLA** builds an explicit (approximate) model of the future via Monte Carlo simulation and searches over candidate actions. It uses the actual stochastic model and cost parameters ($c_s$, $c_h$, $c_m$) for evaluation.

**Boundary between CFA, CFA-Lookahead, and DLA:**

| | Single-step? | Stochastic? | What's modified? |
|---|---|---|---|
| CFA | Yes | No | Parameters of the optimization ($\theta_e$) |
| CFA-Lookahead | No (horizon $H$) | No (uses $\mu$) | Horizon length and buffer |
| DLA | Tunable ($H$ steps) | Yes (samples) | Number of Monte Carlo samples, horizon |

### Why Not VFA?

Value Function Approximation (the fourth meta-class) would require learning
$\bar{V}(S)$ — a mapping from state to expected downstream cost. This is
omitted because:

1. The state space is modest enough for DLA to enumerate actions directly.
2. The pedagogical goal is to contrast PFA/CFA/DLA on a concrete problem.

---

## Design Notes

Based on review feedback (W.B. Powell):

1. **Hierarchical coupling via $Q_d$.** The production target $Q_d$ appears in
   the execution state, coupling planning and control levels. From the control
   policy's perspective, $Q_d$ is exogenous information that arrived before
   control began. The control policy is therefore not purely reactive — it is
   anchored to a prior planning decision. This is standard in hierarchical and
   rolling-horizon control architectures.

2. **Pure CFA vs. deterministic lookahead.** A pure CFA solves a *single-period*
   optimization with a parameterized cost modification (here: $\theta_e$
   discounts assumed output). The original receding-horizon policy is *not* a
   pure CFA — it explicitly optimizes over a future window $H$, making it a
   deterministic DLA. We now implement both: `CFAPolicy` (pure, single-step)
   and `CFALookaheadPolicy` (hybrid, multi-step deterministic).

3. **DLA horizon is tunable.** The lookahead horizon $H$ is now an explicit
   tunable parameter. $H = 1$ gives the classic myopic one-step DLA; larger
   values trade compute for better multi-step cost estimates. With $N+1$
   candidate actions, $K$ samples, and $H$ steps, cost per decision is
   $O((N+1) \cdot K \cdot H)$. A greedy rate-spread inner policy is used for
   steps $2, \ldots, H$ to keep rollouts tractable without full tree search.

4. **Genuine stochasticity.** The fact that demand is deterministic but machine
   output is stochastic creates *real* sequential decision-making. After $Q_d$
   is set, actual output varies step-by-step, and the control policy must adapt
   $u_{d,\tau}$ to realized production outcomes — the core of SDA.
