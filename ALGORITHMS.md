# Algorithms Reference

Detailed formulae, pseudocode, and hyperparameters for every algorithm implemented in this repo. The interactive HTML guide (`docs/index.html`) is the best place to *build intuition* for the Bellman equation step by step; this file is the companion reference for *looking things up* — what each script actually implements, and why.

## Table of contents

- [1. Value Iteration](#1-value-iteration)
- [2. Policy Iteration](#2-policy-iteration)
- [3. Tabular Q-Learning](#3-tabular-q-learning)
- [4. SARSA](#4-sarsa)
- [5. Cross-Entropy Method (CEM)](#5-cross-entropy-method-cem)
- [6. All five, side by side](#6-all-five-side-by-side)
- [References](#references)

---

## 1. Value Iteration

**Implemented in:** [`python/value_iteration_gridworld.py`](./python/value_iteration_gridworld.py)

### What it is

Every other algorithm in this repo is **model-free**: it never sees the environment's transition probabilities or reward function, only the `(state, action) -> (reward, next state)` samples it happens to experience by actually acting. Value Iteration is the opposite. It's a **model-based planning** algorithm — it's handed the full transition model up front (every possible outcome of every action, with its exact probability, computed directly from the known dynamics rather than sampled) and never takes a single real step in the environment. Instead, it repeatedly sweeps over *every* state at once, applying the **Bellman optimality equation**, until the value estimates stop changing.

**In plain English:** where Q-learning has to actually walk into the pit a few times before it learns that state is bad, Value Iteration can just look up the transition model and compute the consequence directly — no trial and error required. It's doing arithmetic on a known map of the world, not exploring an unknown one.

### The Bellman optimality equation

The value that Value Iteration converges toward, $V^*(s)$, satisfies:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s,a)\,\Big[R(s,a,s') + \gamma V^*(s')\Big]$$

Value Iteration turns this into an update rule by applying the right-hand side, using the *current* estimate $V_k$ in place of the unknown $V^*$, and repeating:

$$V_{k+1}(s) \;\leftarrow\; \max_a \sum_{s'} P(s' \mid s,a)\,\Big[R(s,a,s') + \gamma V_k(s')\Big]$$

| Symbol | Meaning |
|---|---|
| $s$ | the state being updated |
| $a$ | an action being considered |
| $P(s' \mid s,a)$ | probability of landing in $s'$, given the model — known exactly, not estimated |
| $R(s,a,s')$ | reward for that specific transition |
| $\gamma$ | discount factor |
| $\max_a(\,\cdot\,)$ | try every action, keep the best expected value |

Notice there's no learning rate $\alpha$ anywhere in this update, unlike every other algorithm in this repo. That's not an oversight — $\alpha$ exists in Q-learning and SARSA specifically to smooth out *noisy, sampled* estimates over time. Value Iteration never samples anything; $\sum_{s'} P(s'\mid s,a)[\,\cdot\,]$ is an *exact* expectation computed from the known model, so there's nothing noisy to smooth.

Compare this directly to Q-learning's update from [Section 3](#3-tabular-q-learning):

**Q-learning:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]$$
*uses ONE sampled transition, nudges the estimate a little*

**Value Iteration:**
$$V(s) \leftarrow \max_a \sum_{s'} P(s'\mid s,a)\Big[R(s,a,s') + \gamma V(s')\Big]$$
*uses EVERY possible transition, weighted exactly, replaces the estimate outright*

Q-learning's $\max_{a'} Q(s',a')$ and Value Iteration's $\max_a \sum_{s'} P(\cdot)[\,\cdot\,]$ are answering the same underlying question — "what's the best I could do from here?" — but Q-learning has to *estimate* that from limited experience, one sample at a time, while Value Iteration can *compute* it exactly, because it already has the model.

### Pseudocode

```
initialize V(s) = 0 for all states s
repeat:
    delta = 0
    for each state s:
        v_old = V(s)
        V(s) = max_a  sum_s'  P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
        delta = max(delta, |v_old - V(s)|)
until delta < theta          # theta is a small convergence threshold, e.g. 1e-4

# extract the policy once V has converged:
policy(s) = argmax_a  sum_s'  P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
```

This repo's script performs a **synchronous** sweep — every state's new value is computed from the *same* snapshot of $V$, and all of them are replaced together at the end of the sweep (rather than updating states one at a time and immediately using the newer values for later states in the same pass). This matches the classical textbook presentation and makes the "wave of value information spreading outward from the goal" visually obvious, one full sweep at a time.

### This repo's exact setup

Same environment as every other script here — the classic 4×3 "Example of Learned Q-Function" gridworld (Berkeley CS188 / Russell & Norvig, *AIMA*):

- **Grid:** 3 rows × 4 columns, one interior wall
- **Goal:** $+1$ terminal reward, **Pit:** $-1$ terminal reward
- **Living reward:** $-0.04$ per non-terminal step
- **Transition model:** $0.8$ probability of moving in the intended direction, $0.1/0.1$ split across the two perpendicular directions (bumping a wall/boundary leaves the agent in place) — this exact model is what `transitions(state, action)` in the script enumerates directly, rather than sampling from it
- $\gamma = 0.9$, convergence threshold $\theta = 10^{-4}$

Running it to convergence takes **16 sweeps** and produces the exact optimal value function and policy for this gridworld — matching, state for state, what Q-learning, SARSA, and CEM all converge toward independently through trial and error elsewhere in this repo.

### Convergence

The Bellman optimality backup is a **contraction mapping** in the max-norm, with contraction factor $\gamma$. That guarantees the sequence $V_0, V_1, V_2, \ldots$ converges to the unique fixed point $V^*$ as $k \to \infty$, regardless of how $V_0$ is initialized (this repo starts at all zeros) — and it converges *geometrically*, at rate $\gamma$ per sweep. Once $V$ has converged, the one-step-lookahead greedy policy

$$\pi(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\Big[R(s,a,s') + \gamma V(s')\Big]$$

is guaranteed optimal. In practice, the greedy policy implied by $V$ often stabilizes into its final, optimal shape well before the numeric values themselves finish converging — worth watching for directly in the interactive GUI's arrows.

---

## 2. Policy Iteration

**Implemented in:** [`python/policy_iteration_gridworld.py`](./python/policy_iteration_gridworld.py)

### What it is

Like Value Iteration, this is a **model-based planning** algorithm — same known transition model, same lack of any real interaction with the environment. But where Value Iteration folds "evaluate every action" and "take the best one" into a single combined update on every sweep, Policy Iteration keeps an **explicit policy** the whole time and alternates between two distinctly different phases:

1. **Policy Evaluation** — hold the policy fixed, and sweep $V(s)$ until it exactly matches the value of *that specific policy* (no $\max$ involved at all — just plug in whatever action the policy already prescribes).
2. **Policy Improvement** — hold $V(s)$ fixed, and for every state switch to whichever action looks best under a one-step lookahead. This is the one place a $\max$ appears in the whole algorithm.

Repeat those two phases until the policy stops changing between improvement steps.

**In plain English:** Policy Iteration asks "given what I'm currently doing, how good is that, exactly?" (evaluation), then "now that I know exactly how good my current plan is, can I do better anywhere?" (improvement) — and alternates those two questions until the answer to the second one is "no, nowhere."

### The two update rules

**Policy Evaluation** (repeated for a *fixed* policy $\pi$, until it converges):

$$V(s) \;\leftarrow\; \sum_{s'} P\big(s' \mid s, \pi(s)\big)\,\Big[R\big(s,\pi(s),s'\big) + \gamma V(s')\Big]$$

Notice: no $\max_a$ here — $\pi(s)$ already tells you exactly which action to plug in. This equation is solving the **Bellman expectation equation** for $V^\pi$, not the Bellman *optimality* equation Value Iteration solves.

**Policy Improvement** (one pass, using the now-converged $V^\pi$):

$$\pi'(s) \;\leftarrow\; \arg\max_a \sum_{s'} P(s'\mid s,a)\,\Big[R(s,a,s') + \gamma V^\pi(s')\Big]$$

This is exactly Value Iteration's update rule, minus taking the max *value* itself — Policy Improvement takes the $\arg\max$ *action* instead, and hands it to the policy.

The **Policy Improvement Theorem** guarantees this step never makes things worse: if $\pi'$ is greedy with respect to $V^\pi$, then $V^{\pi'}(s) \ge V^\pi(s)$ for every state, with equality only when $\pi$ was already optimal. That's what makes "alternate evaluate and improve" a valid strategy at all — every improvement step either strictly helps or confirms you're already done.

### Pseudocode

```
initialize V(s) = 0 for all s
initialize policy(s) arbitrarily for all s     # this repo starts every state at "UP"

repeat:
    # --- Policy Evaluation ---
    repeat:
        delta = 0
        for each state s:
            v_old = V(s)
            V(s) = sum_s'  P(s'|s, policy(s)) * [ R(s, policy(s), s') + gamma * V(s') ]
            delta = max(delta, |v_old - V(s)|)
    until delta < theta

    # --- Policy Improvement ---
    policy_stable = True
    for each state s:
        old_action = policy(s)
        policy(s) = argmax_a  sum_s'  P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
        if policy(s) != old_action:
            policy_stable = False

until policy_stable
```

### This repo's exact setup

Same environment, $\gamma = 0.9$, evaluation convergence threshold $\theta = 10^{-4}$. The script deliberately starts from a **bad, arbitrary policy** — every state initialized to `UP` — rather than something close to optimal, so the interactive GUI shows a genuine before/after rather than a policy that barely has to move.

Running it to convergence took **3 policy-improvement iterations** (compared to Value Iteration's 16 raw sweeps) — but each of those 3 iterations includes a full Policy Evaluation phase running to convergence internally, for a total of **65 evaluation sweeps** across the whole run. Both scripts converge to *exactly* the same $V$ and the same policy, which is expected: they're two different roads to the same unique optimum.

### Convergence

Because a finite MDP has only finitely many distinct deterministic policies, and each Policy Improvement step either strictly improves the policy (by the Policy Improvement Theorem above) or leaves it unchanged — which is exactly the stopping condition — Policy Iteration is guaranteed to reach the optimal policy in a **finite** number of improvement steps. That's a qualitatively different guarantee from Value Iteration's *asymptotic* convergence ($V_k$ approaches $V^*$ in the limit, but technically never exactly reaches it in finite time under the contraction-mapping argument above).

### Value Iteration vs. Policy Iteration, at a glance

| | Value Iteration | Policy Iteration |
|---|---|---|
| **Maintains** | Just $V(s)$ — no explicit policy tracked until you read one off the end | Both $V(s)$ *and* an explicit policy, updated in alternation |
| **Uses a $\max$ every sweep?** | Yes — every single update | No — only during Policy Improvement; Policy Evaluation just plugs in the current policy's action |
| **Convergence** | Asymptotic — $V$ approaches $V^*$ in the limit, at a rate governed by $\gamma$ | Finite — provably reaches the optimal policy in a bounded number of policy changes |
| **Cost per outer step** | One cheap sweep | A full inner Policy Evaluation loop run to convergence, *then* one Policy Improvement sweep — more expensive per outer iteration |
| **In this repo** | 16 sweeps to converge | 3 policy-improvement iterations, 65 total evaluation sweeps |
| **Where in this repo** | `python/value_iteration_gridworld.py` | `python/policy_iteration_gridworld.py` |

---

Everything from here on is **model-free**: unlike Value Iteration and Policy Iteration above, none of the following three algorithms are ever given the transition model or reward function. They only know what they experience by actually acting in the environment.

## 3. Tabular Q-Learning

**Implemented in:** [`python/q_learning_gridworld.py`](./python/q_learning_gridworld.py)

### What it is

Q-learning is a **model-free, off-policy, temporal-difference (TD) control** algorithm. It learns the optimal action-value function $Q^*(s,a)$ — the expected discounted return of taking action $a$ in state $s$ and acting optimally thereafter — directly from sampled transitions, without ever being told the environment's transition probabilities or reward function.

"Off-policy" means the *behavior* used to pick actions while learning (e.g. $\varepsilon$-greedy exploration) can differ from the *target* policy being learned (the greedy, optimal one) — Q-learning converges to $Q^*$ regardless of how exploration is done, as long as every state-action pair keeps getting visited.

### The update rule

For a transition $(s, a, r, s')$:

$$Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\Big[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,\Big]$$

| Symbol | Meaning |
|---|---|
| $s, a$ | current state, action taken |
| $r$ | reward received |
| $s'$ | resulting next state |
| $\alpha$ | learning rate — how much each new sample shifts the estimate |
| $\gamma$ | discount factor — how much future reward is worth today |
| $\max_{a'} Q(s',a')$ | the best value achievable from $s'$ under the *current* Q-table |
| $r + \gamma \max_{a'} Q(s',a') - Q(s,a)$ | the **TD error** — how wrong the old estimate was |

The $\max_{a'}$ term is what makes this off-policy: the target uses the greedy action at $s'$, not whatever action the exploring agent actually takes next.

### Pseudocode

```
initialize Q(s,a) = 0 for all s, a
for each episode:
    s = start state
    while s is not terminal:
        a = choose action from s (e.g. epsilon-greedy over Q(s, ·))
        take action a, observe reward r and next state s'
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        s = s'
```

### This repo's exact setup

The gridworld is the classic 4×3 "Example of Learned Q-Function" environment (Berkeley CS188 / Russell & Norvig, *AIMA*):

- **Grid:** 3 rows × 4 columns, one interior wall
- **Goal:** $+1$ terminal reward
- **Pit:** $-1$ terminal reward
- **Living reward:** $-0.04$ for every non-terminal step (this is what makes the agent prefer *short* paths, not just *any* path to $+1$)
- **Transition noise:** the classic AIMA stochastic dynamics — $0.8$ probability of moving in the intended direction, $0.1/0.1$ split across the two perpendicular directions (bumping a wall/boundary leaves the agent in place)
- $\alpha = 0.1$, $\gamma = 0.9$ — matching the worked example in the main README:

  ```
  Q((2, 0), RIGHT) <- 0.000 + 0.1 * [ -0.04 + 0.9 * 0.000 - 0.000 ]
  Q((2, 0), RIGHT) <- 0.000 + 0.1 * (-0.040)  =  -0.004
  ```

### Convergence

Given every $(s,a)$ pair is visited infinitely often (in the limit) and the learning rate satisfies the usual stochastic-approximation conditions, tabular Q-learning provably converges to $Q^*$, from which the optimal policy is simply $\pi^*(s) = \arg\max_a Q^*(s,a)$.

---

## 4. SARSA

**Implemented in:** [`python/sarsa_gridworld.py`](./python/sarsa_gridworld.py)

### What it is

SARSA ("**S**tate-**A**ction-**R**eward-**S**tate-**A**ction," after the five quantities its update rule touches) is a **model-free, on-policy, temporal-difference (TD) control** algorithm. Structurally it looks almost identical to Q-learning — both learn a $Q(s,a)$ table from single transitions using a TD update — but it learns the value of *the policy it is actually following*, exploration and all, rather than the value of the greedy policy.

"On-policy" means the *target* being learned and the *behavior* generating the data are the same policy (here, $\varepsilon$-greedy over the current $Q$). This is the one-word difference from Q-learning that changes everything about how the two algorithms behave near risk.

### The update rule

For a transition $(s, a, r, s')$, where $a'$ is the action the agent's own $\varepsilon$-greedy policy *actually selects* at $s'$ (not the greedy max):

$$Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\Big[\,r + \gamma\, Q(s',a') - Q(s,a)\,\Big]$$

Compare this directly to Q-learning's update:

**Q-learning's target:**
$$r + \gamma \max_{a'} Q(s',a')$$
*— the value of the BEST possible next action*

**SARSA's target:**
$$r + \gamma\, Q(s',a')$$
*— the value of the next action actually taken*

That single substitution — $\max_{a'} Q(s',a')$ becoming $Q(s',a')$ for a *sampled* $a'$ — is the entire difference between the two algorithms.

### Pseudocode

```
initialize Q(s,a) = 0 for all s, a
for each episode:
    s = start state
    a = choose action from s (epsilon-greedy over Q(s, ·))
    while s is not terminal:
        take action a, observe reward r and next state s'
        a' = choose action from s' (epsilon-greedy over Q(s', ·))
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        s, a = s', a'
```

Notice $a'$ is chosen *before* the update and is then actually taken next — unlike Q-learning, which re-derives a fresh greedy action from $s'$ on the following iteration and never uses $a'$ in the update at all.

### Why this matters: on-policy risk-awareness

Because SARSA's target uses the value of whatever action its own (exploring) policy is about to take, it implicitly bakes in the *cost of exploring* near dangerous states. If a wrong, non-greedy move could land the agent in a $-1$ cell, SARSA's $Q$ for the safer neighboring state reflects that risk — the classic illustration of this is the "Cliff Walking" gridworld in Sutton & Barto (Ch. 6), where Q-learning learns the optimal-but-risky path along a cliff edge while SARSA learns a longer, safer detour, because Q-learning's $\max$ target implicitly assumes it will behave greedily forever after — even though during training it's still exploring.

You can see a small version of this effect directly in this repo's gridworld: state `(1,2)` sits directly next to the pit. Under SARSA, `Q((1,2), RIGHT)` (moving toward the pit) is pulled sharply negative, because an epsilon-greedy slip really can walk the agent into $-1$ from there — the algorithm is "aware" of its own exploration risk in a way Q-learning's greedy-max target is not.

### This repo's exact setup

Same environment as the Q-learning script ($\gamma = 0.9$, living reward $-0.04$, same 4×3 layout, same stochastic transition model). Default hyperparameters:

| Hyperparameter | Default | Meaning |
|---|---|---|
| `alpha` ($\alpha$) | 0.1 | learning rate — matches `q_learning_gridworld.py` for direct comparison |
| `epsilon` ($\varepsilon$) | 0.1 | epsilon-greedy exploration rate (used for *both* behavior and the SARSA target itself) |
| `episodes` | 1000 | training length for `--headless` / "Train 1000 Episodes" |

### Convergence

SARSA converges to the optimal policy **under the same $\varepsilon$-greedy behavior policy**, provided $\varepsilon$ is decayed to zero over time (an assumption called GLIE — "Greedy in the Limit with Infinite Exploration"). With $\varepsilon$ held fixed, as it is by default in this repo's script, SARSA converges instead to the best policy *available to an agent that keeps exploring at that fixed rate* — which is exactly what makes it a more risk-aware learner in practice.

---

## 5. Cross-Entropy Method (CEM)

**Implemented in:** [`python/cross_entropy_method.py`](./python/cross_entropy_method.py)

### What it is

CEM is a **derivative-free, population-based, episodic policy search** method. It was originally developed by Reuven Rubinstein for rare-event probability estimation and combinatorial optimization, and later repurposed as a simple, surprisingly effective reinforcement-learning baseline.

The key conceptual difference from Q-learning: **CEM never builds a value function and never looks at individual transitions.** It only ever asks "how much total reward did this whole episode get?", then reshapes a search distribution toward whatever parameters produced the best full episodes. It's a black box optimizer wrapped around the environment.

### Policy parameterization (this repo)

The policy is a table of per-state action *preferences* (logits), $\theta$ with shape $(|S|, |A|)$, converted into a probability distribution with softmax:

$$\pi_\theta(a \mid s) = \operatorname{softmax}\big(\theta[s,:]\big)_a = \frac{\exp\big(\theta[s,a]\big)}{\sum_{a'} \exp\big(\theta[s,a']\big)}$$

This plays the same structural role as the Q-table in the Q-learning demo — one number per $(state, action)$ pair — but it's optimized completely differently: never bootstrapped from a Bellman backup, only ever nudged toward whatever full episodes scored highest.

### The algorithm

Let $\theta$ be flattened into a single vector of length $d = |S| \times |A|$, and let $R(\theta)$ denote the average total discounted return of rolling out $\pi_\theta$:

$$R(\theta) = \mathbb{E}\Big[\, \sum_t \gamma^t r_t \;\Big|\; \text{actions sampled from } \pi_\theta \,\Big]$$

CEM maintains a Gaussian search distribution $\mathcal{N}(\mu, \sigma^2)$ over $\theta$ and, each generation:

1. **Sample** a population of $N$ candidates: $\theta_i \sim \mathcal{N}\big(\mu, \operatorname{diag}(\sigma^2)\big)$, for $i = 1, \ldots, N$
2. **Evaluate** each candidate by rolling out $\pi_{\theta_i}$ (one or more episodes, averaged for a less noisy score) and recording its return $R(\theta_i)$
3. **Select the elites** — the top $\lceil N\rho \rceil$ candidates by return, where $\rho$ is the elite fraction (e.g. $0.2$ → top 20%)
4. **Refit** the search distribution to just the elites:

   $$\mu \leftarrow \operatorname{mean}(\text{elite } \theta\text{'s}) \qquad \sigma \leftarrow \operatorname{std}(\text{elite } \theta\text{'s}) + \text{noise floor}$$

   (the decaying "noise floor" term is `noise_floor(generation)` in the code — see [why it's there](#why-the-extra-noise-floor-term) below)

5. Repeat until the population converges (returns stop improving).

The final policy is the **greedy** one implied by the converged mean: $\pi(s) = \arg\max_a \theta[s,a]$ where $\theta = \operatorname{reshape}(\mu)$.

### Why the extra "noise floor" term?

Plain CEM can converge prematurely: once a handful of similar elites are found, their variance collapses toward zero within a few generations, and the search distribution stops exploring — often before finding the actual optimum. Adding a small amount of *extra* Gaussian noise back into $\sigma$ each generation, decaying it over time, keeps exploration alive for longer. This is the same trick used in Szita & Lörincz's well-known "Learning Tetris Using the Noisy Cross-Entropy Method" (2006) — see [References](#references).

### This repo's exact setup

`python/cross_entropy_method.py` uses the **same environment, discount, and living reward** as the Q-learning script ($\gamma = 0.9$, living reward $-0.04$, same 4×3 layout) so the two methods are directly comparable. Default hyperparameters:

| Hyperparameter | Default | Meaning |
|---|---|---|
| `population` ($N$) | 60 | candidates sampled per generation |
| `elite-frac` ($\rho$) | 0.2 | fraction kept as elites (top 12 of 60) |
| `episodes` | 8 | rollouts averaged per candidate, per generation (reduces evaluation noise) |
| `init-std` | 1.0 | initial spread of the search distribution |
| `noise-floor` | 0.5 | extra exploration noise added to $\sigma$, linearly decayed to 0 over 100 generations |
| `generations` | 100 | training length |

### Pseudocode

```
initialize mu = 0, sigma = init_std, for a d-dimensional theta
for each generation:
    sample N thetas from N(mu, sigma^2)
    for each theta_i: R_i = average return of rolling out pi_theta_i
    elites = top rho-fraction of thetas by R_i
    mu    = mean(elites)
    sigma = std(elites) + decaying noise floor
return greedy policy from final mu
```

---

## 6. All five, side by side

### The big picture: planning vs. learning

| | Value Iteration | Policy Iteration | Q-Learning | SARSA | Cross-Entropy Method |
|---|---|---|---|---|---|
| **Category** | Model-based planning | Model-based planning | Model-free learning | Model-free learning | Model-free learning |
| **Needs the transition model $P(s'\mid s,a)$?** | Yes | Yes | No | No | No |
| **Ever takes a real step in the environment?** | No | No | Yes | Yes | Yes |
| **Has a learning rate $\alpha$?** | No — exact expectation, nothing to smooth | No — same reason | Yes | Yes | No (uses a search distribution instead) |
| **What it ultimately produces** | $V^*(s)$ (policy read off at the end) | An explicit policy, improved in alternation | $Q^*(s,a)$ | $Q(s,a)$ for the policy actually followed | Policy parameters $\theta$, directly |
| **Where in this repo** | `python/value_iteration_gridworld.py` | `python/policy_iteration_gridworld.py` | `python/q_learning_gridworld.py` | `python/sarsa_gridworld.py` | `python/cross_entropy_method.py` |

Value Iteration and Policy Iteration both converge to *exactly* the same optimum in this repo's gridworld — and so, given enough training, do Q-learning, SARSA, and CEM, purely from experience, with no access to the model the first two are handed for free. Watching all five scripts arrive at the same answer by such different routes is the point of having all of them side by side.

### Comparing the three learning methods in detail

| | Tabular Q-Learning | SARSA | Cross-Entropy Method |
|---|---|---|---|
| **Learns** | A value function, $Q(s,a)$ | A value function, $Q(s,a)$ | Policy parameters $\theta$, directly |
| **Update source** | Every single transition (TD, bootstrapped) | Every single transition (TD, bootstrapped) | Every full episode return (Monte Carlo, population-based) |
| **Update target** | $r + \gamma \max_{a'} Q(s',a')$ | $r + \gamma\, Q(s',a')$ for the actually-chosen $a'$ | Whole-episode discounted return $R(\theta)$ |
| **On/off policy** | **Off**-policy — learns $Q^*$ no matter what exploration policy is used | **On**-policy — learns the value of the policy it's actually following, exploration included | Population-level — elites always come from the *current* search distribution |
| **Needs a value function?** | Yes — it *is* the value function | Yes — it *is* the value function | No — never estimated |
| **Gradient-based?** | No (table lookup + max) | No (table lookup, no max) | No (gradient-free, black-box) |
| **Sample efficiency** | Higher — one update per environment step | Higher — one update per environment step | Lower — needs many full rollouts per generation before any update happens |
| **Exploration mechanism** | Explicit (e.g. $\varepsilon$-greedy), and *not* reflected in what's learned | Explicit ($\varepsilon$-greedy), and **is** reflected in what's learned — risk from exploring gets baked into $Q$ | Implicit, built into the population's variance $\sigma$ |
| **Behavior near risk** | Learns the truly optimal (possibly risky) policy, assuming greedy behavior after training | Learns a more cautious policy if exploration itself is dangerous (see the "Cliff Walking" discussion above) | Depends entirely on the stochastic policy used during rollouts, same as SARSA in spirit |
| **What happens to rarely-visited states** | $Q$ stays near its initial value until visited | $Q$ stays near its initial value until visited | $\theta$ row stays near its initial value — may show an arbitrary "policy" that was simply never exercised |
| **Natural next step in this repo's roadmap** | → DQN (swap the table for a neural network) | → forms the "on-policy" half of the DQN/Actor-Critic family later in the roadmap | → Policy Gradient methods (REINFORCE), which replace CEM's population search with an actual gradient estimate of $\mathbb{E}[R]$ w.r.t. $\theta$ |
| **Where in this repo** | `python/q_learning_gridworld.py` | `python/sarsa_gridworld.py` | `python/cross_entropy_method.py` |

All three are **model-free** — none of them is ever given the transition probabilities or reward function; they only interact with the environment through `(state, action) -> (reward, next state)`. That's what distinguishes all three from Value Iteration and Policy Iteration above, which require a fully known MDP model up front.

Q-learning and SARSA are also both **tabular TD-control** methods — same update shape, same one-transition-at-a-time learning — which makes them the cleanest possible pair for seeing on-policy vs. off-policy side by side. CEM sits in a different family entirely: **episodic, gradient-free policy search**, evaluating whole policies rather than individual actions.

---

## References

- Bellman, R. (1957). *Dynamic Programming.* Princeton University Press — the original formulation of the optimality equation that both Value Iteration and Policy Iteration are built on.
- Howard, R.A. (1960). *Dynamic Programming and Markov Processes.* MIT Press — introduced Policy Iteration.
- Rummery, G.A. & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems.* Technical Report CUED/F-INFENG/TR 166, Cambridge University — the original SARSA algorithm (named later by Sutton & Barto).
- Rubinstein, R.Y. (1999). *The Cross-Entropy Method for Combinatorial and Continuous Optimization.*
- Szita, I. & Lörincz, A. (2006). *Learning Tetris Using the Noisy Cross-Entropy Method.* Neural Computation. — source of the decaying "noise floor" trick used here.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 4 covers Value Iteration and Policy Iteration (Dynamic Programming); Chapter 6 covers both Q-learning and SARSA, including the Cliff Walking on-policy-vs-off-policy example referenced above; the broader policy-search family (of which CEM is a gradient-free member) is discussed in Chapter 13.
- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* — source of the 4×3 gridworld environment used throughout this repo.
- Dan Klein & Pieter Abbeel, Berkeley CS188 course slides — popularized the "Example of Learned Q-Function" version of the gridworld used here.