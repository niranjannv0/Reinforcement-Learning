# Algorithms Reference

Detailed formulae, pseudocode, and hyperparameters for every algorithm implemented in this repo. The interactive HTML guide (`docs/index.html`) is the best place to *build intuition* for the Bellman equation step by step; this file is the companion reference for *looking things up* — what each script actually implements, and why.

## Table of contents

- [1. Tabular Q-Learning](#1-tabular-q-learning)
- [2. SARSA](#2-sarsa)
- [3. Cross-Entropy Method (CEM)](#3-cross-entropy-method-cem)
- [4. All three, side by side](#4-all-three-side-by-side)
- [References](#references)

---

## 1. Tabular Q-Learning

**Implemented in:** [`python/q_learning_gridworld.py`](./python/q_learning_gridworld.py)

### What it is

Q-learning is a **model-free, off-policy, temporal-difference (TD) control** algorithm. It learns the optimal action-value function `Q*(s,a)` — the expected discounted return of taking action `a` in state `s` and acting optimally thereafter — directly from sampled transitions, without ever being told the environment's transition probabilities or reward function.

"Off-policy" means the *behavior* used to pick actions while learning (e.g. ε-greedy exploration) can differ from the *target* policy being learned (the greedy, optimal one) — Q-learning converges to `Q*` regardless of how exploration is done, as long as every state-action pair keeps getting visited.

### The update rule

For a transition `(s, a, r, s')`:

```
Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
```

| Symbol | Meaning |
|---|---|
| `s, a` | current state, action taken |
| `r` | reward received |
| `s'` | resulting next state |
| `alpha` | learning rate — how much each new sample shifts the estimate |
| `gamma` | discount factor — how much future reward is worth today |
| `max_a' Q(s',a')` | the best value achievable from `s'` under the *current* Q-table |
| `r + gamma * max_a' Q(s',a') - Q(s,a)` | the **TD error** — how wrong the old estimate was |

The `max_a'` term is what makes this off-policy: the target uses the greedy action at `s'`, not whatever action the exploring agent actually takes next.

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
- **Goal:** `+1` terminal reward
- **Pit:** `-1` terminal reward
- **Living reward:** `-0.04` for every non-terminal step (this is what makes the agent prefer *short* paths, not just *any* path to `+1`)
- **Transition noise:** the classic AIMA stochastic dynamics — 0.8 probability of moving in the intended direction, 0.1/0.1 split across the two perpendicular directions (bumping a wall/boundary leaves the agent in place)
- **`alpha = 0.1`, `gamma = 0.9`** — matching the worked example in the main README:

  ```
  Q((2, 0), RIGHT) <- 0.000 + 0.1 * [ -0.04 + 0.9 * 0.000 - 0.000 ]
  Q((2, 0), RIGHT) <- 0.000 + 0.1 * (-0.040)  =  -0.004
  ```

### Convergence

Given every `(s,a)` pair is visited infinitely often (in the limit) and the learning rate satisfies the usual stochastic-approximation conditions, tabular Q-learning provably converges to `Q*`, from which the optimal policy is simply `pi*(s) = argmax_a Q*(s,a)`.

---

## 2. SARSA

**Implemented in:** [`python/sarsa_gridworld.py`](./python/sarsa_gridworld.py)

### What it is

SARSA ("**S**tate-**A**ction-**R**eward-**S**tate-**A**ction," after the five quantities its update rule touches) is a **model-free, on-policy, temporal-difference (TD) control** algorithm. Structurally it looks almost identical to Q-learning — both learn a `Q(s,a)` table from single transitions using a TD update — but it learns the value of *the policy it is actually following*, exploration and all, rather than the value of the greedy policy.

"On-policy" means the *target* being learned and the *behavior* generating the data are the same policy (here, epsilon-greedy over the current `Q`). This is the one-word difference from Q-learning that changes everything about how the two algorithms behave near risk.

### The update rule

For a transition `(s, a, r, s')`, where `a'` is the action the agent's own epsilon-greedy policy *actually selects* at `s'` (not the greedy max):

```
Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * Q(s',a')  -  Q(s,a) ]
```

Compare this directly to Q-learning's update:

```
Q-learning:  target = r + gamma * max_a' Q(s',a')     <- value of the BEST possible next action
SARSA:       target = r + gamma * Q(s',a')             <- value of the next action actually taken
```

That single substitution — `max_a' Q(s',a')` becoming `Q(s',a')` for a *sampled* `a'` — is the entire difference between the two algorithms.

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

Notice `a'` is chosen *before* the update and is then actually taken next — unlike Q-learning, which re-derives a fresh greedy action from `s'` on the following iteration and never uses `a'` in the update at all.

### Why this matters: on-policy risk-awareness

Because SARSA's target uses the value of whatever action its own (exploring) policy is about to take, it implicitly bakes in the *cost of exploring* near dangerous states. If a wrong, non-greedy move could land the agent in a `-1` cell, SARSA's `Q` for the safer neighboring state reflects that risk — the classic illustration of this is the "Cliff Walking" gridworld in Sutton & Barto (Ch. 6), where Q-learning learns the optimal-but-risky path along a cliff edge while SARSA learns a longer, safer detour, because Q-learning's `max` target implicitly assumes it will behave greedily forever after — even though during training it's still exploring.

You can see a small version of this effect directly in this repo's gridworld: state `(1,2)` sits directly next to the pit. Under SARSA, `Q((1,2), RIGHT)` (moving toward the pit) is pulled sharply negative, because an epsilon-greedy slip really can walk the agent into `-1` from there — the algorithm is "aware" of its own exploration risk in a way Q-learning's greedy-max target is not.

### This repo's exact setup

Same environment as the Q-learning script (`gamma = 0.9`, living reward `-0.04`, same 4×3 layout, same stochastic transition model). Default hyperparameters:

| Hyperparameter | Default | Meaning |
|---|---|---|
| `alpha` | 0.1 | learning rate — matches `q_learning_gridworld.py` for direct comparison |
| `epsilon` | 0.1 | epsilon-greedy exploration rate (used for *both* behavior and the SARSA target itself) |
| `episodes` | 1000 | training length for `--headless` / "Train 1000 Episodes" |

### Convergence

SARSA converges to the optimal policy **under the same epsilon-greedy behavior policy**, provided epsilon is decayed to zero over time (an assumption called GLIE — "Greedy in the Limit with Infinite Exploration"). With epsilon held fixed, as it is by default in this repo's script, SARSA converges instead to the best policy *available to an agent that keeps exploring at that fixed rate* — which is exactly what makes it a more risk-aware learner in practice.

---

## 3. Cross-Entropy Method (CEM)

**Implemented in:** [`python/cross_entropy_method.py`](./python/cross_entropy_method.py)

### What it is

CEM is a **derivative-free, population-based, episodic policy search** method. It was originally developed by Reuven Rubinstein for rare-event probability estimation and combinatorial optimization, and later repurposed as a simple, surprisingly effective reinforcement-learning baseline.

The key conceptual difference from Q-learning: **CEM never builds a value function and never looks at individual transitions.** It only ever asks "how much total reward did this whole episode get?", then reshapes a search distribution toward whatever parameters produced the best full episodes. It's a black box optimizer wrapped around the environment.

### Policy parameterization (this repo)

The policy is a table of per-state action *preferences* (logits), `theta` with shape `(|S|, |A|)`, converted into a probability distribution with softmax:

```
pi_theta(a | s) = softmax( theta[s, :] )_a
                = exp(theta[s,a]) / sum_a' exp(theta[s,a'])
```

This plays the same structural role as the Q-table in the Q-learning demo — one number per `(state, action)` pair — but it's optimized completely differently: never bootstrapped from a Bellman backup, only ever nudged toward whatever full episodes scored highest.

### The algorithm

Let `theta` be flattened into a single vector of length `d = |S| x |A|`, and let `R(theta)` denote the average total discounted return of rolling out `pi_theta`:

```
R(theta) = E[ sum_t  gamma^t * r_t  |  actions sampled from pi_theta ]
```

CEM maintains a Gaussian search distribution `N(mu, sigma^2)` over `theta` and, each generation:

1. **Sample** a population of `N` candidates: `theta_i ~ N(mu, diag(sigma^2))`, for `i = 1..N`
2. **Evaluate** each candidate by rolling out `pi_theta_i` (one or more episodes, averaged for a less noisy score) and recording its return `R(theta_i)`
3. **Select the elites** — the top `ceil(N * rho)` candidates by return, where `rho` is the elite fraction (e.g. `0.2` → top 20%)
4. **Refit** the search distribution to just the elites:

   ```
   mu    <-  mean(elite thetas)
   sigma <-  std(elite thetas)  +  noise_floor(generation)
   ```

5. Repeat until the population converges (returns stop improving).

The final policy is the **greedy** one implied by the converged mean: `pi(s) = argmax_a theta[s,a]` where `theta = reshape(mu)`.

### Why the extra "noise floor" term?

Plain CEM can converge prematurely: once a handful of similar elites are found, their variance collapses toward zero within a few generations, and the search distribution stops exploring — often before finding the actual optimum. Adding a small amount of *extra* Gaussian noise back into `sigma` each generation, decaying it over time, keeps exploration alive for longer. This is the same trick used in Szita & Lörincz's well-known "Learning Tetris Using the Noisy Cross-Entropy Method" (2006) — see [References](#references).

### This repo's exact setup

`python/cross_entropy_method.py` uses the **same environment, discount, and living reward** as the Q-learning script (`gamma = 0.9`, living reward `-0.04`, same 4×3 layout) so the two methods are directly comparable. Default hyperparameters:

| Hyperparameter | Default | Meaning |
|---|---|---|
| `population` (`N`) | 60 | candidates sampled per generation |
| `elite-frac` (`rho`) | 0.2 | fraction kept as elites (top 12 of 60) |
| `episodes` | 8 | rollouts averaged per candidate, per generation (reduces evaluation noise) |
| `init-std` | 1.0 | initial spread of the search distribution |
| `noise-floor` | 0.5 | extra exploration noise added to `sigma`, linearly decayed to 0 over 100 generations |
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

## 4. All three, side by side

| | Tabular Q-Learning | SARSA | Cross-Entropy Method |
|---|---|---|---|
| **Learns** | A value function, `Q(s,a)` | A value function, `Q(s,a)` | Policy parameters `theta`, directly |
| **Update source** | Every single transition (TD, bootstrapped) | Every single transition (TD, bootstrapped) | Every full episode return (Monte Carlo, population-based) |
| **Update target** | `r + gamma * max_a' Q(s',a')` | `r + gamma * Q(s',a')` for the actually-chosen `a'` | Whole-episode discounted return `R(theta)` |
| **On/off policy** | **Off**-policy — learns `Q*` no matter what exploration policy is used | **On**-policy — learns the value of the policy it's actually following, exploration included | Population-level — elites always come from the *current* search distribution |
| **Needs a value function?** | Yes — it *is* the value function | Yes — it *is* the value function | No — never estimated |
| **Gradient-based?** | No (table lookup + max) | No (table lookup, no max) | No (gradient-free, black-box) |
| **Sample efficiency** | Higher — one update per environment step | Higher — one update per environment step | Lower — needs many full rollouts per generation before any update happens |
| **Exploration mechanism** | Explicit (e.g. epsilon-greedy), and *not* reflected in what's learned | Explicit (epsilon-greedy), and **is** reflected in what's learned — risk from exploring gets baked into `Q` | Implicit, built into the population's variance `sigma` |
| **Behavior near risk** | Learns the truly optimal (possibly risky) policy, assuming greedy behavior after training | Learns a more cautious policy if exploration itself is dangerous (see the "Cliff Walking" discussion above) | Depends entirely on the stochastic policy used during rollouts, same as SARSA in spirit |
| **What happens to rarely-visited states** | `Q` stays near its initial value until visited | `Q` stays near its initial value until visited | `theta` row stays near its initial value — may show an arbitrary "policy" that was simply never exercised |
| **Natural next step in this repo's roadmap** | → DQN (swap the table for a neural network) | → forms the "on-policy" half of the DQN/Actor-Critic family later in the roadmap | → Policy Gradient methods (REINFORCE), which replace CEM's population search with an actual gradient estimate of `E[R]` w.r.t. `theta` |
| **Where in this repo** | `python/q_learning_gridworld.py` | `python/sarsa_gridworld.py` | `python/cross_entropy_method.py` |

All three are **model-free** — none of them is ever given the transition probabilities or reward function; they only interact with the environment through `(state, action) -> (reward, next state)`. That's what distinguishes all three from Value/Q/Policy *Iteration* (covered in `docs/index.html`), which require a fully known MDP model up front.

Q-learning and SARSA are also both **tabular TD-control** methods — same update shape, same one-transition-at-a-time learning — which makes them the cleanest possible pair for seeing on-policy vs. off-policy side by side. CEM sits in a different family entirely: **episodic, gradient-free policy search**, evaluating whole policies rather than individual actions.

---

## References

- Rummery, G.A. & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems.* Technical Report CUED/F-INFENG/TR 166, Cambridge University — the original SARSA algorithm (named later by Sutton & Barto).
- Rubinstein, R.Y. (1999). *The Cross-Entropy Method for Combinatorial and Continuous Optimization.*
- Szita, I. & Lörincz, A. (2006). *Learning Tetris Using the Noisy Cross-Entropy Method.* Neural Computation. — source of the decaying "noise floor" trick used here.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 6 covers both Q-learning and SARSA, including the Cliff Walking on-policy-vs-off-policy example referenced above; the broader policy-search family (of which CEM is a gradient-free member) is discussed in Chapter 13.
- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* — source of the 4×3 gridworld environment used throughout this repo.
- Dan Klein & Pieter Abbeel, Berkeley CS188 course slides — popularized the "Example of Learned Q-Function" version of the gridworld used here.
