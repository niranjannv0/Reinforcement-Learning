# Algorithms Reference

Detailed formulae, pseudocode, and hyperparameters for every algorithm implemented in this repo. The interactive HTML guide (`docs/index.html`) is the best place to *build intuition* for the Bellman equation step by step; this file is the companion reference for *looking things up* — what each script actually implements, and why.

New to reinforcement learning? Start with [Foundations](#foundations) below — it defines every symbol (`Q(s,a)`, `gamma`, `pi`, ...) used in the rest of this document from scratch. Already comfortable with MDPs and Q-values? Skip straight to [Section 1](#1-tabular-q-learning).

## Table of contents

- [Foundations](#foundations)
- [1. Tabular Q-Learning](#1-tabular-q-learning)
- [2. SARSA](#2-sarsa)
- [3. Cross-Entropy Method (CEM)](#3-cross-entropy-method-cem)
- [4. All three, side by side](#4-all-three-side-by-side)
- [References](#references)

---

## Foundations

### The building blocks of an MDP

An RL agent interacts with an environment modeled as a **Markov Decision Process (MDP)**. At every timestep, the agent is in some **state**, takes an **action**, and the environment hands back a **reward** and a new state. "Markov" just means the future only depends on the *current* state — not on the history of how the agent got there.

| Term | Meaning | In this repo's gridworld |
|---|---|---|
| **State**, `s` | Everything the agent needs to know to decide what to do next | A grid cell, e.g. `(2, 0)` |
| **Action**, `a` | A choice the agent can make | `UP`, `DOWN`, `LEFT`, `RIGHT` |
| **Reward**, `r` | A number the environment hands back after each action | `-0.04` per step, `+1` at the goal, `-1` in the pit |
| **Episode** | One full run from a start state to a terminal state | Start at `(2,0)`, ends at the goal or the pit |
| **Discount factor**, `gamma` (`0 <= gamma <= 1`) | How much a reward *one step in the future* is worth *today* | `0.9` — a reward next step is worth 90% of the same reward right now |
| **Return**, `G_t` | Total discounted reward from time `t` onward: `G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...` | Reaching the goal in 4 steps returns more than reaching it in 8, because of the `-0.04` per-step cost and the discount |
| **Policy**, `pi` | The agent's strategy: a rule for picking an action given a state (can be deterministic or random) | A *greedy* policy always picks `argmax_a Q(s,a)`; an *epsilon-greedy* policy usually does, but sometimes picks randomly |

### Value functions: V(s) and Q(s,a)

Two questions come up constantly in RL, and this repo's algorithms all revolve around answering one or the other of them:

> **"How good is it to *be* in this state?"** → the **state-value function**:
> ```
> V_pi(s) = E[ G_t | s_t = s, following policy pi ]
> ```
>
> **"How good is it to take *this specific action* in this state?"** → the **action-value function**, better known as the **Q-value**:
> ```
> Q_pi(s,a) = E[ G_t | s_t = s, a_t = a, following policy pi from then on ]
> ```

In plain English: `Q(s,a)` is "if I'm standing in state `s` and I take action `a` right now, how much total reward can I expect from here on, assuming I act well afterward?" `V(s)` is the same question but *without* fixing a first action — it's the value averaged over whatever the policy would actually do.

The two are directly related:

```
V_pi(s)    =  sum_a  pi(a|s) * Q_pi(s,a)      <- average Q over the actions the policy would take
Q_pi(s,a)  =  E[ r + gamma * V_pi(s') ]        <- one step of real reward, then the value of wherever you land
```

Once you have a good estimate of `Q(s,a)`, getting a good policy out of it is trivial — **just take the action with the highest Q-value**: `pi(s) = argmax_a Q(s,a)`. That single move — "look up every action's Q-value for the current state, take the biggest one" — is called acting **greedily**, and it's the engine behind Q-learning and SARSA below.

Q-learning and SARSA both work by estimating `Q(s,a)` directly, one table entry at a time. The Cross-Entropy Method takes a totally different approach: it never computes a `Q(s,a)` or `V(s)` at all — it searches for a good *policy* directly, judging each candidate only by the total reward its whole episode earned.

### Exploitation vs. Exploration

This is the single most important tension in reinforcement learning, and every algorithm in this repo resolves it differently:

- **Exploitation** = use what you currently believe is best. Look at your current `Q(s,a)` estimates and take `argmax_a Q(s,a)`. It's how you cash in on what you've already learned.
- **Exploration** = deliberately try something *other* than your current best guess, because that estimate might still be wrong — especially early on, when most of the table is untouched and sitting at its initial value. The only way to find out if an untested action is actually better is to try it.

An agent that only exploits from an untrained Q-table gets stuck immediately: if the very first action tried from a state happens to look okay, a purely greedy agent repeats it forever and never discovers that a different action was actually better. An agent that only explores never uses anything it learns. Every algorithm below needs *some* balance of the two — but they strike that balance in very different places:

| | How it **exploits** | How it **explores** |
|---|---|---|
| **Q-Learning** | `argmax_a Q(s,a)` — both to choose actions, *and* baked directly into its update target | Bolted on separately, via epsilon-greedy action selection |
| **SARSA** | `argmax_a Q(s,a)` — to choose actions, most of the time | Also epsilon-greedy — but unlike Q-learning, the random exploratory action can leak directly into what gets *learned*, not just how it *behaves* (see [Section 2](#2-sarsa)) |
| **Cross-Entropy Method** | Keeping only the elite (best-scoring) candidate policies each generation, and narrowing the search toward them | Built into the algorithm itself — sampling a whole *population* of different candidate policies every generation. No epsilon-greedy, no per-action randomness at all (see [Section 3](#3-cross-entropy-method-cem)) |

The standard way to balance exploitation and exploration for value-based methods like Q-learning and SARSA is **epsilon-greedy** action selection:

```
with probability (1 - epsilon):   take argmax_a Q(s,a)          <- exploit
with probability epsilon:         take a uniformly random action <- explore
```

`epsilon` is a small number like `0.1`, read as "act greedily 90% of the time, and try something random 10% of the time." This is exactly what "taking the max of the Q-values" refers to below, in both Q-learning and SARSA. What actually separates the two algorithms is subtler than that, though — it's not *whether* they take a max, but *where*: does the max only decide how the agent behaves, or does it also decide what gets learned? Sections [1](#1-tabular-q-learning) and [2](#2-sarsa) walk through exactly that difference.

The Cross-Entropy Method never uses epsilon-greedy, because it never compares individual Q-values in the first place — there's no `Q(s,a)` table to take a max over. Its exploration happens once per *generation*, over whole candidate policies at once, by sampling from a search distribution. [Section 3](#3-cross-entropy-method-cem) covers exactly how.

---

## 1. Tabular Q-Learning

**Implemented in:** [`python/q_learning_gridworld.py`](./python/q_learning_gridworld.py)

### What it is

Q-learning is a **model-free, off-policy, temporal-difference (TD) control** algorithm. It learns the optimal action-value function `Q*(s,a)` directly from sampled transitions — it never needs to be told the environment's transition probabilities or reward function up front, it just learns by trying things and observing what happens.

**In plain English:** the agent wanders around, and after every single step it nudges one number in its Q-table — the entry for the state and action it just took — a little closer to "reward I actually got, plus the best value I currently think is available from wherever I ended up."

"Off-policy" means the *behavior* used to pick actions while learning (e.g. epsilon-greedy exploration) can differ from the *target* policy being learned (the purely greedy one) — Q-learning converges to `Q*` regardless of how exploration is done, as long as every state-action pair keeps getting visited. Section [Foundations](#exploitation-vs-exploration) above covers exploitation/exploration in general; the next two subsections show exactly how Q-learning uses the "take the max" idea.

### The update rule

For a transition `(s, a, r, s')` — meaning: the agent was in state `s`, took action `a`, received reward `r`, and landed in state `s'`:

```
Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
```

| Symbol | Meaning |
|---|---|
| `s, a` | current state, action taken |
| `r` | reward received |
| `s'` | resulting next state |
| `alpha` | learning rate — how much each new sample shifts the estimate (a number between 0 and 1) |
| `gamma` | discount factor — how much future reward is worth today |
| `max_a' Q(s',a')` | the best value achievable from `s'`, according to the *current* Q-table |
| `r + gamma * max_a' Q(s',a') - Q(s,a)` | the **TD error** — how wrong the old estimate turned out to be |

The `max_a'` term is what makes this off-policy: the update target always uses the *greedy* action at `s'`, regardless of which action the exploring agent actually takes next.

### Exploitation and exploration in Q-learning

Q-learning uses `argmax_a Q(s,a)` — "take the biggest Q-value" — in **two separate places**, and it's worth being precise about which is which:

1. **Choosing actions while training (behavior policy).** This is handled entirely *outside* the update rule above, typically with epsilon-greedy: take the greedy action `argmax_a Q(s,a)` most of the time, and a random action `epsilon` fraction of the time. This is where exploration actually happens for Q-learning.
2. **The update target itself.** Notice the update rule always uses `max_a' Q(s',a')` — the *best possible* value from `s'` — no matter which action the agent's epsilon-greedy behavior policy is actually about to take next. Even if the agent is about to wander off and explore randomly, the value it's learning for `(s,a)` assumes it will behave perfectly greedily from `s'` onward.

That gap — behavior can explore, but the *target* always assumes full exploitation — is precisely what "off-policy" means, and it's why Q-learning is sometimes described as learning the value of the greedy policy "from the sidelines," using data collected by a different (exploring) policy.

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

SARSA ("**S**tate-**A**ction-**R**eward-**S**tate-**A**ction," after the five quantities its update touches) is a **model-free, on-policy, temporal-difference (TD) control** algorithm. Structurally it looks almost identical to Q-learning — both learn a `Q(s,a)` table from single transitions using a TD update — but it learns the value of *the policy it is actually following*, exploration and all, rather than the value of the purely greedy policy.

**In plain English:** SARSA does the same "nudge one Q-table entry a bit" update as Q-learning, but instead of asking "what's the *best* thing I could do from here?" it asks "what am I *actually* about to do from here?" — and uses that answer, greedy or not, as part of the update.

"On-policy" means the *target* being learned and the *behavior* generating the data are literally the same policy (here, epsilon-greedy over the current `Q`). This one difference from Q-learning changes how the two algorithms behave around risk.

### The update rule

For a transition `(s, a, r, s')`, where `a'` is the action the agent's own epsilon-greedy policy *actually selects* at `s'` — not necessarily the greedy one:

```
Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * Q(s',a')  -  Q(s,a) ]
```

Compare this directly to Q-learning's update:

```
Q-learning:  target = r + gamma * max_a' Q(s',a')     <- value of the BEST possible next action
SARSA:       target = r + gamma * Q(s',a')             <- value of the next action actually taken
```

That single substitution — `max_a' Q(s',a')` becoming `Q(s',a')` for a *sampled* `a'` — is the entire difference between the two algorithms.

### Exploitation and exploration in SARSA

SARSA also uses `argmax_a Q(s,a)` to act, via the same epsilon-greedy scheme as Q-learning: greedy `(1 - epsilon)` of the time, random `epsilon` of the time. The behavior policy is identical between the two algorithms — **the difference is entirely in the update target**:

- **Q-learning's target** always plugs in `max_a' Q(s',a')`, no matter what the behavior policy is about to do. Exploration never shows up in what gets learned — only in what data gets collected.
- **SARSA's target** plugs in `Q(s',a')` for whichever `a'` the epsilon-greedy policy *actually just picked* for the next step. Roughly `epsilon` fraction of the time, that `a'` is a random, non-greedy action — and when that happens, SARSA's update target directly reflects the value of that (possibly bad) exploratory move.

That's the mechanical reason SARSA is described as "exploration-aware": the chance of exploring, and the consequences of exploring, are literally folded into the numbers it learns — not just into how it happened to behave while collecting data.

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
| `epsilon` | 0.1 | epsilon-greedy exploration rate (used for *both* behavior and, unlike Q-learning, the update target itself) |
| `episodes` | 1000 | training length for `--headless` / "Train 1000 Episodes" |

### Convergence

SARSA converges to the optimal policy **under the same epsilon-greedy behavior policy**, provided epsilon is decayed to zero over time (an assumption called GLIE — "Greedy in the Limit with Infinite Exploration"). With epsilon held fixed, as it is by default in this repo's script, SARSA converges instead to the best policy *available to an agent that keeps exploring at that fixed rate* — which is exactly what makes it a more risk-aware learner in practice.

---

## 3. Cross-Entropy Method (CEM)

**Implemented in:** [`python/cross_entropy_method.py`](./python/cross_entropy_method.py)

### What it is

CEM is a **derivative-free, population-based, episodic policy search** method. It was originally developed by Reuven Rubinstein for rare-event probability estimation and combinatorial optimization, and later repurposed as a simple, surprisingly effective reinforcement-learning baseline.

**In plain English:** instead of carefully tracking a Q-value for every state-action pair, CEM just tries a big batch of *entire candidate policies* at once, runs each one for a full episode, keeps whichever ones scored best, and treats those as a hint for where to look next. It's closer to "guess and check, then guess again near the best guesses" than to the step-by-step bookkeeping Q-learning and SARSA do.

The key conceptual difference from Q-learning and SARSA: **CEM never builds a value function and never looks at individual transitions.** It only ever asks "how much total reward did this whole episode get?", then reshapes a search distribution toward whatever parameters produced the best full episodes. It's a black-box optimizer wrapped around the environment.

### Policy parameterization (this repo)

The policy is a table of per-state action *preferences* (logits), `theta` with shape `(|S|, |A|)`, converted into a probability distribution with softmax:

```
pi_theta(a | s) = softmax( theta[s, :] )_a
                = exp(theta[s,a]) / sum_a' exp(theta[s,a'])
```

This plays the same structural role as the Q-table in the Q-learning and SARSA demos — one number per `(state, action)` pair — but it's optimized in a completely different way: never bootstrapped from a Bellman-style update, only ever nudged toward whatever full episodes scored highest.

Because `pi_theta` is a *probability distribution* over actions (via softmax) rather than a hard `argmax`, a single rollout can still take non-best actions sometimes, purely due to that randomness — but that's a different kind of randomness from CEM's actual exploration mechanism. See the next subsection.

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

### How CEM explores (no epsilon-greedy at all)

This is the part that trips people up coming from Q-learning/SARSA, so it's worth spelling out explicitly: **CEM has no per-action exploration step, and no epsilon.** There is no moment where it looks at `Q(s,a)` values and decides "explore this time." Instead, exploration and exploitation both happen at the *population* level, once per generation, not the per-action level:

- **Exploration** = the spread (`sigma`) of the search distribution. A large `sigma` means the sampled candidate policies `theta_i` are wildly different from each other and from the current mean — the algorithm is trying lots of very different strategies at once, most of which will fail.
- **Exploitation** = throwing away everything except the elites, then recentering `mu` on them. Only the best-performing candidates influence where the search distribution moves next — everything below the elite cutoff is discarded, no matter how close it came.

Think of `sigma` as playing a similar *role* to `epsilon` in Q-learning/SARSA — it's the knob that controls how much the algorithm is still trying new things versus converging on what it already believes is good — but it operates on entire *policies*, sampled once per generation, rather than on individual *actions*, sampled once per step.

Left alone, `sigma` naturally shrinks over generations as the elites cluster together (that's exploitation happening automatically). The problem is it can shrink *too fast*: once a handful of similar elites are found, their variance can collapse toward zero within a few generations, and the search stops exploring — often before finding the actual optimum. That's what the extra "noise floor" term is for.

### Why the extra "noise floor" term?

Adding a small amount of *extra* Gaussian noise back into `sigma` each generation, decaying it over time, keeps exploration alive for longer than plain CEM would on its own. This is the same trick used in Szita & Lörincz's well-known "Learning Tetris Using the Noisy Cross-Entropy Method" (2006) — see [References](#references).

### This repo's exact setup

`python/cross_entropy_method.py` uses the **same environment, discount, and living reward** as the Q-learning script (`gamma = 0.9`, living reward `-0.04`, same 4×3 layout) so all three methods are directly comparable. Default hyperparameters:

| Hyperparameter | Default | Meaning |
|---|---|---|
| `population` (`N`) | 60 | candidates sampled per generation |
| `elite-frac` (`rho`) | 0.2 | fraction kept as elites (top 12 of 60) |
| `episodes` | 8 | rollouts averaged per candidate, per generation (reduces evaluation noise) |
| `init-std` | 1.0 | initial spread of the search distribution — CEM's starting "exploration budget" |
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
| **How it exploits** | `argmax_a Q(s,a)` — used both to act and inside the update target | `argmax_a Q(s,a)` — used to act, most of the time | Keeping only the elite candidates and recentering the search distribution on them |
| **How it explores** | Epsilon-greedy random actions, bolted on to behavior only — never affects the update target | Epsilon-greedy random actions — and unlike Q-learning, these *do* affect the update target when `a'` is the random choice | Sampling a whole population of candidate policies from `N(mu, sigma^2)`; `sigma` (plus the decaying noise floor) controls how much exploration remains |
| **Behavior near risk** | Learns the truly optimal (possibly risky) policy, assuming greedy behavior after training | Learns a more cautious policy if exploration itself is dangerous (see the "Cliff Walking" discussion above) | Depends entirely on the stochastic `pi_theta` used during rollouts, same spirit as SARSA |
| **What happens to rarely-visited states** | `Q` stays near its initial value until visited | `Q` stays near its initial value until visited | `theta` row stays near its initial value — may show an arbitrary "policy" that was simply never exercised |
| **Natural next step in this repo's roadmap** | → DQN (swap the table for a neural network) | → forms the "on-policy" half of the DQN/Actor-Critic family later in the roadmap | → Policy Gradient methods (REINFORCE), which replace CEM's population search with an actual gradient estimate of `E[R]` w.r.t. `theta` |
| **Where in this repo** | `python/q_learning_gridworld.py` | `python/sarsa_gridworld.py` | `python/cross_entropy_method.py` |

All three are **model-free** — none of them is ever given the transition probabilities or reward function; they only interact with the environment through `(state, action) -> (reward, next state)`. That's what distinguishes all three from Value/Q/Policy *Iteration* (covered in `docs/index.html`), which require a fully known MDP model up front.

Q-learning and SARSA are also both **tabular TD-control** methods — same update shape, same one-transition-at-a-time learning — which makes them the cleanest possible pair for seeing on-policy vs. off-policy side by side. CEM sits in a different family entirely: **episodic, gradient-free policy search**, evaluating whole policies rather than individual actions, and exploring in policy space rather than action space.

---

## References

- Rummery, G.A. & Niranjan, M. (1994). *On-Line Q-Learning Using Connectionist Systems.* Technical Report CUED/F-INFENG/TR 166, Cambridge University — the original SARSA algorithm (named later by Sutton & Barto).
- Rubinstein, R.Y. (1999). *The Cross-Entropy Method for Combinatorial and Continuous Optimization.*
- Szita, I. & Lörincz, A. (2006). *Learning Tetris Using the Noisy Cross-Entropy Method.* Neural Computation. — source of the decaying "noise floor" trick used here.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.) — Chapter 2 covers the exploration-exploitation dilemma in its simplest form (multi-armed bandits); Chapter 6 covers both Q-learning and SARSA, including the Cliff Walking on-policy-vs-off-policy example referenced above; the broader policy-search family (of which CEM is a gradient-free member) is discussed in Chapter 13.
- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* — source of the 4×3 gridworld environment used throughout this repo.
- Dan Klein & Pieter Abbeel, Berkeley CS188 course slides — popularized the "Example of Learned Q-Function" version of the gridworld used here.
