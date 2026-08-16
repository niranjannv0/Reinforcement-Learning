# The Bellman Equation: A Complete Interactive Walkthrough

[![CI](https://github.com/niranjannv0/Reinforcement-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/niranjannv0/Reinforcement-Learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Learn Value Iteration, Q-Iteration, and Policy Iteration by watching them happen — a single self-contained HTML page with animated, worked-through Bellman equation examples, plus hands-on Python demos of Tabular Q-Learning, SARSA, and the Cross-Entropy Method. Built for reinforcement learning beginners.

**[→ Live demo](#live-demo)** &nbsp;|&nbsp; **[→ Topics covered](#-topics-covered)** &nbsp;|&nbsp; **[→ Algorithms reference](./ALGORITHMS.md)** &nbsp;|&nbsp; **[→ Roadmap](#-roadmap--planned-additions)**


<p align="center">
  <img src="figs/rl_information_flow.svg" alt="Reinforcement learning agent-environment information flow with value function updates" width="700">
</p>

---

## Table of contents

- [About](#about)
- [Key Concepts: Q-values & Explore vs. Exploit](#-key-concepts-q-values--explore-vs-exploit)
- [What's inside](#-whats-inside)
- [Live demo](#live-demo)
- [Python demo: Q-learning gridworld](#-python-demo-q-learning-gridworld)
- [Python demo: SARSA](#-python-demo-sarsa)
- [Python demo: Cross-Entropy Method](#-python-demo-cross-entropy-method-cem)
- [Repo structure](#-repo-structure)
- [Topics covered](#-topics-covered)
- [Roadmap — planned additions](#-roadmap--planned-additions)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## About

Most explanations of the Bellman equation show you the formula and expect you to trust it. This repo does the opposite: every number in every equation traces back to one small, consistent worked example (a 3-state MDP) that you can click through step by step, so you can see exactly where each value comes from and why it changes.

The HTML guide is a single file — no build step, no framework, no install. Open it in a browser and everything runs, including the animations.

For the three Python demos (Tabular Q-Learning, SARSA, and the Cross-Entropy Method), see [`ALGORITHMS.md`](./ALGORITHMS.md) for the full formulae, pseudocode, and a side-by-side comparison of how each one actually learns.

## 🔑 Key Concepts: Q-values & Explore vs. Exploit

New to reinforcement learning? Start here before the demos below.

A **Q-value**, written `Q(s,a)`, answers one question: *"If I'm in state `s` and I take action `a`, how much total reward can I expect from here on, assuming I act well afterward?"* All three algorithms in this repo are, underneath, about answering that question — Q-learning and SARSA estimate `Q(s,a)` directly and store it in a table, one number per (state, action) pair; the Cross-Entropy Method skips `Q(s,a)` entirely and searches for a good policy directly, judging it only by how much total reward a full episode earned.

Every one of these algorithms also has to solve the same dilemma, usually called **exploitation vs. exploration**:

- **Exploit** = use what you currently believe is best — take `argmax_a Q(s,a)`, i.e. "the action with the highest Q-value so far."
- **Explore** = deliberately try something else anyway, because your current Q-values might still be wrong, especially early in training when most of the table hasn't been tested yet.

| Algorithm | Exploits by... | Explores by... |
|---|---|---|
| **Q-Learning** | Taking `argmax_a Q(s,a)` — both to act, *and* baked directly into its learning target | Bolted on separately via epsilon-greedy: act randomly `epsilon` fraction of the time, purely to gather data |
| **SARSA** | Taking `argmax_a Q(s,a)` — to act, most of the time | Also epsilon-greedy — but here, the random exploratory action can directly shape *what gets learned*, not just how the agent behaves (see [`ALGORITHMS.md`](./ALGORITHMS.md#2-sarsa)) |
| **Cross-Entropy Method** | Keeping only the best-performing ("elite") candidate policies each generation, and narrowing the search toward them | Sampling a whole population of different candidate policies from a search distribution every generation — no epsilon-greedy, no per-action randomness at all |

For the full definitions (states, actions, rewards, returns, `V(s)` vs. `Q(s,a)` and how they relate) and a much deeper dive into exactly how each algorithm balances exploration and exploitation, see the **[Foundations section of `ALGORITHMS.md`](./ALGORITHMS.md#foundations)**.

## ✨ What's inside

- A worked 3-state MDP (S1, S2, S3) with two actions per state, used consistently across every section so the numbers stay comparable throughout
- A beginner glossary — state, action, reward, transition, discount factor, value, Q-value, policy, terminal state
- An animated agent that physically moves between states in the diagram, with each step explaining **why** that action was chosen (via a live Q-value comparison) and **which** state it led to
- Full round-by-round Bellman equations for Value Iteration (V-based) and Q-Iteration (Q-based) — not just the final answer, every intermediate round
- Interactive step-through animations for Value Iteration, Q-Iteration, and Policy Iteration, each with live counters showing how often utilities vs. policy actually get updated
- A side-by-side comparison chart of all three algorithms
- A standalone Python script, `python/q_learning_gridworld.py`, that recreates the classic 4×3 "learned Q-function" gridworld and runs **tabular Q-learning** on it — with an interactive matplotlib GUI (click-to-step buttons, live Q-value triangles, cell position labels) that starts from an all-zero Q-table so you can watch the very first episode learn in real time
- A second standalone Python script, `python/sarsa_gridworld.py`, that solves the **same gridworld** with **SARSA** — the on-policy sibling of Q-learning — using the same triangle-per-cell Q-value GUI, so you can watch its learned Q-values diverge from Q-learning's near the pit, where exploring is actually risky
- A third standalone Python script, `python/cross_entropy_method.py`, that solves the **same gridworld** with the **Cross-Entropy Method** — a gradient-free, population-based policy search — so you can watch a completely different learning paradigm converge on the same problem, with its own interactive matplotlib GUI (step-by-generation, live policy arrows, convergence curve)
- [`ALGORITHMS.md`](./ALGORITHMS.md) — a standalone reference doc with the full math, pseudocode, hyperparameters, and a Q-Learning-vs-SARSA-vs-CEM comparison table for all three implemented algorithms, starting with a beginner-friendly [Foundations](./ALGORITHMS.md#foundations) section that defines states, actions, rewards, `V(s)`, `Q(s,a)`, and the exploration-vs-exploitation dilemma from scratch

## Live demo

1. In this repo, go to **Settings → Pages**
2. Under "Build and deployment," set **Source** to "Deploy from a branch," branch `main`, folder **`/docs`**
3. Save — GitHub will publish the page at:

   ```
   https://niranjannv0.github.io/Reinforcement-Learning/
   ```

No hosting? Just clone the repo and open `docs/index.html` directly in any browser. Everything works fully offline **except** the final comparison chart, which loads Chart.js from a CDN.

```bash
git clone https://github.com/niranjannv0/Reinforcement-Learning.git
cd Reinforcement-Learning
open docs/index.html   # or just double-click it
```

## 🐍 Python demo: Q-learning gridworld

`python/q_learning_gridworld.py` recreates the classic 4×3 gridworld (a wall, a +1 goal, a −1 pit — based on the well-known "Example of Learned Q-Function" slide popularized in Berkeley's CS188, itself derived from the AIMA textbook's gridworld) and runs real tabular **Q-learning** on it, not value/Q-iteration — meaning the agent doesn't know the reward or transition model up front and has to learn Q(s,a) purely by walking around and experiencing transitions.

The whole thing is rendered with **matplotlib**: each cell is split into four triangles (one per action — up/down/left/right), colored and labeled with that action's live Q-value, exactly like the original slide. Every cell also shows its `(row, col)` position in the corner, and the agent's current location is marked with an orange dot.

**Exploit vs. explore here:** while training, the agent picks actions epsilon-greedily — mostly `argmax_a Q(s,a)` (exploit), occasionally a random action (explore) — but the update rule itself always plugs in `max_a' Q(s',a')` regardless of what it's about to actually do. See [Exploitation and exploration in Q-learning](./ALGORITHMS.md#exploitation-and-exploration-in-q-learning) for exactly why that matters.

Requires Python 3.8+ and matplotlib — see [`requirements.txt`](./requirements.txt):

```bash
pip install -r requirements.txt
python python/q_learning_gridworld.py
```

This opens an interactive window with three buttons:

- **Next Step** — takes exactly one Q-learning update from an all-zero Q-table, moving the agent one cell and printing the exact update underneath the grid, e.g.:

   ```
   Q((2, 0),RIGHT) <- 0.000 + 0.1 * [ -0.04 + 0.9 * 0.000 - 0.000 ]
   Q((2, 0),RIGHT) <- 0.000 + 0.1 * (-0.040)  =  -0.004
   ```

   Click it repeatedly to watch Q-values spread outward from zero, cell by cell, as the agent explores — including the moment its greedy policy flips direction once it learns enough.

- **Train 1,000 Episodes** — resets to a blank Q-table and trains fresh in one shot (matching the slide's "Q-VALUES AFTER 1000 EPISODES" caption), then displays the converged grid with the best action in each cell shown in bold.

- **Reset** — clears everything back to an all-zero Q-table.

No display available (e.g. over SSH, or in CI)? Run headless — it trains silently and saves a PNG snapshot instead of opening a window:

```bash
python python/q_learning_gridworld.py --headless --episodes 500
```

## 🐍 Python demo: SARSA

`python/sarsa_gridworld.py` solves the **exact same 4×3 gridworld** as the Q-learning demo (same layout, same `-0.04` living reward, same `gamma = 0.9` discount, same stochastic transition model), but with **SARSA** instead — the on-policy sibling of Q-learning. The two algorithms look almost identical (both learn `Q(s,a)` from single transitions via TD updates), but SARSA's update target uses the value of the action its own epsilon-greedy policy *actually takes next*, rather than the best possible action. See [`ALGORITHMS.md`](./ALGORITHMS.md#2-sarsa) for the full derivation and why that one substitution matters.

The visualization reuses the same four-triangle-per-cell Q-value layout as the Q-learning demo (color-coded by value, best action's triangle outlined in bold), so you can run both scripts side by side and directly compare what each one learns for the same state. Watch state `(1,2)`, right next to the pit — SARSA's `Q((1,2), RIGHT)` gets pulled sharply negative, because an unlucky exploratory move really can walk the agent into the pit from there, while Q-learning's greedy-max target doesn't factor that exploration risk in at all.

**Exploit vs. explore here:** SARSA uses the exact same epsilon-greedy action selection as Q-learning — but where Q-learning's update always assumes the best possible next action, SARSA's update plugs in whichever action epsilon-greedy *actually* just picked. That means roughly `epsilon` fraction of updates use a genuinely random next action, and the risk of that shows up directly in the learned Q-values. See [Exploitation and exploration in SARSA](./ALGORITHMS.md#exploitation-and-exploration-in-sarsa) for the full picture.

Requires Python 3.8+ and matplotlib (same as above):

```bash
pip install -r requirements.txt
python python/sarsa_gridworld.py
```

This opens an interactive window with three buttons:

- **Next Step** — takes exactly one SARSA update from an all-zero Q-table, advances the agent, and prints the exact update underneath the grid, e.g.:

   ```
   Q((0, 2),RIGHT) <- 0.832 + 0.1 * [ +1.00 + 0.9 * 0.000 - 0.832 ]
   Q((0, 2),RIGHT) <- 0.832 + 0.1 * (+0.168)  =  0.848
   ```

- **Train 1000 Episodes** — runs 1000 more episodes from wherever the Q-table currently is, then redraws the converged grid with the best action in each cell outlined in bold.

- **Reset** — clears everything back to an all-zero Q-table.

No display available (e.g. over SSH, or in CI)? Run headless — it trains silently and saves a PNG snapshot instead of opening a window:

```bash
python python/sarsa_gridworld.py --headless --episodes 1000
```

Other useful flags: `--alpha` (learning rate, default 0.1), `--epsilon` (exploration rate, default 0.1), `--seed` (for reproducible runs). Run `python python/sarsa_gridworld.py --help` for the full list.

## 🐍 Python demo: Cross-Entropy Method (CEM)

`python/cross_entropy_method.py` solves the **exact same 4×3 gridworld** (same layout, same `-0.04` living reward, same `gamma = 0.9` discount, same stochastic transition model), but with a completely different learning paradigm: the **Cross-Entropy Method**, a gradient-free, population-based policy search. Where Q-learning bootstraps a value function one transition at a time, CEM never builds a value function at all — it samples a population of candidate policies, rolls each one out for full episodes, keeps only the best-scoring ("elite") ones, and reshapes its search distribution toward them, generation after generation. See [`ALGORITHMS.md`](./ALGORITHMS.md#3-cross-entropy-method-cem) for the full derivation.

The policy is a table of per-state action logits, converted to action probabilities via softmax — visualized as a single best-action arrow per cell (the argmax of that state's logits), alongside a live convergence curve of best/average return per generation.

**Exploit vs. explore here:** CEM has no epsilon and no per-action randomness at all — instead, exploration happens once per *generation*, by sampling a whole population of candidate policies from a search distribution `N(mu, sigma^2)`; exploitation happens by keeping only the best-scoring ("elite") candidates and recentering the search on them. See [How CEM explores](./ALGORITHMS.md#how-cem-explores-no-epsilon-greedy-at-all) for the full mechanism, including why a decaying "noise floor" is added on top.

Requires Python 3.8+, numpy, and matplotlib (same as above):

```bash
pip install -r requirements.txt
python python/cross_entropy_method.py
```

This opens an interactive window with three buttons:

- **Next Gen** — runs exactly one CEM generation (sample a population, evaluate, select elites, refit the search distribution) and redraws the current greedy policy and convergence curve.
- **Train 20 Gens** — runs 20 generations back to back, useful for watching the policy converge without clicking repeatedly.
- **Reset** — reinitializes the search distribution (`mu = 0`, `sigma = init_std`) and clears the convergence history.

No display available? Run headless — it trains silently for a fixed number of generations and saves a PNG (policy grid + convergence curve) instead of opening a window:

```bash
python python/cross_entropy_method.py --headless --generations 100 --out cem_result.png
```

Other useful flags: `--population` (candidates per generation, default 60), `--elite-frac` (fraction kept as elites, default 0.2), `--episodes` (rollouts averaged per candidate, default 8), `--seed` (for reproducible runs). Run `python python/cross_entropy_method.py --help` for the full list.

## 🗂 Repo structure

```
Reinforcement-Learning/
├── README.md
├── ALGORITHMS.md                 # formulae, pseudocode & hyperparameters for every implemented algorithm
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                    # syntax-checks + smoke-tests all three Python demos on every push/PR
├── docs/
│   └── index.html                    # the entire interactive guide, single file (served via GitHub Pages)
├── figs/
│   └── rl_information_flow.svg       # agent-environment loop diagram used above
└── python/
    ├── q_learning_gridworld.py       # tabular Q-learning, 4x3 gridworld, interactive matplotlib GUI
    ├── sarsa_gridworld.py            # on-policy SARSA, same 4x3 gridworld, same GUI layout
    └── cross_entropy_method.py       # Cross-Entropy Method policy search, same 4x3 gridworld
```

## 🧠 Topics covered

| Topic | Where |
|---|---|
| MDP fundamentals (states, actions, rewards, transitions) | Glossary |
| Discounting and long-term value | Glossary, Value Iteration |
| Watching an agent transition between states | Section 3 |
| Bellman optimality equation | Section 4 |
| Value Iteration (equations + animation) | Section 4 |
| Action-values, Q(s,a), and how they relate to V(s) | Section 5 |
| Q-Iteration (equations + animation) | Section 5 |
| Value Iteration vs. Q-Iteration trade-offs | Section 5 |
| Policy Evaluation & Policy Improvement | Section 6 |
| Policy Iteration (equations + animation) | Section 6 |
| Comparing all three algorithms | Section 7 |
| Q-values & V(s) — plain-English definitions and how they relate | [`ALGORITHMS.md`](./ALGORITHMS.md#value-functions-vs-and-qsa) |
| Exploitation vs. exploration, epsilon-greedy | This README ([Key Concepts](#-key-concepts-q-values--explore-vs-exploit)), [`ALGORITHMS.md`](./ALGORITHMS.md#exploitation-vs-exploration) |
| Tabular Q-learning (off-policy, model-free, learned from experience) | `python/q_learning_gridworld.py`, [`ALGORITHMS.md`](./ALGORITHMS.md#1-tabular-q-learning) |
| SARSA (on-policy TD control, contrasted directly with Q-learning) | `python/sarsa_gridworld.py`, [`ALGORITHMS.md`](./ALGORITHMS.md#2-sarsa) |
| Cross-Entropy Method (gradient-free, black-box policy search) | `python/cross_entropy_method.py`, [`ALGORITHMS.md`](./ALGORITHMS.md#3-cross-entropy-method-cem) |
| Q-Learning vs. SARSA vs. CEM — how all three paradigms compare | [`ALGORITHMS.md`](./ALGORITHMS.md#4-all-three-side-by-side) |

## 🛣 Roadmap — planned additions

This started as a Bellman-equation-only deep dive. The plan is to keep extending it with the same **equations + interactive animation** treatment as the topics get more advanced — moving from tabular, model-based methods toward modern deep RL:

- [x] **Tabular Q-Learning** — ✅ done, see [`python/q_learning_gridworld.py`](./python/q_learning_gridworld.py) — the same Q(s,a) from this guide, but learned from sampled experience instead of a known reward/transition model
- [x] **SARSA** — ✅ done, see [`python/sarsa_gridworld.py`](./python/sarsa_gridworld.py) — on-policy TD control, contrasted directly with Q-learning's off-policy update; same environment, same GUI, one-line difference in the update rule
- [x] **Cross-Entropy Method (CEM)** — ✅ done, see [`python/cross_entropy_method.py`](./python/cross_entropy_method.py) — a gradient-free, population-based policy search baseline, solving the same gridworld a completely different way; a natural bridge before moving to gradient-based policy methods below
- [ ] **Deep Q-Networks (DQN)** — replacing the Q-table with a neural network for large or continuous state spaces, animated with a toy function-approximation example
- [ ] **Experience Replay & Target Networks** — why plain DQN training is unstable and what these two tricks fix
- [ ] **Double DQN** — the Q-value overestimation problem and its fix
- [ ] **Dueling DQN** — splitting Q into a state-value stream and an advantage stream
- [ ] **Prioritized Experience Replay**
- [ ] **Policy Gradient methods (REINFORCE)** — moving from CEM's gradient-free population search to an actual gradient estimate of expected return
- [ ] **Actor-Critic / A2C** — combining value estimation and policy optimization
- [ ] **Proximal Policy Optimization (PPO)** — the modern default for many RL problems
- [ ] **Continuous action spaces** — why "max over actions" breaks down, and how DDPG/SAC handle it
- [ ] **Exploration strategies** — ε-greedy, softmax/Boltzmann, UCB, intrinsic motivation
- [ ] **Multi-agent RL basics**

Want to help build one of these? See below.

## 🤝 Contributing

1. Fork the repo
2. For guide content: add your section inside `docs/index.html`, following the existing pattern — a short, beginner-friendly explanation, an `.eqn-block` walkthrough with real worked numbers, and a click-through stepper animation if the topic lends itself to one.
3. For algorithm code: add a new script under `python/`, and if it introduces a new dependency, add it to `requirements.txt` with a comment explaining why. Also add a matching section to [`ALGORITHMS.md`](./ALGORITHMS.md) — the formula, pseudocode, and this repo's exact hyperparameters — following the existing Q-Learning/SARSA/CEM pattern.
4. Keep the HTML guide dependency-free (Chart.js via CDN is the one exception) — it should stay copy-paste-able and work by just opening the file.
5. Run `python -m py_compile python/*.py` locally before opening a PR — CI will check this too.
6. Open a PR with a short description of what you added and why.

## 📄 License

MIT — see [LICENSE](./LICENSE). Replace `[Your Name]` in that file with your own name (or your organization's) before publishing.

## 🙏 Acknowledgments

Built as an interactive companion for learning the Bellman equation and classic MDP-solving algorithms — the kind of step-by-step, numbers-first explanation this repo tries to give for every concept it covers.

The 4×3 gridworld used across all three Python demos (`q_learning_gridworld.py`, `sarsa_gridworld.py`, and `cross_entropy_method.py`) is the classic teaching example popularized in Dan Klein and Pieter Abbeel's Berkeley CS188 course slides, itself derived from the gridworld in Russell & Norvig's *Artificial Intelligence: A Modern Approach*.

SARSA follows the original formulation from Rummery & Niranjan's *On-Line Q-Learning Using Connectionist Systems* (1994). The Cross-Entropy Method implementation's "noise floor" exploration trick follows Szita & Lörincz's *Learning Tetris Using the Noisy Cross-Entropy Method* (2006). See [`ALGORITHMS.md`](./ALGORITHMS.md#references) for the full reference list.

This repository also draws on material presented in the **CIS 522: Deep Learning** lecture series, particularly the lecture covering Markov Decision Processes and Dynamic Programming:
- https://youtu.be/rCk_hvwZ6iA
