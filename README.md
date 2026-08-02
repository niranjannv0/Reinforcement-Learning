# The Bellman Equation: A Complete Interactive Walkthrough

[![CI](https://github.com/<your-username>/bellman-equation-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/bellman-equation-guide/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Learn Value Iteration, Q-Iteration, and Policy Iteration by watching them happen — a single self-contained HTML page with animated, worked-through Bellman equation examples, plus a hands-on Python Q-learning demo. Built for reinforcement learning beginners.

**[→ Live demo](#live-demo)** &nbsp;|&nbsp; **[→ Topics covered](#-topics-covered)** &nbsp;|&nbsp; **[→ Roadmap](#-roadmap--planned-additions)**

---

## Table of contents

- [About](#about)
- [What's inside](#-whats-inside)
- [Live demo](#live-demo)
- [Python demo: Q-learning gridworld](#-python-demo-q-learning-gridworld)
- [Repo structure](#-repo-structure)
- [Topics covered](#-topics-covered)
- [Roadmap — planned additions](#-roadmap--planned-additions)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## About

Most explanations of the Bellman equation show you the formula and expect you to trust it. This repo does the opposite: every number in every equation traces back to one small, consistent worked example (a 3-state MDP) that you can click through step by step, so you can see exactly where each value comes from and why it changes.

The HTML guide is a single file — no build step, no framework, no install. Open it in a browser and everything runs, including the animations.

## ✨ What's inside

- A worked 3-state MDP (S1, S2, S3) with two actions per state, used consistently across every section so the numbers stay comparable throughout
- A beginner glossary — state, action, reward, transition, discount factor, value, Q-value, policy, terminal state
- An animated agent that physically moves between states in the diagram, with each step explaining **why** that action was chosen (via a live Q-value comparison) and **which** state it led to
- Full round-by-round Bellman equations for Value Iteration (V-based) and Q-Iteration (Q-based) — not just the final answer, every intermediate round
- Interactive step-through animations for Value Iteration, Q-Iteration, and Policy Iteration, each with live counters showing how often utilities vs. policy actually get updated
- A side-by-side comparison chart of all three algorithms
- A standalone Python script, `python/q_learning_gridworld.py`, that recreates the classic 4×3 "learned Q-function" gridworld and runs **tabular Q-learning** on it — with an interactive, press-Enter-to-step trace that starts from an all-zero Q-table so you can watch the very first episode learn in real time

## Live demo

1. In this repo, go to **Settings → Pages**
2. Under "Build and deployment," set **Source** to "Deploy from a branch," branch `main`, folder **`/docs`**
3. Save — GitHub will publish the page at:

   ```
   https://github.com/niranjannv0/Reinforcement-Learning.git
   ```

No hosting? Just clone the repo and open `docs/index.html` directly in any browser. Everything works fully offline **except** the final comparison chart, which loads Chart.js from a CDN.

```bash
git remote add origin https://github.com/niranjannv0/Reinforcement-Learning.git
cd Reinforcement-Learning
open docs/index.html   # or just double-click it
```

## 🐍 Python demo: Q-learning gridworld

`python/q_learning_gridworld.py` recreates the classic 4×3 gridworld (a wall, a +1 goal, a −1 pit — based on the well-known "Example of Learned Q-Function" slide popularized in Berkeley's CS188, itself derived from the AIMA textbook's gridworld) and runs real tabular **Q-learning** on it, not value/Q-iteration — meaning the agent doesn't know the reward or transition model up front and has to learn Q(s,a) purely by walking around and experiencing transitions.

Requires Python 3.8+. No third-party packages — see [`requirements.txt`](./requirements.txt):

```bash
python python/q_learning_gridworld.py
```

What happens when you run it:

1. **Interactive trace** — steps through the agent's very first episode one action at a time (press Enter to advance), starting from a completely blank, all-zero Q-table. Each step prints the exact Q-learning update with real numbers substituted in:

   ```
   Q((2, 0),RIGHT) <- 0.000 + 0.1 * [ -0.04 + 0.9 * 0.000 - 0.000 ]
   Q((2, 0),RIGHT) <- 0.000 + 0.1 * (-0.040)
   Q((2, 0),RIGHT) <- -0.004
   ```

   along with an ASCII redraw of the whole grid after every update, so you can watch the Q-values spread outward from zero as the agent explores.

2. **Full training run (optional)** — trains fresh for 1,000 episodes (matching the slide's "Q-VALUES AFTER 1000 EPISODES" caption) and prints the final Q-table plus the greedy policy it implies.

## 🗂 Repo structure

```
bellman-equation-guide/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               # syntax-checks + smoke-tests the Python demo on every push/PR
├── docs/
│   └── index.html                # the entire interactive guide, single file (served via GitHub Pages)
└── python/
    └── q_learning_gridworld.py   # tabular Q-learning, 4x3 gridworld, interactive CLI trace
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
| Tabular Q-learning (model-free, learned from experience) | `python/q_learning_gridworld.py` |

## 🛣 Roadmap — planned additions

This started as a Bellman-equation-only deep dive. The plan is to keep extending it with the same **equations + interactive animation** treatment as the topics get more advanced — moving from tabular, model-based methods toward modern deep RL:

- [x] **Tabular Q-Learning** — ✅ done, see [`python/q_learning_gridworld.py`](./python/q_learning_gridworld.py) — the same Q(s,a) from this guide, but learned from sampled experience instead of a known reward/transition model
- [ ] **SARSA** — on-policy TD control, contrasted directly with Q-learning's off-policy update
- [ ] **Deep Q-Networks (DQN)** — replacing the Q-table with a neural network for large or continuous state spaces, animated with a toy function-approximation example
- [ ] **Experience Replay & Target Networks** — why plain DQN training is unstable and what these two tricks fix
- [ ] **Double DQN** — the Q-value overestimation problem and its fix
- [ ] **Dueling DQN** — splitting Q into a state-value stream and an advantage stream
- [ ] **Prioritized Experience Replay**
- [ ] **Policy Gradient methods (REINFORCE)** — moving from value-based to policy-based RL
- [ ] **Actor-Critic / A2C** — combining value estimation and policy optimization
- [ ] **Proximal Policy Optimization (PPO)** — the modern default for many RL problems
- [ ] **Continuous action spaces** — why "max over actions" breaks down, and how DDPG/SAC handle it
- [ ] **Exploration strategies** — ε-greedy, softmax/Boltzmann, UCB, intrinsic motivation
- [ ] **Multi-agent RL basics**

Want to help build one of these? See below.

## 🤝 Contributing

1. Fork the repo
2. For guide content: add your section inside `docs/index.html`, following the existing pattern — a short, beginner-friendly explanation, an `.eqn-block` walkthrough with real worked numbers, and a click-through stepper animation if the topic lends itself to one.
3. For algorithm code: add a new script under `python/`, and if it introduces a new dependency, add it to `requirements.txt` with a comment explaining why.
4. Keep the HTML guide dependency-free (Chart.js via CDN is the one exception) — it should stay copy-paste-able and work by just opening the file.
5. Run `python -m py_compile python/*.py` locally before opening a PR — CI will check this too.
6. Open a PR with a short description of what you added and why.

## 📄 License

MIT — see [LICENSE](./LICENSE). Replace `[Your Name]` in that file with your own name (or your organization's) before publishing.

## 🙏 Acknowledgments

Built as an interactive companion for learning the Bellman equation and classic MDP-solving algorithms — the kind of step-by-step, numbers-first explanation this repo tries to give for every concept it covers.

The 4×3 gridworld in `python/q_learning_gridworld.py` is the classic teaching example popularized in Dan Klein and Pieter Abbeel's Berkeley CS188 course slides, itself derived from the gridworld in Russell & Norvig's *Artificial Intelligence: A Modern Approach*.

This repository also draws on material presented in the **CIS 522: Deep Learning** lecture series, particularly the lecture covering Markov Decision Processes and Dynamic Programming:
- https://youtu.be/rCk_hvwZ6iA

