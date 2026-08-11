"""
cross_entropy_method.py
========================

Cross-Entropy Method (CEM) applied to the classic 4x3 gridworld -- the same
"Example of Learned Q-Function" environment used in python/q_learning_gridworld.py
(Berkeley CS188 / Russell & Norvig AIMA), so results are directly comparable
to the tabular Q-learning demo.

Unlike Q-learning, which learns Q(s,a) values via temporal-difference
bootstrapping from every single transition, the Cross-Entropy Method is a
*black-box, gradient-free policy search* method: it never looks inside the
environment's dynamics or computes a Bellman backup at all. Instead it:

  1. Parameterizes the policy directly as a table of per-state action
     preferences (logits) theta[s, a], turned into a probability
     distribution via softmax.
  2. Maintains a Gaussian search distribution N(mu, sigma^2) over the
     flattened parameter vector theta.
  3. Each generation, samples a population of candidate parameter vectors,
     evaluates each one by rolling out full episodes and summing discounted
     reward, keeps only the top-performing ("elite") fraction, and refits
     mu/sigma to those elites.
  4. Repeats until the population converges on a high-return policy.

See ALGORITHMS.md for the full derivation, formulae, and a side-by-side
comparison with Tabular Q-Learning.

Usage:
    python cross_entropy_method.py                     # interactive matplotlib GUI
    python cross_entropy_method.py --headless --generations 60
"""

import argparse

import numpy as np
import matplotlib

# ----------------------------------------------------------------------
# Environment: classic 4x3 AIMA / CS188 gridworld
# (same layout/rewards/discount as python/q_learning_gridworld.py)
# ----------------------------------------------------------------------
N_ROWS, N_COLS = 3, 4
WALL = (1, 1)
GOAL = (0, 3)
PIT = (1, 3)
START = (2, 0)
LIVING_REWARD = -0.04
GAMMA = 0.9
TRANSITION_NOISE = 0.8  # probability of moving in the intended direction
MAX_STEPS = 100

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
PERPENDICULAR = {
    "UP": ["LEFT", "RIGHT"],
    "DOWN": ["LEFT", "RIGHT"],
    "LEFT": ["UP", "DOWN"],
    "RIGHT": ["UP", "DOWN"],
}

STATES = [(r, c) for r in range(N_ROWS) for c in range(N_COLS) if (r, c) != WALL]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)
N_ACTIONS = len(ACTIONS)


def in_bounds(rc):
    r, c = rc
    return 0 <= r < N_ROWS and 0 <= c < N_COLS and (r, c) != WALL


def env_step(state, action, rng):
    """One stochastic transition. Returns (next_state, reward, done)."""
    if state == GOAL or state == PIT:
        return state, 0.0, True

    roll = rng.random()
    if roll < TRANSITION_NOISE:
        actual_action = action
    elif roll < TRANSITION_NOISE + (1 - TRANSITION_NOISE) / 2:
        actual_action = PERPENDICULAR[action][0]
    else:
        actual_action = PERPENDICULAR[action][1]

    dr, dc = ACTION_DELTA[actual_action]
    nxt = (state[0] + dr, state[1] + dc)
    if not in_bounds(nxt):
        nxt = state

    if nxt == GOAL:
        return nxt, 1.0, True
    if nxt == PIT:
        return nxt, -1.0, True
    return nxt, LIVING_REWARD, False


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def rollout(theta, rng, max_steps=MAX_STEPS):
    """Run one episode under the stochastic policy defined by theta
    (shape N_STATES x N_ACTIONS). Returns total discounted return."""
    state = START
    total, discount = 0.0, 1.0
    for _ in range(max_steps):
        probs = softmax(theta[STATE_IDX[state]])
        action = ACTIONS[rng.choice(N_ACTIONS, p=probs)]
        state, reward, done = env_step(state, action, rng)
        total += discount * reward
        discount *= GAMMA
        if done:
            break
    return total


def evaluate(theta_flat, episodes_per_candidate, rng):
    theta = theta_flat.reshape(N_STATES, N_ACTIONS)
    returns = [rollout(theta, rng) for _ in range(episodes_per_candidate)]
    return float(np.mean(returns))


# ----------------------------------------------------------------------
# Cross-Entropy Method
# ----------------------------------------------------------------------
class CEMTrainer:
    def __init__(self, population=60, elite_frac=0.2, init_std=1.0,
                 noise_floor=0.5, noise_decay_generations=100,
                 episodes_per_candidate=8, seed=None):
        self.n_params = N_STATES * N_ACTIONS
        self.population = population
        self.n_elite = max(1, int(population * elite_frac))
        self.episodes_per_candidate = episodes_per_candidate
        self.noise_floor = noise_floor
        self.noise_decay_generations = noise_decay_generations
        self.init_std = init_std
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(self.n_params)
        self.std = np.full(self.n_params, init_std)
        self.generation = 0
        self.history_best = []
        self.history_mean = []

    def step_generation(self):
        """Run exactly one CEM generation. Returns (best_return, mean_return)."""
        samples = self.mean + self.std * self.rng.standard_normal(
            (self.population, self.n_params)
        )
        fitness = np.array([
            evaluate(theta, self.episodes_per_candidate, self.rng)
            for theta in samples
        ])
        elite_idx = np.argsort(fitness)[-self.n_elite:]
        elites = samples[elite_idx]

        self.mean = elites.mean(axis=0)
        # Extra decaying noise keeps exploration alive and avoids premature
        # collapse of the search distribution (Szita & Lorincz, 2006).
        decay = max(0.0, 1.0 - self.generation / self.noise_decay_generations)
        self.std = elites.std(axis=0) + self.noise_floor * decay

        self.generation += 1
        best, mean_ret = float(fitness.max()), float(fitness.mean())
        self.history_best.append(best)
        self.history_mean.append(mean_ret)
        return best, mean_ret

    def greedy_policy(self):
        """Deterministic argmax policy implied by the current mean parameters."""
        theta = self.mean.reshape(N_STATES, N_ACTIONS)
        return {s: ACTIONS[int(np.argmax(theta[STATE_IDX[s]]))] for s in STATES}

    def reset(self):
        self.mean = np.zeros(self.n_params)
        self.std = np.full(self.n_params, self.init_std)
        self.generation = 0
        self.history_best.clear()
        self.history_mean.clear()


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def draw_grid(ax, trainer):
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_xlim(0, N_COLS)
    ax.set_ylim(0, N_ROWS)
    ax.set_xticks(range(N_COLS + 1))
    ax.set_yticks(range(N_ROWS + 1))
    ax.grid(True, color="#dddddd")
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    policy = trainer.greedy_policy()

    for r in range(N_ROWS):
        for c in range(N_COLS):
            rc = (r, c)
            x, y = c, N_ROWS - 1 - r  # flip so row 0 renders at the top

            if rc == WALL:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="dimgray"))
                continue
            if rc == GOAL:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#c8f7c5"))
                ax.text(x + 0.5, y + 0.5, "+1", ha="center", va="center",
                        fontsize=14, fontweight="bold")
                continue
            if rc == PIT:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#f7c5c5"))
                ax.text(x + 0.5, y + 0.5, "-1", ha="center", va="center",
                        fontsize=14, fontweight="bold")
                continue

            ax.text(x + 0.06, y + 0.94, f"{r},{c}", fontsize=7,
                     color="gray", va="top")

            action = policy[rc]
            dr, dc = ACTION_DELTA[action]
            ax.arrow(x + 0.5, y + 0.5, dc * 0.28, -dr * 0.28,
                      head_width=0.12, head_length=0.12,
                      fc="#1565c0", ec="#1565c0", linewidth=1.5)

            if rc == START:
                ax.plot(x + 0.5, y + 0.5, "o", color="#e65100",
                         markersize=9, zorder=5)

    best = max(trainer.history_best) if trainer.history_best else 0.0
    ax.set_title(f"Generation {trainer.generation}  |  best return so far: {best:.3f}",
                 fontsize=11)


def draw_curve(ax, trainer):
    ax.clear()
    if trainer.history_best:
        ax.plot(trainer.history_best, label="best in generation", color="#1565c0")
        ax.plot(trainer.history_mean, label="population mean", color="#e65100", alpha=0.7)
        ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("generation")
    ax.set_ylabel("discounted return")
    ax.set_title("CEM convergence", fontsize=11)


def run_headless(trainer, generations, out_path):
    import matplotlib.pyplot as plt

    for _ in range(generations):
        trainer.step_generation()

    fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_grid(ax_grid, trainer)
    draw_curve(ax_curve, trainer)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    best = max(trainer.history_best) if trainer.history_best else 0.0
    print(f"Ran {trainer.generation} generations. Best discounted return: {best:.3f}")
    print(f"Saved figure to {out_path}")


def run_interactive(trainer):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(11, 5.5))
    ax_grid = fig.add_axes([0.05, 0.16, 0.42, 0.75])
    ax_curve = fig.add_axes([0.55, 0.16, 0.40, 0.75])

    def redraw():
        draw_grid(ax_grid, trainer)
        draw_curve(ax_curve, trainer)
        fig.canvas.draw_idle()

    def on_next(_event):
        trainer.step_generation()
        redraw()

    def on_train(_event):
        for _ in range(20):
            trainer.step_generation()
        redraw()

    def on_reset(_event):
        trainer.reset()
        redraw()

    ax_next = fig.add_axes([0.06, 0.02, 0.16, 0.06])
    ax_train = fig.add_axes([0.24, 0.02, 0.22, 0.06])
    ax_reset = fig.add_axes([0.48, 0.02, 0.14, 0.06])
    btn_next = Button(ax_next, "Next Gen")
    btn_train = Button(ax_train, "Train 20 Gens")
    btn_reset = Button(ax_reset, "Reset")
    btn_next.on_clicked(on_next)
    btn_train.on_clicked(on_train)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Entropy Method (CEM) policy search on the 4x3 AIMA gridworld"
    )
    parser.add_argument("--headless", action="store_true",
                         help="train silently and save a PNG instead of opening a window")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=8,
                         help="rollouts averaged per candidate, per generation")
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--noise-floor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="cem_result.png")
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    trainer = CEMTrainer(
        population=args.population,
        elite_frac=args.elite_frac,
        init_std=args.init_std,
        noise_floor=args.noise_floor,
        episodes_per_candidate=args.episodes,
        seed=args.seed,
    )

    if args.headless:
        run_headless(trainer, args.generations, args.out)
    else:
        run_interactive(trainer)


if __name__ == "__main__":
    main()
