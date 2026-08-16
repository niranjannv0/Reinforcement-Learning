"""
value_iteration_gridworld.py
=============================

Value Iteration on the classic 4x3 gridworld -- the same environment used in
q_learning_gridworld.py, sarsa_gridworld.py, and cross_entropy_method.py
(Berkeley CS188 / Russell & Norvig AIMA), so results are directly comparable.

This is the odd one out among the five demos in this repo, and deliberately
so: Q-learning, SARSA, and the Cross-Entropy Method are all **model-free** --
they never see the environment's transition probabilities or reward
function, only the (state, action) -> (reward, next state) samples they
happen to experience while wandering around. Value Iteration is the
opposite: it's a **model-based planning** algorithm. It's handed the full
transition model up front (every possible outcome of every action, with its
exact probability) and never takes a single real step in the environment --
it just repeatedly sweeps over every state, applying the Bellman optimality
equation, until the value estimates stop changing.

    V(s) <- max_a  sum_{s'} P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]

Where Q-learning has to walk into the pit a few times before it learns that
state is bad, Value Iteration already "knows" -- it can just look up the
transition model and compute the consequence directly, without ever setting
foot in the environment.

See ALGORITHMS.md for the full derivation, formulae, and how this fits
alongside the model-free methods elsewhere in this repo.

Usage:
    python value_iteration_gridworld.py                     # interactive matplotlib GUI
    python value_iteration_gridworld.py --headless --sweeps 30
"""

import argparse

import numpy as np
import matplotlib

# ----------------------------------------------------------------------
# Environment: classic 4x3 AIMA / CS188 gridworld
# (same layout/rewards/discount as the other scripts in this repo)
# ----------------------------------------------------------------------
N_ROWS, N_COLS = 3, 4
WALL = (1, 1)
GOAL = (0, 3)
PIT = (1, 3)
START = (2, 0)
LIVING_REWARD = -0.04
GAMMA = 0.9
TRANSITION_NOISE = 0.8  # probability of moving in the intended direction
THETA = 1e-4            # convergence threshold on the largest per-sweep change in V

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
NONTERMINAL_STATES = [s for s in STATES if s not in (GOAL, PIT)]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)


def in_bounds(rc):
    r, c = rc
    return 0 <= r < N_ROWS and 0 <= c < N_COLS and (r, c) != WALL


def transitions(state, action):
    """The full transition model for one (state, action) pair: every possible
    outcome under the classic AIMA 0.8/0.1/0.1 stochastic dynamics, as a list
    of (probability, next_state, reward) tuples. This is the "model" that
    makes Value Iteration model-*based* -- it never has to sample a real
    transition, it can enumerate all of them directly."""
    outcomes = []
    for actual_action, prob in (
        (action, TRANSITION_NOISE),
        (PERPENDICULAR[action][0], (1 - TRANSITION_NOISE) / 2),
        (PERPENDICULAR[action][1], (1 - TRANSITION_NOISE) / 2),
    ):
        dr, dc = ACTION_DELTA[actual_action]
        nxt = (state[0] + dr, state[1] + dc)
        if not in_bounds(nxt):
            nxt = state  # bumping a wall/boundary leaves the agent in place
        if nxt == GOAL:
            reward = 1.0
        elif nxt == PIT:
            reward = -1.0
        else:
            reward = LIVING_REWARD
        outcomes.append((prob, nxt, reward))
    return outcomes


def q_from_v(V, state, action):
    """One-step lookahead: the action-value implied by the CURRENT V estimate.
    This is exactly the right-hand side of the Bellman equation for one
    fixed action -- Value Iteration takes the max of this over all actions;
    Policy Iteration (in policy_iteration_gridworld.py) plugs in whatever
    action the current policy prescribes instead."""
    return sum(p * (r + GAMMA * V[STATE_IDX[s2]]) for p, s2, r in transitions(state, action))


# ----------------------------------------------------------------------
# Value Iteration
# ----------------------------------------------------------------------
class ValueIterationTrainer:
    def __init__(self):
        self.V = np.zeros(N_STATES)
        self.sweep = 0
        self.history_delta = []

    def next_sweep(self):
        """One synchronous sweep: compute a fresh V for every non-terminal
        state from the OLD V (all states updated from the same snapshot),
        then replace V all at once. Returns the largest change seen."""
        new_V = self.V.copy()
        delta = 0.0
        for s in NONTERMINAL_STATES:
            idx = STATE_IDX[s]
            best_value = max(q_from_v(self.V, s, a) for a in ACTIONS)
            delta = max(delta, abs(best_value - self.V[idx]))
            new_V[idx] = best_value
        self.V = new_V
        self.sweep += 1
        self.history_delta.append(delta)
        return delta

    def run_to_convergence(self, max_sweeps=500):
        for _ in range(max_sweeps):
            d = self.next_sweep()
            if d < THETA:
                break

    def greedy_policy(self):
        """The policy implied by the current V estimate -- recomputed fresh
        every time via one-step lookahead. Even a half-converged V already
        implies *some* policy; watching this policy's arrows settle into
        place as V converges is the classic Value Iteration visualization."""
        return {s: max(ACTIONS, key=lambda a: q_from_v(self.V, s, a)) for s in NONTERMINAL_STATES}

    def reset(self):
        self.V[:] = 0.0
        self.sweep = 0
        self.history_delta.clear()


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def draw_grid(ax, trainer):
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    ax.clear()
    ax.set_xlim(0, N_COLS)
    ax.set_ylim(0, N_ROWS)
    ax.set_xticks(range(N_COLS + 1))
    ax.set_yticks(range(N_ROWS + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.grid(True, color="#cccccc", linewidth=0.5)

    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
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

            v = trainer.V[STATE_IDX[rc]]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=cmap(norm(v)), alpha=0.85))
            ax.text(x + 0.5, y + 0.34, f"{v:.3f}", ha="center", va="center",
                     fontsize=11, fontweight="bold")
            ax.text(x + 0.05, y + 0.95, f"{r},{c}", fontsize=6,
                     color="black", va="top", ha="left", alpha=0.6)

            action = policy[rc]
            dr, dc = ACTION_DELTA[action]
            ax.arrow(x + 0.5, y + 0.68, dc * 0.2, -dr * 0.2,
                      head_width=0.1, head_length=0.1,
                      fc="#111111", ec="#111111", linewidth=1.3)

            if rc == START:
                ax.plot(x + 0.9, y + 0.1, marker="*", color="#1565c0",
                         markersize=9, zorder=6)

    ax.set_title(f"Sweep {trainer.sweep}  |  max |delta V| = "
                 f"{trainer.history_delta[-1] if trainer.history_delta else float('nan'):.4f}",
                 fontsize=11)


def draw_curve(ax, trainer):
    ax.clear()
    if trainer.history_delta:
        ax.plot(trainer.history_delta, color="#1565c0")
        ax.axhline(THETA, color="#e65100", linestyle="--", linewidth=1,
                    label=f"convergence threshold ({THETA})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_yscale("log")
    ax.set_xlabel("sweep")
    ax.set_ylabel("max |delta V|  (Bellman residual)")
    ax.set_title("Convergence", fontsize=11)


def run_headless(trainer, sweeps, out_path):
    import matplotlib.pyplot as plt

    if sweeps > 0:
        for _ in range(sweeps):
            trainer.next_sweep()
    else:
        trainer.run_to_convergence()

    fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_grid(ax_grid, trainer)
    draw_curve(ax_curve, trainer)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    last_delta = trainer.history_delta[-1] if trainer.history_delta else float("nan")
    print(f"Ran {trainer.sweep} sweeps. Final max |delta V| = {last_delta:.6f}")
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
        trainer.next_sweep()
        redraw()

    def on_run(_event):
        trainer.run_to_convergence()
        redraw()

    def on_reset(_event):
        trainer.reset()
        redraw()

    ax_next = fig.add_axes([0.08, 0.02, 0.18, 0.06])
    ax_run = fig.add_axes([0.28, 0.02, 0.24, 0.06])
    ax_reset = fig.add_axes([0.54, 0.02, 0.14, 0.06])
    btn_next = Button(ax_next, "Next Sweep")
    btn_run = Button(ax_run, "Run to Convergence")
    btn_reset = Button(ax_reset, "Reset")
    btn_next.on_clicked(on_next)
    btn_run.on_clicked(on_run)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Value Iteration (model-based planning) on the 4x3 AIMA gridworld"
    )
    parser.add_argument("--headless", action="store_true",
                         help="run silently and save a PNG instead of opening a window")
    parser.add_argument("--sweeps", type=int, default=0,
                         help="run exactly this many sweeps (headless mode). "
                              "0 (default) means: run to convergence")
    parser.add_argument("--out", type=str, default="value_iteration_result.png")
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    trainer = ValueIterationTrainer()

    if args.headless:
        run_headless(trainer, args.sweeps, args.out)
    else:
        run_interactive(trainer)


if __name__ == "__main__":
    main()