"""
policy_iteration_gridworld.py
==============================

Policy Iteration on the classic 4x3 gridworld -- the same environment used
in value_iteration_gridworld.py, q_learning_gridworld.py, sarsa_gridworld.py,
and cross_entropy_method.py (Berkeley CS188 / Russell & Norvig AIMA).

Like Value Iteration, this is a **model-based planning** algorithm: it's
handed the full transition model up front and never takes a single real
step in the environment. But where Value Iteration sweeps V(s) toward the
optimum directly (taking a max over actions on every single update), Policy
Iteration keeps an explicit policy the whole time and alternates between two
very different phases:

    Policy Evaluation:   hold the policy fixed, sweep V(s) until it exactly
                          matches the value of THAT policy (no max involved --
                          just plug in whatever action the policy prescribes)

    Policy Improvement:  hold V(s) fixed, and for every state switch to
                          whichever action looks best under a one-step
                          lookahead (this is where the max shows up)

Repeat those two phases until the policy stops changing. Perhaps
surprisingly, this usually converges in far fewer *policy changes* than
Value Iteration takes sweeps, even though each Policy Evaluation phase does
more total work per iteration -- see ALGORITHMS.md for why.

Usage:
    python policy_iteration_gridworld.py                     # interactive matplotlib GUI
    python policy_iteration_gridworld.py --headless
"""

import argparse

import numpy as np
import matplotlib

# ----------------------------------------------------------------------
# Environment: classic 4x3 AIMA / CS188 gridworld
# (identical to value_iteration_gridworld.py and the other scripts here)
# ----------------------------------------------------------------------
N_ROWS, N_COLS = 3, 4
WALL = (1, 1)
GOAL = (0, 3)
PIT = (1, 3)
START = (2, 0)
LIVING_REWARD = -0.04
GAMMA = 0.9
TRANSITION_NOISE = 0.8
THETA = 1e-4  # convergence threshold for a policy-evaluation sweep

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
    """Full transition model: every possible outcome of `action` from
    `state`, as (probability, next_state, reward) tuples, under the classic
    AIMA 0.8/0.1/0.1 stochastic dynamics."""
    outcomes = []
    for actual_action, prob in (
        (action, TRANSITION_NOISE),
        (PERPENDICULAR[action][0], (1 - TRANSITION_NOISE) / 2),
        (PERPENDICULAR[action][1], (1 - TRANSITION_NOISE) / 2),
    ):
        dr, dc = ACTION_DELTA[actual_action]
        nxt = (state[0] + dr, state[1] + dc)
        if not in_bounds(nxt):
            nxt = state
        if nxt == GOAL:
            reward = 1.0
        elif nxt == PIT:
            reward = -1.0
        else:
            reward = LIVING_REWARD
        outcomes.append((prob, nxt, reward))
    return outcomes


def q_from_v(V, state, action):
    """One-step lookahead value of taking `action` in `state`, under the
    current V estimate."""
    return sum(p * (r + GAMMA * V[STATE_IDX[s2]]) for p, s2, r in transitions(state, action))


# ----------------------------------------------------------------------
# Policy Iteration
# ----------------------------------------------------------------------
class PolicyIterationTrainer:
    def __init__(self):
        self.V = np.zeros(N_STATES)
        # Start from a deliberately bad, arbitrary policy (always UP) --
        # watching it improve from something naive is much more convincing
        # than starting near-optimal.
        self.policy = {s: "UP" for s in NONTERMINAL_STATES}
        self.outer_iter = 0     # number of completed (evaluate, improve) cycles
        self.eval_sweep = 0     # sweeps within the CURRENT evaluation phase
        self.history_delta = []  # flat, across every evaluation sweep ever run
        self.improve_markers = []  # sweep indices where a policy improvement happened
        self.stable = False
        self.last_changed = None

    def evaluate_sweep(self):
        """One synchronous sweep of Policy Evaluation: update V(s) for every
        state using ONLY the action the current policy prescribes there (no
        max -- that's what makes this different from a Value Iteration
        sweep)."""
        new_V = self.V.copy()
        delta = 0.0
        for s in NONTERMINAL_STATES:
            idx = STATE_IDX[s]
            value = q_from_v(self.V, s, self.policy[s])
            delta = max(delta, abs(value - self.V[idx]))
            new_V[idx] = value
        self.V = new_V
        self.eval_sweep += 1
        self.history_delta.append(delta)
        return delta

    def evaluate_to_convergence(self, max_sweeps=500):
        for _ in range(max_sweeps):
            d = self.evaluate_sweep()
            if d < THETA:
                break

    def improve_policy(self):
        """One full Policy Improvement pass: for every state, switch to
        whichever action looks best under a one-step lookahead using the
        (now-converged, for this policy) V. Returns how many states changed
        action; zero means the policy is stable, i.e. optimal."""
        changed = 0
        for s in NONTERMINAL_STATES:
            old_a = self.policy[s]
            best_a = max(ACTIONS, key=lambda a: q_from_v(self.V, s, a))
            self.policy[s] = best_a
            if best_a != old_a:
                changed += 1
        self.outer_iter += 1
        self.eval_sweep = 0
        self.improve_markers.append(len(self.history_delta))
        self.stable = changed == 0
        self.last_changed = changed
        return changed

    def run_to_convergence(self, max_outer=50):
        for _ in range(max_outer):
            self.evaluate_to_convergence()
            changed = self.improve_policy()
            if changed == 0:
                break

    def reset(self):
        self.V[:] = 0.0
        self.policy = {s: "UP" for s in NONTERMINAL_STATES}
        self.outer_iter = 0
        self.eval_sweep = 0
        self.history_delta.clear()
        self.improve_markers.clear()
        self.stable = False
        self.last_changed = None


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

    for r in range(N_ROWS):
        for c in range(N_COLS):
            rc = (r, c)
            x, y = c, N_ROWS - 1 - r

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

            action = trainer.policy[rc]
            dr, dc = ACTION_DELTA[action]
            ax.arrow(x + 0.5, y + 0.68, dc * 0.2, -dr * 0.2,
                      head_width=0.1, head_length=0.1,
                      fc="#111111", ec="#111111", linewidth=1.3)

            if rc == START:
                ax.plot(x + 0.9, y + 0.1, marker="*", color="#1565c0",
                         markersize=9, zorder=6)

    status = "STABLE (optimal)" if trainer.stable else "changing"
    ax.set_title(
        f"Outer iter {trainer.outer_iter}  |  eval sweep {trainer.eval_sweep}  |  "
        f"policy: {status}",
        fontsize=10,
    )


def draw_curve(ax, trainer):
    ax.clear()
    if trainer.history_delta:
        ax.plot(trainer.history_delta, color="#1565c0", label="policy evaluation |delta V|")
        for i, marker in enumerate(trainer.improve_markers):
            ax.axvline(marker - 0.5, color="#8e24aa", linestyle=":", linewidth=1,
                        label="policy improvement" if i == 0 else None)
        ax.axhline(THETA, color="#e65100", linestyle="--", linewidth=1,
                    label=f"convergence threshold ({THETA})")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("evaluation sweep (cumulative)")
    ax.set_ylabel("max |delta V|")
    ax.set_title("Policy evaluation convergence", fontsize=11)


def run_headless(trainer, out_path):
    import matplotlib.pyplot as plt

    trainer.run_to_convergence()

    fig, (ax_grid, ax_curve) = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_grid(ax_grid, trainer)
    draw_curve(ax_curve, trainer)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Converged after {trainer.outer_iter} policy-improvement iterations "
          f"({len(trainer.history_delta)} total evaluation sweeps).")
    print(f"Saved figure to {out_path}")


def run_interactive(trainer):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(11, 5.7))
    ax_grid = fig.add_axes([0.05, 0.18, 0.42, 0.72])
    ax_curve = fig.add_axes([0.55, 0.18, 0.40, 0.72])

    def redraw():
        draw_grid(ax_grid, trainer)
        draw_curve(ax_curve, trainer)
        fig.canvas.draw_idle()

    def on_eval(_event):
        trainer.evaluate_sweep()
        redraw()

    def on_improve(_event):
        trainer.improve_policy()
        redraw()

    def on_run(_event):
        trainer.run_to_convergence()
        redraw()

    def on_reset(_event):
        trainer.reset()
        redraw()

    ax_eval = fig.add_axes([0.05, 0.03, 0.20, 0.06])
    ax_improve = fig.add_axes([0.27, 0.03, 0.20, 0.06])
    ax_run = fig.add_axes([0.49, 0.03, 0.24, 0.06])
    ax_reset = fig.add_axes([0.75, 0.03, 0.14, 0.06])
    btn_eval = Button(ax_eval, "Evaluate Sweep")
    btn_improve = Button(ax_improve, "Improve Policy")
    btn_run = Button(ax_run, "Run to Convergence")
    btn_reset = Button(ax_reset, "Reset")
    btn_eval.on_clicked(on_eval)
    btn_improve.on_clicked(on_improve)
    btn_run.on_clicked(on_run)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Policy Iteration (model-based planning) on the 4x3 AIMA gridworld"
    )
    parser.add_argument("--headless", action="store_true",
                         help="run silently to convergence and save a PNG instead of "
                              "opening a window")
    parser.add_argument("--out", type=str, default="policy_iteration_result.png")
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    trainer = PolicyIterationTrainer()

    if args.headless:
        run_headless(trainer, args.out)
    else:
        run_interactive(trainer)


if __name__ == "__main__":
    main()