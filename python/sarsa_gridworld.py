"""
sarsa_gridworld.py
===================

Tabular SARSA on the classic 4x3 gridworld -- the same "Example of Learned
Q-Function" environment used in python/q_learning_gridworld.py and
python/cross_entropy_method.py (Berkeley CS188 / Russell & Norvig AIMA), so
results are directly comparable to both of those demos.

SARSA ("State-Action-Reward-State-Action") looks almost identical to
Q-learning on the page -- both learn a Q(s,a) table from single transitions
via temporal-difference updates -- but with one crucial difference: SARSA is
**on-policy**. Its update target uses the value of the action the agent
*actually takes next* (sampled from its own epsilon-greedy behavior policy),
not the greedy max-over-actions value Q-learning uses. That one substitution
changes what "optimal" even means: SARSA learns the best policy *given that
it keeps exploring*, which can differ from the truly optimal policy whenever
exploration itself carries risk (the classic illustration is the "Cliff
Walking" example in Sutton & Barto, Ch. 6).

See ALGORITHMS.md for the full derivation, formulae, and a three-way
comparison with Tabular Q-Learning and the Cross-Entropy Method.

Usage:
    python sarsa_gridworld.py                       # interactive matplotlib GUI
    python sarsa_gridworld.py --headless --episodes 1000
"""

import argparse

import numpy as np
import matplotlib

# ----------------------------------------------------------------------
# Environment: classic 4x3 AIMA / CS188 gridworld
# (same layout/rewards/discount as q_learning_gridworld.py and
#  cross_entropy_method.py)
# ----------------------------------------------------------------------
N_ROWS, N_COLS = 3, 4
WALL = (1, 1)
GOAL = (0, 3)
PIT = (1, 3)
START = (2, 0)
LIVING_REWARD = -0.04
GAMMA = 0.9
TRANSITION_NOISE = 0.8  # probability of moving in the intended direction
MAX_STEPS_PER_EPISODE = 200

ALPHA = 0.1     # learning rate
EPSILON = 0.1   # epsilon-greedy exploration rate

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


def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    qs = Q[STATE_IDX[state]]
    best_q = np.max(qs)
    best_actions = [a for a, q in zip(ACTIONS, qs) if q == best_q]
    return rng.choice(best_actions)


# ----------------------------------------------------------------------
# SARSA
# ----------------------------------------------------------------------
class SARSATrainer:
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.state = START
        self.action = epsilon_greedy(self.Q, self.state, self.epsilon, self.rng)
        self.episode = 0
        self.step_count = 0
        self.last_update_str = ""

    def next_step(self):
        """One SARSA update: take the current (s,a), observe (r, s'), choose
        a' from s' via epsilon-greedy, update Q(s,a) using Q(s',a') -- the
        value of the action actually about to be taken -- then advance."""
        s, a = self.state, self.action
        s_next, r, done = env_step(s, a, self.rng)
        a_next = None if done else epsilon_greedy(self.Q, s_next, self.epsilon, self.rng)

        s_idx, a_idx = STATE_IDX[s], ACTIONS.index(a)
        q_sa = self.Q[s_idx, a_idx]
        q_next = 0.0 if done else self.Q[STATE_IDX[s_next], ACTIONS.index(a_next)]
        td_error = r + self.gamma * q_next - q_sa
        new_q = q_sa + self.alpha * td_error

        self.last_update_str = (
            f"Q({s},{a}) <- {q_sa:.3f} + {self.alpha} * "
            f"[ {r:+.2f} + {self.gamma} * {q_next:.3f} - {q_sa:.3f} ]\n"
            f"Q({s},{a}) <- {q_sa:.3f} + {self.alpha} * ({td_error:+.3f})  =  {new_q:.3f}"
        )
        self.Q[s_idx, a_idx] = new_q
        self.step_count += 1

        if done:
            self.episode += 1
            self.state = START
            self.action = epsilon_greedy(self.Q, self.state, self.epsilon, self.rng)
        else:
            self.state, self.action = s_next, a_next

    def train_episodes(self, n):
        target = self.episode + n
        guard = 0
        max_guard = n * MAX_STEPS_PER_EPISODE * 5
        while self.episode < target and guard < max_guard:
            self.next_step()
            guard += 1

    def reset(self):
        self.Q[:] = 0.0
        self.state = START
        self.action = epsilon_greedy(self.Q, self.state, self.epsilon, self.rng)
        self.episode = 0
        self.step_count = 0
        self.last_update_str = ""

    def greedy_policy(self):
        return {s: ACTIONS[int(np.argmax(self.Q[STATE_IDX[s]]))] for s in STATES}


# ----------------------------------------------------------------------
# Visualization -- four Q-value triangles per cell (same idea as
# q_learning_gridworld.py, so the two are visually comparable side by side)
# ----------------------------------------------------------------------
def draw_grid(ax, trainer):
    from matplotlib.patches import Rectangle, Polygon
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

    tri_offsets = {
        "UP": [(0, 1), (1, 1), (0.5, 0.5)],
        "DOWN": [(0, 0), (1, 0), (0.5, 0.5)],
        "LEFT": [(0, 0), (0, 1), (0.5, 0.5)],
        "RIGHT": [(1, 0), (1, 1), (0.5, 0.5)],
    }
    text_offsets = {
        "UP": (0.5, 0.78),
        "DOWN": (0.5, 0.22),
        "LEFT": (0.22, 0.5),
        "RIGHT": (0.78, 0.5),
    }

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

            qs = trainer.Q[STATE_IDX[rc]]
            best_a = policy[rc]
            for a_idx, a in enumerate(ACTIONS):
                q_val = qs[a_idx]
                pts = [(x + dx, y + dy) for dx, dy in tri_offsets[a]]
                color = cmap(norm(q_val))
                lw = 2.2 if a == best_a else 0.5
                ax.add_patch(Polygon(pts, facecolor=color, edgecolor="black", linewidth=lw))
                tx, ty = x + text_offsets[a][0], y + text_offsets[a][1]
                ax.text(tx, ty, f"{q_val:.2f}", ha="center", va="center", fontsize=6.5)

            ax.text(x + 0.04, y + 0.96, f"{r},{c}", fontsize=6,
                     color="black", va="top", ha="left", alpha=0.6)

            if rc == trainer.state:
                ax.plot(x + 0.5, y + 0.5, "o", color="#1565c0", markersize=9, zorder=6)
            if rc == START:
                ax.plot(x + 0.1, y + 0.1, marker="*", color="#e65100",
                         markersize=8, zorder=6)

    ax.set_title(f"Episode {trainer.episode}  |  step {trainer.step_count}", fontsize=11)


def run_headless(trainer, episodes, out_path):
    import matplotlib.pyplot as plt

    trainer.train_episodes(episodes)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    draw_grid(ax, trainer)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Ran {trainer.episode} episodes ({trainer.step_count} steps).")
    print(f"Saved figure to {out_path}")
    print(trainer.last_update_str)


def run_interactive(trainer):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(7, 7.2))
    ax_grid = fig.add_axes([0.08, 0.22, 0.86, 0.72])
    ax_text = fig.add_axes([0.05, 0.10, 0.9, 0.08])
    ax_text.axis("off")
    update_text = ax_text.text(0.5, 0.5, "", ha="center", va="center",
                                fontsize=9, family="monospace")

    def redraw():
        draw_grid(ax_grid, trainer)
        update_text.set_text(trainer.last_update_str)
        fig.canvas.draw_idle()

    def on_next(_event):
        trainer.next_step()
        redraw()

    def on_train(_event):
        trainer.train_episodes(1000)
        redraw()

    def on_reset(_event):
        trainer.reset()
        redraw()

    ax_next = fig.add_axes([0.08, 0.02, 0.22, 0.06])
    ax_train = fig.add_axes([0.36, 0.02, 0.30, 0.06])
    ax_reset = fig.add_axes([0.72, 0.02, 0.20, 0.06])
    btn_next = Button(ax_next, "Next Step")
    btn_train = Button(ax_train, "Train 1000 Episodes")
    btn_reset = Button(ax_reset, "Reset")
    btn_next.on_clicked(on_next)
    btn_train.on_clicked(on_train)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Tabular SARSA (on-policy TD control) on the 4x3 AIMA gridworld"
    )
    parser.add_argument("--headless", action="store_true",
                         help="train silently and save a PNG instead of opening a window")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="sarsa_result.png")
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    trainer = SARSATrainer(alpha=args.alpha, epsilon=args.epsilon, seed=args.seed)

    if args.headless:
        run_headless(trainer, args.episodes, args.out)
    else:
        run_interactive(trainer)


if __name__ == "__main__":
    main()
