r"""
Q-Learning on a 4x3 GridWorld -- matplotlib edition
=====================================================

Recreates the classic gridworld from the "Example of Learned Q-Function"
slide (based on Dan Klein's CS188 slides / the AIMA 4x3 gridworld):

    +---------+---------+---------+---------+
    |         |         |         |         |
    |  (0,0)  |  (0,1)  |  (0,2)  | (0,3)=+1|
    |         |         |         | GOAL    |
    +---------+---------+---------+---------+
    |         |         |         |         |
    |  (1,0)  |  WALL   |  (1,2)  | (1,3)=-1|
    |         |         |         | PIT     |
    +---------+---------+---------+---------+
    |         |         |         |         |
    |  (2,0)  |  (2,1)  |  (2,2)  |  (2,3)  |
    |  START  |         |         |         |
    +---------+---------+---------+---------+

Every non-terminal move costs a small "living reward" (-0.04), so the agent
is encouraged to reach the +1 goal quickly and avoid the -1 pit entirely.

This script implements tabular Q-LEARNING (model-free, off-policy TD
control) -- NOT value iteration or Q-iteration. Value/Q-iteration (covered
in the companion HTML guide) assume you already KNOW the reward function
and transition model and sweep over every state. Q-learning assumes you
know neither -- the agent has to walk around, actually experience
transitions, and learn Q(s,a) purely from that experience.

The core update, applied after every single step the agent takes:

    Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
                                     \_______ TD target _______/
                                     \_____________ TD error _____________/

    alpha  = learning rate      (how much we trust each new sample)
    gamma  = discount factor    (how much future reward matters)
    r      = reward just received
    s'     = the state we landed in
    max_a' Q(s',a') = our current best guess at how good s' is

    (the TD error is the target MINUS the old Q(s,a) -- it's the wider
    span, not the same span as the target)

HOW TO RUN
----------
Interactive GUI (opens a matplotlib window with click-to-step buttons):

    python q_learning_gridworld.py

Headless mode (no window -- trains silently and saves a PNG; this is what
CI uses to smoke-test the script on machines with no display):

    python q_learning_gridworld.py --headless
    python q_learning_gridworld.py --headless --episodes 500

Dependency: matplotlib (see requirements.txt). Everything else is
standard library.
"""

import argparse
import os
import random
import sys

# ---------------------------------------------------------------------------
# Pick a matplotlib backend BEFORE importing pyplot. CI runners (and any
# machine with no display) can't open a GUI window, so we fall back to the
# non-interactive "Agg" backend automatically -- this is what lets
# `--headless` (and CI) work on a machine with no screen at all.
# ---------------------------------------------------------------------------
HEADLESS = ("--headless" in sys.argv) or (
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
)

import matplotlib  # noqa: E402

if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

try:
    CMAP = matplotlib.colormaps["RdYlGn"]
except AttributeError:  # older matplotlib
    CMAP = plt.get_cmap("RdYlGn")

NORM = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
AGENT_COLOR = "#eb6834"  # matches the accent color used throughout the HTML guide

# ---------------------------------------------------------------------------
# 1. GRID DEFINITION
# ---------------------------------------------------------------------------

ROWS, COLS = 3, 4
WALL = (1, 1)
TERMINALS = {(0, 3): 1.0, (1, 3): -1.0}   # state -> terminal reward
START = (2, 0)
LIVING_REWARD = -0.04

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

ALPHA = 0.1     # learning rate
GAMMA = 0.9     # discount factor
EPSILON = 0.1   # exploration rate for epsilon-greedy


def all_states():
    """Every non-wall cell, including terminals."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != WALL]


def is_terminal(s):
    return s in TERMINALS


def step(s, a):
    """
    Deterministic transition: moving into a wall or off the grid just
    leaves the agent where it was. (The original gridworld is stochastic --
    80% intended direction, 10%/10% sideways slip -- but we simplify to
    deterministic here so every number stays traceable by hand.)
    Returns (next_state, reward).
    """
    if is_terminal(s):
        raise ValueError("Can't step from a terminal state.")
    dr, dc = DELTA[a]
    nr, nc = s[0] + dr, s[1] + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) == WALL:
        nr, nc = s  # bump into wall/edge -> stay put
    next_state = (nr, nc)
    reward = TERMINALS[next_state] if next_state in TERMINALS else LIVING_REWARD
    return next_state, reward


# ---------------------------------------------------------------------------
# 2. Q-TABLE
# ---------------------------------------------------------------------------

def new_q_table():
    """Every Q(s,a) starts at exactly 0 -- the agent knows nothing yet."""
    return {s: {a: 0.0 for a in ACTIONS} for s in all_states()}


def best_action(Q, s):
    """argmax_a Q(s,a), breaking ties randomly so early all-zero states
    don't always default to the same action."""
    q_s = Q[s]
    best_val = max(q_s.values())
    best_actions = [a for a, v in q_s.items() if v == best_val]
    return random.choice(best_actions)


def choose_action(Q, s, epsilon):
    """Epsilon-greedy: explore with probability epsilon, else exploit."""
    if random.random() < epsilon:
        return random.choice(ACTIONS), "explore"
    return best_action(Q, s), "exploit"


def q_update(Q, s, a, r, s_next, alpha=ALPHA, gamma=GAMMA):
    """The Q-learning update -- the line of code the equation up top compiles
    down to. Returns the pieces so callers can display them step by step."""
    old_q = Q[s][a]
    best_next = 0.0 if is_terminal(s_next) else max(Q[s_next].values())
    td_target = r + gamma * best_next
    td_error = td_target - old_q
    Q[s][a] = old_q + alpha * td_error
    return old_q, best_next, td_error, Q[s][a]


def train(Q, num_episodes=1000, max_steps_per_episode=100):
    """Silently run many episodes, like the slide's "Q-VALUES AFTER 1000 EPISODES"."""
    for _ in range(num_episodes):
        s = START
        for _ in range(max_steps_per_episode):
            if is_terminal(s):
                break
            a, _ = choose_action(Q, s, EPSILON)
            s_next, r = step(s, a)
            q_update(Q, s, a, r, s_next)
            s = s_next
    return Q


def greedy_policy_text(Q):
    """The policy implied by the learned Q-table, as a small text grid."""
    arrow = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}
    lines = []
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            s = (r, c)
            if s == WALL:
                row_str.append("#")
            elif s in TERMINALS:
                row_str.append("+" if TERMINALS[s] > 0 else "-")
            else:
                row_str.append(arrow[best_action(Q, s)])
        lines.append("  " + "  ".join(row_str))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. MATPLOTLIB RENDERING
#    Each cell is split into 4 triangles (up/down/left/right), colored and
#    labeled by that action's Q-value -- the same layout as the original
#    slide. Every cell also shows its (row, col) position in the corner.
# ---------------------------------------------------------------------------

def q_color(value):
    clipped = max(-1.0, min(1.0, value))
    return CMAP(NORM(clipped))


def draw_grid(ax, Q, agent=None, title=""):
    ax.clear()

    for r in range(ROWS):
        y = ROWS - 1 - r  # row 0 drawn at the top
        for c in range(COLS):
            s = (r, c)
            x = c

            if s == WALL:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#3a3a3a", edgecolor="white"))
                ax.text(x + 0.5, y + 0.5, "WALL", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

            elif s in TERMINALS:
                val = TERMINALS[s]
                color = "#2ecc71" if val > 0 else "#e74c3c"
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white"))
                ax.text(x + 0.5, y + 0.5, f"{val:+.2f}", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")

            else:
                q = Q[s]
                cx, cy = x + 0.5, y + 0.5
                triangles = {
                    "UP":    [(cx, cy), (x, y + 1), (x + 1, y + 1)],
                    "RIGHT": [(cx, cy), (x + 1, y + 1), (x + 1, y)],
                    "DOWN":  [(cx, cy), (x + 1, y), (x, y)],
                    "LEFT":  [(cx, cy), (x, y), (x, y + 1)],
                }
                label_pos = {
                    "UP": (cx, y + 0.78),
                    "RIGHT": (x + 0.78, cy),
                    "DOWN": (cx, y + 0.22),
                    "LEFT": (x + 0.22, cy),
                }
                is_best = {a: (q[a] == max(q.values())) for a in ACTIONS}
                for a in ACTIONS:
                    ax.add_patch(Polygon(triangles[a], closed=True,
                                          facecolor=q_color(q[a]),
                                          edgecolor="white", linewidth=0.8))
                    ax.text(*label_pos[a], f"{q[a]:.2f}", ha="center", va="center",
                            fontsize=8.5, fontweight="bold" if is_best[a] else "normal")
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor="white", linewidth=1.4))

            # cell position label -- every cell, including WALL/terminals
            ax.text(x + 0.04, y + 0.965, f"({r},{c})", ha="left", va="top",
                    fontsize=6.5, color="#666666")

    if agent is not None:
        ar, ac = agent
        ay = ROWS - 1 - ar
        ax.add_patch(Circle((ac + 0.5, ay + 0.5), 0.13, facecolor=AGENT_COLOR,
                             edgecolor="white", linewidth=1.5, zorder=5))

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


# ---------------------------------------------------------------------------
# 4. INTERACTIVE GUI -- click "Next Step" to watch Q-learning happen one
#    update at a time, starting from an all-zero Q-table.
# ---------------------------------------------------------------------------

class GridWorldApp:
    def __init__(self, episodes_per_train=1000):
        self.episodes_per_train = episodes_per_train
        self.Q = new_q_table()
        self.state = START
        self.episode_num = 1
        self.step_num = 0

        self.fig, self.ax = plt.subplots(figsize=(7.6, 7.6))
        plt.subplots_adjust(bottom=0.32, top=0.90)

        self.status = self.fig.text(
            0.06, 0.235, "", ha="left", va="top", fontsize=8.5,
            family="monospace",
        )

        ax_next = self.fig.add_axes([0.06, 0.03, 0.26, 0.07])
        ax_train = self.fig.add_axes([0.37, 0.03, 0.30, 0.07])
        ax_reset = self.fig.add_axes([0.71, 0.03, 0.23, 0.07])
        self.btn_next = Button(ax_next, "Next Step")
        self.btn_train = Button(ax_train, f"Train {episodes_per_train} Episodes")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_next.on_clicked(self.on_next)
        self.btn_train.on_clicked(self.on_train)
        self.btn_reset.on_clicked(self.on_reset)

        self.redraw("Click 'Next Step' to take the first action from an all-zero Q-table.")

    def redraw(self, message):
        title = f"Episode {self.episode_num}, step {self.step_num} \u2014 agent at {self.state}"
        draw_grid(self.ax, self.Q, agent=self.state, title=title)
        self.status.set_text(message)
        self.fig.canvas.draw_idle()

    def on_next(self, event):
        s = self.state
        if is_terminal(s):
            self.episode_num += 1
            self.step_num = 0
            self.state = START
            self.redraw(f"Reached a terminal state (reward {TERMINALS[s]:+.2f}).\n"
                        f"Starting episode {self.episode_num} from {START}.")
            return

        self.step_num += 1
        q_before = dict(self.Q[s])
        a, mode = choose_action(self.Q, s, EPSILON)
        s_next, r = step(s, a)
        old_q, best_next, td_error, new_q = q_update(self.Q, s, a, r, s_next)

        msg = (
            f"State {s}  ->  action {a}  ({mode}, epsilon={EPSILON})\n"
            f"Q(s,.) before: " + ", ".join(f"{act}={q_before[act]:.3f}" for act in ACTIONS) + "\n"
            f"Moved to {s_next}, reward = {r:+.2f}\n"
            f"Q({s},{a}) <- {old_q:.3f} + {ALPHA} * [ {r:+.2f} + {GAMMA} * {best_next:.3f} - {old_q:.3f} ]\n"
            f"Q({s},{a}) <- {old_q:.3f} + {ALPHA} * ({td_error:+.3f})  =  {new_q:.3f}"
        )
        self.state = s_next
        self.redraw(msg)

    def on_train(self, event):
        self.Q = new_q_table()
        train(self.Q, num_episodes=self.episodes_per_train)
        self.state = START
        self.episode_num = 1
        self.step_num = 0
        self.redraw(f"Trained fresh for {self.episodes_per_train} episodes.\n"
                    f"This is the converged Q-table and its greedy policy\n"
                    f"(the bold value in each cell is the best action).")

    def on_reset(self, event):
        self.Q = new_q_table()
        self.state = START
        self.episode_num = 1
        self.step_num = 0
        self.redraw("Reset to an all-zero Q-table. Click 'Next Step' to begin again.")

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# 5. HEADLESS MODE -- no window, used by CI / anyone without a display.
#    Trains silently and saves a PNG snapshot of the final Q-values instead.
# ---------------------------------------------------------------------------

def headless_run(episodes):
    Q = new_q_table()
    train(Q, num_episodes=episodes)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    draw_grid(ax, Q, agent=None, title=f"Q-VALUES AFTER {episodes} EPISODES (headless)")
    out_path = "q_values_headless.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")

    print(f"[headless] Trained {episodes} episodes with no GUI (Agg backend).")
    print(f"[headless] Saved final Q-value grid to: {out_path}")
    print("[headless] Greedy policy read off the learned Q-table:")
    print(greedy_policy_text(Q))


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-learning on a 4x3 gridworld, with a matplotlib GUI."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Skip the GUI: train silently and save a PNG instead (used by CI).",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Episodes to train, either in headless mode or via the Train button (default: 1000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(0)

    if args.headless or HEADLESS:
        headless_run(args.episodes)
    else:
        app = GridWorldApp(episodes_per_train=args.episodes)
        app.run()
