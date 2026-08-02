r"""
Q-Learning on a 4x3 GridWorld
==============================

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
control) -- NOT value iteration or Q-iteration. The difference matters:
value/Q-iteration (covered in the companion HTML guide) assume you already
KNOW the reward function and transition model and sweep over every state.
Q-learning assumes you know neither -- the agent has to walk around,
actually experience transitions, and learn Q(s,a) purely from that
experience. That's why the numbers below only update for states the agent
actually visits, and why the slide this script recreates is captioned
"Q-VALUES AFTER 1000 EPISODES" rather than "after N sweeps."

The core update, applied after every single step the agent takes:

    Q(s,a)  <-  Q(s,a)  +  alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
                                     \_______ TD target _______/
                                     \_____________ TD error _____________/

    (the TD error is the target MINUS the old Q(s,a) -- it's the wider span,
    not the same span as the target)

    alpha  = learning rate      (how much we trust each new sample)
    gamma  = discount factor    (how much future reward matters)
    r      = reward just received
    s'     = the state we landed in
    max_a' Q(s',a') = our current best guess at how good s' is

Run it directly:  python q_learning_gridworld.py
No dependencies beyond the standard library.
"""

import random

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
    deterministic here, same as the rest of this guide, so every number
    stays traceable by hand.)
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
    """The Q-learning update -- the line of code the equation above compiles
    down to. Returns the pieces so callers can print them out step by step."""
    old_q = Q[s][a]
    best_next = 0.0 if is_terminal(s_next) else max(Q[s_next].values())
    td_target = r + gamma * best_next
    td_error = td_target - old_q
    Q[s][a] = old_q + alpha * td_error
    return old_q, best_next, td_error, Q[s][a]


# ---------------------------------------------------------------------------
# 3. ASCII RENDERING (mirrors the 4-numbers-per-cell layout in the slide)
# ---------------------------------------------------------------------------

def render_grid(Q, agent=None, title=""):
    """Each cell is drawn as a small cross: UP on top, LEFT/RIGHT on the
    sides, DOWN on the bottom -- the same "one Q-value per action" idea as
    the triangles in the original slide, laid out in plain text."""
    if title:
        print(f"\n{title}")
    width = 13
    for r in range(ROWS):
        top, mid, bot = [], [], []
        for c in range(COLS):
            s = (r, c)
            if s == WALL:
                top.append(" " * width)
                mid.append("WALL".center(width))
                bot.append(" " * width)
            elif s in TERMINALS:
                top.append(" " * width)
                mid.append(f"{TERMINALS[s]:+.2f}".center(width))
                bot.append(" " * width)
            else:
                q = Q[s]
                marker = "@" if agent == s else " "
                top.append(f"{q['UP']:>5.2f}".center(width))
                mid.append(f"{q['LEFT']:>5.2f}{marker}{q['RIGHT']:>5.2f}".center(width))
                bot.append(f"{q['DOWN']:>5.2f}".center(width))
        sep = "+" + "+".join("-" * width for _ in range(COLS)) + "+"
        print(sep)
        print("|" + "|".join(top) + "|")
        print("|" + "|".join(mid) + "|")
        print("|" + "|".join(bot) + "|")
    print("+" + "+".join("-" * width for _ in range(COLS)) + "+")


# ---------------------------------------------------------------------------
# 4. INTERACTIVE TRACE -- watch the very first episode, step by step,
#    starting from an all-zero Q-table.
# ---------------------------------------------------------------------------

def interactive_episode(Q, max_steps=20):
    print("\n" + "=" * 70)
    print("INTERACTIVE TRACE -- episode 1, starting from an all-zero Q-table")
    print("=" * 70)
    s = START
    render_grid(Q, agent=s, title="Initial state (everything is still 0):")

    for step_num in range(1, max_steps + 1):
        if is_terminal(s):
            print(f"\nReached terminal state {s} (reward {TERMINALS[s]:+.2f}). Episode over.")
            break

        input("\nPress Enter to take the next step...")

        q_before = dict(Q[s])
        a, mode = choose_action(Q, s, EPSILON)
        s_next, r = step(s, a)
        old_q, best_next, td_error, new_q = q_update(Q, s, a, r, s_next)

        print(f"\nStep {step_num}: agent is at {s}")
        print("  Q(s,.) before update: " + ", ".join(f"{act}={q_before[act]:.3f}" for act in ACTIONS))
        print(f"  Action chosen: {a}  ({mode}, epsilon={EPSILON})")
        print(f"  -> moved to {s_next}, reward = {r:+.2f}")
        print("  Q-learning update:")
        print(f"    Q({s},{a}) <- {old_q:.3f} + {ALPHA} * [ {r:+.2f} + {GAMMA} * {best_next:.3f} - {old_q:.3f} ]")
        print(f"    Q({s},{a}) <- {old_q:.3f} + {ALPHA} * ({td_error:+.3f})")
        print(f"    Q({s},{a}) <- {new_q:.3f}")

        render_grid(Q, agent=s_next, title=f"Grid after this update (agent now at {s_next}):")
        s = s_next

    if not is_terminal(s):
        print(f"\nStopped after {max_steps} steps without reaching a terminal state.")


# ---------------------------------------------------------------------------
# 5. FULL TRAINING RUN -- silently run many episodes, like the slide's
#    "Q-VALUES AFTER 1000 EPISODES".
# ---------------------------------------------------------------------------

def train(Q, num_episodes=1000, max_steps_per_episode=100):
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


def greedy_policy(Q):
    """The policy implied by the learned Q-table: best action per state."""
    arrow = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}
    print("\nGreedy policy read off the learned Q-table:")
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
        print("  " + "  ".join(row_str))


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(0)  # reproducible trace for the walkthrough
    Q = new_q_table()

    interactive_episode(Q)

    proceed = input("\nRun 1000 training episodes now and see the final Q-values? [Y/n] ")
    if proceed.strip().lower() != "n":
        Q = new_q_table()  # start fresh, same as the slide's "after 1000 episodes"
        random.seed(1)
        train(Q, num_episodes=1000)
        render_grid(Q, title="Q-VALUES AFTER 1000 EPISODES")
        greedy_policy(Q)
