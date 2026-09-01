"""Algorithm 1: Tabular Q-Learning policy for the inventory-control project.

Improvements over the original version:
  1. Compressed state representation (26 -> 14 dims) to fight the curse
     of dimensionality given a fixed episode budget.
  2. Coarser action grid (11 -> 6 levels/product, 1331 -> 216 joint actions)
     so the table converges with fewer visits per action.
  3. Visit-count-based learning rate (alpha = 1 / (1 + N(s,a))) instead of
     a fixed alpha, so rare pairs move fast and common pairs stabilize.
  4. Optimistic Q-value initialization to encourage trying under-explored
     actions instead of relying purely on epsilon-greedy noise.
  5. A heuristic order-up-to-par fallback for states never seen in
     training, instead of always returning [0, 0, 0] (which guarantees
     stockouts at inference time on the many unseen states).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np



from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROLL_NUMBER = "DA25G504"

MODEL_PATH = Path(__file__).with_name("q_table.npz")

# Coarser action grid: 6 levels per product instead of 11.
# Quantities become 0, 20, 40, 60, 80, 100.
N_ACTIONS_PER_PRODUCT = 6
ACTION_STEP = 20
ACTION_VALUES = np.arange(0, 101, ACTION_STEP, dtype=np.int16)

N_PRODUCTS = 3

# Optimistic initial Q-value. Set this a bit above the reward scale you
# expect from a good episode-step so unexplored actions get sampled.
# Tune this against your reward function's typical magnitude.
Q_INIT_VALUE = 5.0

# Order-up-to-par target used only as a fallback for unseen states.
FALLBACK_PAR_LEVEL = 50


# ---------------------------------------------------------------------
# State preprocessing
# ---------------------------------------------------------------------

def _bucket_inventory(x: float) -> int:
    """Bucket inventory into 20-unit intervals."""
    return int(np.clip(np.floor(float(x) / 20.0), 0, 10))


def _bucket_pipeline_total(x: float) -> int:
    """Bucket total incoming pipeline quantity into coarse levels."""
    return int(np.clip(np.floor(float(x) / 40.0), 0, 6))


def _bucket_demand_mean(x: float) -> int:
    """Bucket mean recent demand into coarse demand levels."""
    return int(np.clip(np.floor(float(x) / 10.0), 0, 10))


def _bucket_capacity(x: float) -> int:
    """Bucket capacity utilisation."""
    return int(np.clip(np.floor(float(x) * 10.0), 0, 10))


def _trend_bucket(recent: np.ndarray) -> int:
    """Coarse trend indicator: -1 falling, 0 flat, +1 rising -> {0,1,2}."""
    if len(recent) < 2:
        return 1
    slope = recent[-1] - recent[0]
    if slope > 5:
        return 2
    if slope < -5:
        return 0
    return 1


def observation_to_state(observation: dict[str, Any]) -> tuple[int, ...]:
    """Convert the official observation dictionary into a compressed
    tabular state.

    Compared to a naive discretization of every raw field, this keeps
    only the signal that matters for reorder decisions:
      - current inventory per product (3 dims)
      - total incoming pipeline per product, rather than a per-lead-day
        breakdown (3 dims instead of 12)
      - recent demand mean and trend per product, rather than every raw
        daily value (6 dims instead of 9)
      - day-of-week seasonality (1 dim)
      - warehouse utilisation (1 dim)

    Total: 14 dims instead of 26. The environment observation itself is
    not modified; only the internal Q-table representation changes.
    """
    inventory = np.asarray(
        observation["inventory"], dtype=np.float32
    ).reshape(3)

    pipeline = np.asarray(
        observation["arrival_pipeline"], dtype=np.float32
    ).reshape(3, 4)

    demand_history = np.asarray(
        observation["demand_history"], dtype=np.float32
    ).reshape(7, 3)

    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    capacity_utilisation = float(
        np.asarray(observation["capacity_utilisation"]).reshape(-1)[0]
    )

    state = []

    # Current inventory.
    for value in inventory:
        state.append(_bucket_inventory(value))

    # Outstanding pipeline, aggregated to a single total per product
    # instead of one bucket per lead day. The exact arrival day matters
    # less than "how much is coming soon" for a reorder decision.
    pipeline_totals = pipeline.sum(axis=1)
    for total in pipeline_totals:
        state.append(_bucket_pipeline_total(total))

    # Demand history: summarise the last 3 days per product as
    # (mean, trend) instead of retaining every raw value.
    recent_history = demand_history[-3:]  # shape (3 days, 3 products)
    for product in range(N_PRODUCTS):
        product_series = recent_history[:, product]
        state.append(_bucket_demand_mean(float(product_series.mean())))
        state.append(_trend_bucket(product_series))

    # Current day modulo one week captures seasonal position without
    # making the table unnecessarily large.
    state.append(day % 7)

    # Warehouse utilisation.
    state.append(_bucket_capacity(capacity_utilisation))

    return tuple(state)


# ---------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------

def action_to_index(action: list[int] | tuple[int, int, int]) -> int:
    """Encode three per-product action indices into one table index."""
    a1, a2, a3 = [int(x) for x in action]

    return (
        a1 * N_ACTIONS_PER_PRODUCT * N_ACTIONS_PER_PRODUCT
        + a2 * N_ACTIONS_PER_PRODUCT
        + a3
    )


def index_to_action(index: int) -> np.ndarray:
    """Decode one table action index into three environment action indices."""
    index = int(index)

    a1 = index // (N_ACTIONS_PER_PRODUCT * N_ACTIONS_PER_PRODUCT)
    remainder = index % (N_ACTIONS_PER_PRODUCT * N_ACTIONS_PER_PRODUCT)

    a2 = remainder // N_ACTIONS_PER_PRODUCT
    a3 = remainder % N_ACTIONS_PER_PRODUCT

    return np.asarray([a1, a2, a3], dtype=np.int64)


N_DISCRETE_ACTIONS = N_ACTIONS_PER_PRODUCT ** N_PRODUCTS


# ---------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------

Q_TABLE: dict[tuple[int, ...], np.ndarray] = {}

# Visit counts per (state, action) pair, used to compute an adaptive
# learning rate instead of a fixed alpha.
VISIT_COUNTS: dict[tuple[int, ...], np.ndarray] = {}


def _get_q_values(state: tuple[int, ...]) -> np.ndarray:
    """Return the Q-values for a state, creating them (optimistically
    initialized) if necessary."""
    if state not in Q_TABLE:
        Q_TABLE[state] = np.full(
            N_DISCRETE_ACTIONS, Q_INIT_VALUE, dtype=np.float32
        )
        VISIT_COUNTS[state] = np.zeros(N_DISCRETE_ACTIONS, dtype=np.int32)

    return Q_TABLE[state]


# ---------------------------------------------------------------------
# Fallback heuristic for unseen states
# ---------------------------------------------------------------------

def _fallback_action(observation: dict[str, Any]) -> list[int]:
    """Simple order-up-to-par heuristic used only when a state has never
    been visited during training.

    This is far more sensible than always returning [0, 0, 0]: with a
    large state space, unseen states are common at inference time, and
    never reordering guarantees stockouts. Order enough to bring each
    product's inventory up toward FALLBACK_PAR_LEVEL, rounded to the
    nearest available action level.
    """
    inventory = np.asarray(
        observation["inventory"], dtype=np.float32
    ).reshape(3)

    quantities = []
    for level in inventory:
        gap = max(0.0, FALLBACK_PAR_LEVEL - float(level))
        # Round down to the nearest available action step.
        rounded = int(gap // ACTION_STEP) * ACTION_STEP
        rounded = min(rounded, ACTION_VALUES[-1])
        quantities.append(int(rounded))

    return quantities


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_q_learning(
    student_config: dict[str, Any],
    episodes: int = 20000,
    alpha_min: float = 0.02,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    seed: int = 2026,
) -> dict[tuple[int, ...], np.ndarray]:
    """Train Algorithm 1 using tabular Q-learning with an adaptive
    learning rate.

    Note: episodes was raised from 5000 to 20000 and epsilon_decay
    loosened from 0.999 to 0.9995 to keep exploration going longer,
    since the (still large) state space needs more visits per state to
    converge. Tune both against your time budget.
    """

    global Q_TABLE, VISIT_COUNTS

    Q_TABLE = {}
    VISIT_COUNTS = {}

    rng = np.random.default_rng(seed)

    env = IndustrialInventoryEnv(
        student_config=student_config,
        scenario_mode="random",
        domain_randomization=True,
    )

    epsilon = epsilon_start

    for episode in range(episodes):

        observation, _ = env.reset(
            seed=int(seed + episode)
        )

        state = observation_to_state(observation)

        while True:

            q_values = _get_q_values(state)

            # Epsilon-greedy exploration.
            if rng.random() < epsilon:
                action_index = int(
                    rng.integers(0, N_DISCRETE_ACTIONS)
                )
            else:
                action_index = int(
                    np.argmax(q_values)
                )

            internal_action = index_to_action(action_index)

            # Convert internal action indices into the actual quantities
            # expected by the official environment helper.
            quantities = (
                internal_action * ACTION_STEP
            ).astype(int).tolist()

            env_action = env.quantities_to_action_indices(
                quantities
            )

            next_observation, reward, terminated, truncated, _ = env.step(
                env_action
            )

            next_state = observation_to_state(
                next_observation
            )

            next_q_values = _get_q_values(next_state)

            if terminated or truncated:
                target = float(reward)
            else:
                target = float(reward) + gamma * float(
                    np.max(next_q_values)
                )

            # Adaptive learning rate: alpha shrinks as a (state, action)
            # pair gets visited more, so frequently-seen pairs stabilize
            # while rare pairs still move meaningfully.
            VISIT_COUNTS[state][action_index] += 1
            n = VISIT_COUNTS[state][action_index]
            alpha = max(alpha_min, 1.0 / (1.0 + n))

            # Q-learning update:
            #
            # Q(s,a) <- Q(s,a) +
            #           alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
            Q_TABLE[state][action_index] += alpha * (
                target - Q_TABLE[state][action_index]
            )

            state = next_state

            if terminated or truncated:
                break

        epsilon = max(
            epsilon_end,
            epsilon * epsilon_decay,
        )

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"epsilon={epsilon:.4f} | "
                f"states={len(Q_TABLE)}"
            )

    return Q_TABLE


# ---------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------

def save_q_table(
    q_table: dict[tuple[int, ...], np.ndarray],
    path: Path = MODEL_PATH,
) -> None:
    """Save the learned tabular policy."""

    states = list(q_table.keys())
    values = np.stack([q_table[state] for state in states])

    state_array = np.asarray(states, dtype=np.int16)

    np.savez_compressed(
        path,
        states=state_array,
        q_values=values,
    )


def load_q_table(
    path: Path = MODEL_PATH,
) -> dict[tuple[int, ...], np.ndarray]:
    """Load a frozen Q-table."""

    data = np.load(path)

    states = data["states"]
    q_values = data["q_values"]

    table = {}

    for state, values in zip(states, q_values):
        table[tuple(int(x) for x in state)] = values.astype(
            np.float32
        )

    return table


# ---------------------------------------------------------------------
# Deterministic policy interface
# ---------------------------------------------------------------------

def run_policy(observation: dict[str, Any]) -> list[int]:
    """Return order quantities for Products 1, 2 and 3.

    This function performs inference only.
    """

    state = observation_to_state(observation)

    if state not in Q_TABLE:
        # Instead of always ordering nothing for an unseen state, fall
        # back to a simple order-up-to-par heuristic. This keeps the
        # policy safe and reasonable on the large fraction of states the
        # table never got to visit during training.
        return _fallback_action(observation)

    q_values = Q_TABLE[state]

    # Deterministic greedy inference.
    best_action_index = int(np.argmax(q_values))

    internal_action = index_to_action(
        best_action_index
    )

    quantities = (
        internal_action * ACTION_STEP
    ).astype(int).tolist()

    return quantities


# ---------------------------------------------------------------------
# Training / export entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    student_config = generate_student_config(
        ROLL_NUMBER
    )

    print("Training Tabular Q-Learning (improved)...")

    q_table = train_q_learning(
        student_config=student_config,
        episodes=20000,
        alpha_min=0.02,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        seed=2026,
    )

    save_q_table(
        q_table,
        MODEL_PATH,
    )

    print()
    print("Training complete.")
    print(f"Number of states: {len(q_table)}")
    print(f"Saved Q-table to: {MODEL_PATH}")