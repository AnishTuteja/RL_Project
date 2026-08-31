"""Algorithm 1: Tabular Q-Learning policy for the inventory-control project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROLL_NUMBER = "DA25G504"

MODEL_PATH = Path(__file__).with_name("q_table.npz")

ACTION_VALUES = np.arange(0, 101, 10, dtype=np.int16)

N_ACTIONS_PER_PRODUCT = 11
N_PRODUCTS = 3


# ---------------------------------------------------------------------
# State preprocessing
# ---------------------------------------------------------------------

def _bucket_inventory(x: float) -> int:
    """Bucket inventory into 20-unit intervals."""
    return int(np.clip(np.floor(float(x) / 20.0), 0, 10))


def _bucket_pipeline(x: float) -> int:
    """Bucket pipeline quantities into 20-unit intervals."""
    return int(np.clip(np.floor(float(x) / 20.0), 0, 5))


def _bucket_demand(x: float) -> int:
    """Bucket recent demand into coarse demand levels."""
    return int(np.clip(np.floor(float(x) / 10.0), 0, 10))


def _bucket_capacity(x: float) -> int:
    """Bucket capacity utilisation."""
    return int(np.clip(np.floor(float(x) * 10.0), 0, 10))


def observation_to_state(observation: dict[str, Any]) -> tuple[int, ...]:
    """Convert the official observation dictionary into a tabular state.

    The environment observation itself is not modified. Only the internal
    representation used by the Q-table is discretized.
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

    # Outstanding pipeline.
    #
    # Rather than retaining every raw pipeline value, aggregate the
    # near-future inventory into four coarse quantities per product.
    for product in range(N_PRODUCTS):
        for lead_day in range(4):
            state.append(_bucket_pipeline(pipeline[product, lead_day]))

    # Demand history: use the most recent three days.
    #
    # The environment provides seven days, but retaining all seven days
    # would make the tabular state space unnecessarily large.
    recent_history = demand_history[-3:]

    for row in recent_history:
        for value in row:
            state.append(_bucket_demand(value))

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


def _get_q_values(state: tuple[int, ...]) -> np.ndarray:
    """Return the Q-values for a state, creating them if necessary."""
    if state not in Q_TABLE:
        Q_TABLE[state] = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)

    return Q_TABLE[state]


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_q_learning(
    student_config: dict[str, Any],
    episodes: int = 5000,
    alpha: float = 0.10,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    seed: int = 2026,
) -> dict[tuple[int, ...], np.ndarray]:
    """Train Algorithm 1 using tabular Q-learning."""

    global Q_TABLE

    Q_TABLE = {}

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
                internal_action * 10
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
        # Conservative deterministic fallback for an unseen state.
        #
        # The fallback keeps the policy valid without modifying the
        # learned model during evaluation.
        return [0, 0, 0]

    q_values = Q_TABLE[state]

    # Deterministic greedy inference.
    best_action_index = int(np.argmax(q_values))

    internal_action = index_to_action(
        best_action_index
    )

    quantities = (
        internal_action * 10
    ).astype(int).tolist()

    return quantities


# ---------------------------------------------------------------------
# Training / export entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    student_config = generate_student_config(
        ROLL_NUMBER
    )

    print("Training Tabular Q-Learning...")

    q_table = train_q_learning(
        student_config=student_config,
        episodes=5000,
        alpha=0.10,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
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

