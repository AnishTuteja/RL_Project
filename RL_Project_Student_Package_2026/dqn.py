"""
DQN policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN

from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

ROLL_NUMBER = "DA25G504"
MODEL_PATH = Path(__file__).with_name("dqn_model.zip")

OBS_DIM = 3 + 3 * 4 + 7 * 3 + 1 + 1

N_ACTIONS_PER_PRODUCT = 11
N_PRODUCTS = 3
N_DISCRETE_ACTIONS = N_ACTIONS_PER_PRODUCT ** N_PRODUCTS

TOTAL_TIMESTEPS = 300_000
BUFFER_SIZE = 100_000
LEARNING_STARTS = 5_000
BATCH_SIZE = 64
GAMMA = 0.98
LEARNING_RATE = 1e-4
TRAIN_FREQ = 4
TARGET_UPDATE_INTERVAL = 1_000
EXPLORATION_FRACTION = 0.3
EXPLORATION_FINAL_EPS = 0.02
POLICY_KWARGS = {"net_arch": [128, 128]}

_MODEL: DQN | None = None


def observation_to_vector(observation: dict[str, Any]) -> np.ndarray:
    inventory = np.asarray(observation["inventory"], dtype=np.float32).reshape(3)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32).reshape(-1)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32).reshape(-1)
    day = np.asarray(observation["day"], dtype=np.float32).reshape(1)
    capacity = np.asarray(observation["capacity_utilisation"], dtype=np.float32).reshape(1)

    return np.concatenate([inventory, pipeline, demand_history, day, capacity])


def action_index_to_quantities(index: int) -> np.ndarray:
    index = int(index)
    a1, remainder = divmod(index, N_ACTIONS_PER_PRODUCT ** 2)
    a2, a3 = divmod(remainder, N_ACTIONS_PER_PRODUCT)
    return np.asarray([a1, a2, a3], dtype=np.int64) * 10


class InventoryGymWrapper(gym.Env):
    """Gymnasium adapter exposing a flat Box observation and a single
    Discrete action space for SB3's DQN."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        student_config: dict[str, Any],
        scenario_mode: str = "random",
        domain_randomization: bool = True,
    ) -> None:
        super().__init__()
        self.env = IndustrialInventoryEnv(
            student_config=student_config,
            scenario_mode=scenario_mode,
            domain_randomization=domain_randomization,
        )
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=10_000.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation_to_vector(observation), info

    def step(self, action):
        quantities = action_index_to_quantities(action)
        env_action = self.env.quantities_to_action_indices(quantities)
        observation, reward, terminated, truncated, info = self.env.step(env_action)
        return observation_to_vector(observation), reward, terminated, truncated, info


def train_dqn(
    student_config: dict[str, Any],
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 2026,
) -> DQN:
    global _MODEL

    env = InventoryGymWrapper(student_config, scenario_mode="random", domain_randomization=True)

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        train_freq=TRAIN_FREQ,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=POLICY_KWARGS,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)

    _MODEL = model
    return model


def save_model(model: DQN, path: Path = MODEL_PATH) -> None:
    model.save(path)


def load_model(path: Path = MODEL_PATH) -> DQN:
    global _MODEL
    _MODEL = DQN.load(path)
    return _MODEL


def run_policy(observation: dict[str, Any]) -> list[int]:
    if _MODEL is None:
        load_model()

    vector = observation_to_vector(observation)
    action, _ = _MODEL.predict(vector, deterministic=True)
    quantities = action_index_to_quantities(action)
    return quantities.tolist()


if __name__ == "__main__":
    student_config = generate_student_config(ROLL_NUMBER)

    print("Training DQN...")

    model = train_dqn(student_config=student_config)
    save_model(model, MODEL_PATH)

    print()
    print("Training complete.")
    print(f"Saved DQN model to: {MODEL_PATH}")