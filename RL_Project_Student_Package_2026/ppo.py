"""
PPO policy for the industrial inventory management RL project.

Trains a Stable-Baselines3 PPO agent on IndustrialInventoryEnv (wrapped as a
flat-Box Gymnasium env), and exposes run_policy() for the shared evaluation
loop in common/evaluation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

ROLL_NUMBER = "DA25G504"
MODEL_PATH = Path(__file__).with_name("ppo_model.zip")

OBS_DIM = 3 + 3 * 4 + 7 * 3 + 1 + 1

TOTAL_TIMESTEPS = 300_000
N_STEPS = 250
BATCH_SIZE = 250
GAMMA = 0.98
LEARNING_RATE = 3e-4
POLICY_KWARGS = {"net_arch": [128, 128]}

_MODEL: PPO | None = None


def observation_to_vector(observation: dict[str, Any]) -> np.ndarray:
    inventory = np.asarray(observation["inventory"], dtype=np.float32).reshape(3)
    pipeline = np.asarray(observation["arrival_pipeline"], dtype=np.float32).reshape(-1)
    demand_history = np.asarray(observation["demand_history"], dtype=np.float32).reshape(-1)
    day = np.asarray(observation["day"], dtype=np.float32).reshape(1)
    capacity = np.asarray(observation["capacity_utilisation"], dtype=np.float32).reshape(1)

    return np.concatenate([inventory, pipeline, demand_history, day, capacity])


class InventoryGymWrapper(gym.Env):
    """Gymnasium adapter exposing a flat Box observation for SB3."""

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
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0.0, high=10_000.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation_to_vector(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation_to_vector(observation), reward, terminated, truncated, info


def train_ppo(
    student_config: dict[str, Any],
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 2026,
) -> PPO:
    global _MODEL

    env = InventoryGymWrapper(student_config, scenario_mode="random", domain_randomization=True)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        policy_kwargs=POLICY_KWARGS,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)

    _MODEL = model
    return model


def save_model(model: PPO, path: Path = MODEL_PATH) -> None:
    model.save(path)


def load_model(path: Path = MODEL_PATH) -> PPO:
    global _MODEL
    _MODEL = PPO.load(path)
    return _MODEL


def run_policy(observation: dict[str, Any]) -> list[int]:
    if _MODEL is None:
        load_model()

    vector = observation_to_vector(observation)
    action, _ = _MODEL.predict(vector, deterministic=True)
    quantities = (np.asarray(action, dtype=np.int64) * 10).tolist()
    return quantities


if __name__ == "__main__":
    student_config = generate_student_config(ROLL_NUMBER)

    print("Training PPO...")

    model = train_ppo(student_config=student_config)
    save_model(model, MODEL_PATH)

    print()
    print("Training complete.")
    print(f"Saved PPO model to: {MODEL_PATH}")