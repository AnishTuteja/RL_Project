# NOTE: Lines 1-42 (the top of the module docstring) were not visible in the
# provided screenshots (they started at line 43). The docstring below picks
# up mid-sentence at the point the screenshots begin. Please paste in the
# missing opening lines from your own file if you have them.
"""
...
data the network trains on close to on-policy throughout.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config

ROLL_NUMBER = "DA25G504"
MODEL_PATH = Path(__file__).with_name("nn_sarsa_model.pt")

OBS_DIM = 3 + 3 * 4 + 7 * 3 + 1 + 1

N_ACTIONS_PER_PRODUCT = 11
N_PRODUCTS = 3
N_DISCRETE_ACTIONS = N_ACTIONS_PER_PRODUCT ** N_PRODUCTS

TOTAL_TIMESTEPS = 300_000
BUFFER_SIZE = 2_000
LEARNING_STARTS = 500
BATCH_SIZE = 32
GAMMA = 0.98
LEARNING_RATE = 1e-4
TRAIN_FREQ = 1
TARGET_UPDATE_INTERVAL = 200
EXPLORATION_FRACTION = 0.3
EXPLORATION_FINAL_EPS = 0.02
HIDDEN_SIZES = (128, 128)
GRAD_CLIP_NORM = 10.0

_MODEL: "QNetwork | None" = None


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


class QNetwork(nn.Module):
    """Simple MLP mapping the flattened observation to per-action Q-values."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_DISCRETE_ACTIONS,
                 hidden_sizes: tuple[int, ...] = HIDDEN_SIZES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, n_actions))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SarsaWindow:
    """Small sliding-window buffer of (s, a, r, s', a', done) transitions.

    Deliberately much smaller than DQN's replay buffer (see module
    docstring): it only decorrelates consecutive minibatch samples, not
    accumulate a long off-policy history, which would violate SARSA's
    on-policy assumption that a' reflects the current behaviour policy.
    """

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_action, done) -> None:
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(next_actions, dtype=np.int64),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def _linear_epsilon(step: int, total_timesteps: int) -> float:
    """Linearly anneal epsilon from 1.0 down to EXPLORATION_FINAL_EPS over
    EXPLORATION_FRACTION of training, then hold it constant."""
    exploration_steps = EXPLORATION_FRACTION * total_timesteps
    progress = min(1.0, step / max(1.0, exploration_steps))
    return 1.0 + progress * (EXPLORATION_FINAL_EPS - 1.0)


def _select_action(net: QNetwork, state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(N_DISCRETE_ACTIONS)
    with torch.no_grad():
        q_values = net(torch.from_numpy(state).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())


def train_nn_sarsa(
    student_config: dict[str, Any],
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 2026,
) -> QNetwork:
    """Train Algorithm 6 using neural-network (semi-gradient) SARSA.

    The behaviour policy is epsilon-greedy over the online network. At every
    step the action for the *next* state is chosen up front (on-policy),
    stored with the transition, and later used verbatim as the bootstrap
    action when computing the TD target -- never a max or an independently
    re-selected argmax, which is what distinguishes this from DQN / Double
    DQN. Training happens every step against a small, near-on-policy window
    of recent transitions (see `SarsaWindow`) rather than a large replay
    buffer, since a large buffer would mix in next-actions chosen under
    stale, much more exploratory past policies.
    """
    global _MODEL

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = IndustrialInventoryEnv(
        student_config=student_config,
        scenario_mode="random",
        domain_randomization=True,
    )

    online_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    window = SarsaWindow(BUFFER_SIZE)

    observation, _ = env.reset(seed=seed)
    state = observation_to_vector(observation)
    epsilon = _linear_epsilon(0, total_timesteps)
    action = _select_action(online_net, state, epsilon)

    episode = 0
    for step in range(total_timesteps):
        epsilon = _linear_epsilon(step, total_timesteps)

        quantities = action_index_to_quantities(action)
        env_action = env.quantities_to_action_indices(quantities)
        next_observation, reward, terminated, truncated, _ = env.step(env_action)
        next_state = observation_to_vector(next_observation)
        done = terminated or truncated

        # Chosen for its own sake (it becomes next step's action if the
        # episode continues) and stored as SARSA's bootstrap action a'.
        # When done, its Q-value is multiplied by zero below, so which
        # action gets selected here is irrelevant to the target.
        next_action = _select_action(online_net, next_state, epsilon)
        window.push(state, action, reward, next_state, next_action, done)

        if done:
            episode += 1
            observation, _ = env.reset(seed=seed + episode)
            state = observation_to_vector(observation)
            action = _select_action(online_net, state, epsilon)
        else:
            state = next_state
            action = next_action

        if step >= LEARNING_STARTS and step % TRAIN_FREQ == 0 and len(window) >= BATCH_SIZE:
            states, actions, rewards, next_states, next_actions, dones = window.sample(BATCH_SIZE)
            states_t = torch.from_numpy(states)
            actions_t = torch.from_numpy(actions)
            rewards_t = torch.from_numpy(rewards)
            next_states_t = torch.from_numpy(next_states)
            next_actions_t = torch.from_numpy(next_actions)
            dones_t = torch.from_numpy(dones)

            q_values = online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # SARSA: bootstrap from the Q-value of the action the
                # behaviour policy actually selected at s' (on-policy),
                # never a max or a separately-derived argmax.
                next_q = target_net(next_states_t).gather(
                    1, next_actions_t.unsqueeze(1)
                ).squeeze(1)
                targets = rewards_t + GAMMA * (1.0 - dones_t) * next_q

            loss = F.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        if step % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(online_net.state_dict())

        if (step + 1) % 10_000 == 0:
            print(f"Step {step + 1}/{total_timesteps} | epsilon={epsilon:.3f} | episodes={episode}")

    _MODEL = online_net
    return online_net


def save_model(model: QNetwork, path: Path = MODEL_PATH) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: Path = MODEL_PATH) -> QNetwork:
    global _MODEL
    model = QNetwork()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    _MODEL = model
    return _MODEL


def run_policy(observation: dict[str, Any]) -> list[int]:
    if _MODEL is None:
        load_model()

    vector = observation_to_vector(observation)
    with torch.no_grad():
        q_values = _MODEL(torch.from_numpy(vector).unsqueeze(0))
        action_index = int(torch.argmax(q_values, dim=1).item())

    quantities = action_index_to_quantities(action_index)
    return quantities.tolist()


if __name__ == "__main__":
    student_config = generate_student_config(ROLL_NUMBER)

    print("Training Neural-Network SARSA...")

    model = train_nn_sarsa(student_config=student_config)
    save_model(model, MODEL_PATH)

    print()
    print("Training complete.")
    print(f"Saved Neural-Network SARSA model to: {MODEL_PATH}")