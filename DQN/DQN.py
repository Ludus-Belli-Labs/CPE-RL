import json
import os
import random
import numpy as np
from Env import CPEEnv
import time

# ── Q-Learning Agent ──────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from Config import (
    LEARNINGRATE, EPSILON_MIN, BATCH_SIZE, EPSILON_DECAY, UPDATE_FREQUENCY
)

class _QNetwork(nn.Module):
    """Simple MLP: obs_dim → 128 → 128 → n_actions."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent (with experience replay + target network).

    Replaces QLearningAgent while keeping the same external interface:
      act(state) → action
      update(state, action, reward, next_state, done)
      decay()
      save(path) / load(path)
      save_checkpoint(episode)
    """

    def __init__(
        self,
        env: "CPEEnv" = None,  
        obs_dim: int       = 10,           # CMOEnv observation size
        n_actions: int     = CPEEnv.N_ACTIONS,
        lr: float          = LEARNINGRATE,
        gamma: float       = 0.99,
        epsilon: float     = 1.0,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        batch_size: int    = BATCH_SIZE,
        buffer_size: int   = 10_000,
        target_update_freq: int = UPDATE_FREQUENCY,     # update target net every N episodes
    ):
        self.env_ref           = env  
        self.obs_dim           = obs_dim
        self.n_actions         = n_actions
        self.lr                = lr
        self.gamma             = gamma
        self.epsilon           = epsilon
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq
        self._episode_done      = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")

        # Online network (trained every step) and frozen target network
        self.online_net = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        # Counters / diagnostics
        self._learn_steps   = 0
        self._episode_count = 0
        self._ep_td_errors  = []
        self._ep_q_values   = []

    # ── Internal helpers ──────────────────────────────────────

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        """Convert a numpy obs array to a float32 tensor on device."""
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _learn(self):
        """Sample a mini-batch and perform one gradient-descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough samples yet

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack([self._obs_to_tensor(s) for s in states])
        next_states = torch.stack([self._obs_to_tensor(s) for s in next_states])
        actions     = torch.tensor(actions,  dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards,  dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,    dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        q_vals = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient through target net)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Diagnostics
        td_errors = (targets - q_vals).detach().cpu().numpy()
        self._ep_td_errors.extend(td_errors.tolist())
        self._ep_q_values.extend(q_vals.detach().cpu().numpy().tolist())

        # Periodically sync target network
        self._learn_steps += 1
  

    # ── Public interface (mirrors QLearningAgent) ─────────────

    def on_episode_end(self):
        self._episode_count += 1

        if self._episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"[DQN] Target network synced at episode {self._episode_count}")

            if self.env_ref is not None:
                self.env_ref._send("VP_PauseSimulation ()", fmt="string")
                print(f"[DQN] Paused simulation for network syncing...")
                time.sleep(0.5)
                self.env_ref._send("VP_RunSimulation ()", fmt="string")

    def act(self, state) -> int:
        """epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online_net(self._obs_to_tensor(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        """Store transition in replay buffer then learn."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        self._episode_done = done
        self._learn()

    def decay(self):
        """Decay epsilon after each episode (unconditional, mirrors old behaviour)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="model.json"):
        """Save the full agent state (weights + hyperparameters) to a .json file."""
        torch.save({
            "hyperparameters": {
                "obs_dim":            self.obs_dim,
                "n_actions":          self.n_actions,
                "lr":                 self.lr,
                "gamma":              self.gamma,
                "epsilon":            self.epsilon,
                "epsilon_min":        self.epsilon_min,
                "epsilon_decay":      self.epsilon_decay,
                "batch_size":         self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        print(f"[DQN] Model saved → {path} (ε={self.epsilon:.4f}, "
              f"buffer={len(self.replay_buffer)})")

    def save_checkpoint(self, episode: int, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"model_ep{episode}.json")
        self.save(path)
        print(f"[DQN] Checkpoint saved → {path}")

    def load(self, path="model.json"):
        """Load a previously saved .json checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if "hyperparameters" in checkpoint:
            hp = checkpoint["hyperparameters"]
            self.obs_dim            = hp.get("obs_dim",            self.obs_dim)
            self.n_actions          = hp.get("n_actions",          self.n_actions)
            self.lr                 = hp.get("lr",                 self.lr)
            self.gamma              = hp.get("gamma",              self.gamma)
            self.epsilon            = hp.get("epsilon",            self.epsilon)
            self.epsilon_min        = hp.get("epsilon_min",        self.epsilon_min)
            self.epsilon_decay      = hp.get("epsilon_decay",      self.epsilon_decay)
            self.batch_size         = hp.get("batch_size",         self.batch_size)
            self.target_update_freq = hp.get("target_update_freq", self.target_update_freq)

        # Rebuild networks with (possibly updated) dims
        self.online_net = _QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = _QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"[DQN] Model loaded ← {path} (ε={self.epsilon:.4f})")
 


ACTION_NAMES = {
    0:  "nothing",
    1:  "move_random",
    2:  "move_to_ally",
    3:  "attack_slot_0",
    4:  "attack_slot_1",
    5:  "attack_slot_2",
    6:  "attack_slot_3",
    7:  "radar_on",
    8:  "radar_off",
    # 9:  "speed_fullstop",
    # 10: "speed_creep",
    # 11: "speed_cruise",
    # 12: "speed_full",
    # 13: "speed_flank",
    9: "move_to_target",
}