import json
import os
import random
import numpy as np

from RL.Env import CMOEnv

# ── Q-Learning Agent ──────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(self, n_actions=CMOEnv.N_ACTIONS, lr=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_actions     = n_actions
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = {}

    def _get_q(self, state):
        key = tuple(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def act(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self._get_q(state)))

    def update(self, state, action, reward, next_state, done):
        q      = self._get_q(state)
        q_next = self._get_q(next_state)
        target = reward + (0 if done else self.gamma * np.max(q_next))
        q[action] += self.lr * (target - q[action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path="try/model.json"):
        data = {
            "hyperparameters": {
                "n_actions":     self.n_actions,
                "lr":            self.lr,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AGENT] Model saved → {path} ({len(self.q_table)} states, ε={self.epsilon:.4f})")

    def save_checkpoint(self, episode: int, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"model_ep{episode}.json")
        self.save(path)
        print(f"[AGENT] Checkpoint saved → {path}")

    def load(self, path="model.json"):
        with open(path) as f:
            data = json.load(f)
        if "hyperparameters" in data:
            hp = data["hyperparameters"]
            self.n_actions     = hp.get("n_actions",     self.n_actions)
            self.lr            = hp.get("lr",            self.lr)
            self.gamma         = hp.get("gamma",         self.gamma)
            self.epsilon       = hp.get("epsilon",       self.epsilon)
            self.epsilon_min   = hp.get("epsilon_min",   self.epsilon_min)
            self.epsilon_decay = hp.get("epsilon_decay", self.epsilon_decay)
            self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
            print(f"[AGENT] Model loaded -----------------------------------------------")
        else:
            self.q_table = {eval(k): np.array(v) for k, v in data.items()}
        print(f"[AGENT] Model loaded ← {path} ({len(self.q_table)} states, ε={self.epsilon:.4f})")