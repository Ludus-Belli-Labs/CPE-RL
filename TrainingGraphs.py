import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import csv
 
 
class TrainingGraph:
    def __init__(
        self,
        moving_avg_window: int = 10,
        save_path: str = "training_progress.png",
        csv_path:  str = "training_log.csv",
        start_episode: int = 0,
    ):
        self.window    = moving_avg_window
        self.save_path = save_path
        self.csv_path  = csv_path
 
        self.episodes        = []
        self.rewards         = []
        self.steps           = []
        self.successes       = []
        self.success_rates   = []
        self.losses          = []   # mean training loss per episode
        self.q_means         = []   # mean Q-value per episode
        self.q_maxes         = []   # max  Q-value per episode
        self.durations       = []   # episode wall-clock time (s)
        self.epsilons        = []   # epsilon at end of episode
        self.most_used_acts  = []   # most-used action label per episode
        self.scenarios       = []   # scenario name per episode
        self.action_counts_history = []  # list of action-count lists per episode
 
        self._csv_file      = None
        self._csv_writer    = None
        self._csv_ready     = False   # header written flag
        self._prior_successes = 0      
        self._prior_episodes  = 0        
        if start_episode > 0 and os.path.isfile(csv_path):
            self._load_existing_csv(csv_path)
 
        # Set up live plot with 3 panels
        plt.ion()
        self.fig = plt.figure(figsize=(11, 10))
        self.fig.suptitle("CMO RL Agent — Training Progress", fontsize=14)
 
        gs = gridspec.GridSpec(3, 1, figure=self.fig, hspace=0.45)
        self.ax_reward  = self.fig.add_subplot(gs[0])
        self.ax_steps   = self.fig.add_subplot(gs[1], sharex=self.ax_reward)
        self.ax_success = self.fig.add_subplot(gs[2], sharex=self.ax_reward)
 
        self._setup_axes()
        plt.show(block=False)
 
    # ── Setup ─────────────────────────────────────────────────
 
    def _load_existing_csv(self, path: str):
        """Load prior training history from CSV so graphs show full run on resume."""
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.episodes.append(int(row["episode"]))
                    self.rewards.append(float(row["total_reward"]))
                    self.steps.append(int(row["step_count"]))
                    self.successes.append(int(row["success"]))
                    self.success_rates.append(float(row["success_rate"]))
                    self.losses.append(float(row["loss"])   if row.get("loss")   not in (None, "") else float("nan"))
                    self.q_means.append(float(row["q_mean"]) if row.get("q_mean") not in (None, "") else float("nan"))
                    self.q_maxes.append(float(row["q_max"])  if row.get("q_max")  not in (None, "") else float("nan"))
                    self.durations.append(float(row["duration_s"]) if row.get("duration_s") not in (None, "") else 0.0)
                    self.epsilons.append(float(row["epsilon"])     if row.get("epsilon")     not in (None, "") else 0.0)
                    self.most_used_acts.append(row.get("most_used_action", ""))
                    self.scenarios.append(row.get("scenario", "Unknown"))
                    ac = []
                    j = 0
                    while f"action_{j}_count" in row:
                        ac.append(int(row[f"action_{j}_count"]))
                        j += 1
                    self.action_counts_history.append(ac)
            self._prior_successes = sum(self.successes)
            self._prior_episodes  = len(self.successes)
            print(f"[GRAPH] Loaded {self._prior_episodes} prior episodes from {path} "
                  f"({self._prior_successes} successes)")
        except Exception as e:
            print(f"[GRAPH] WARNING: could not load existing CSV ({e}) — starting fresh.")

    def _setup_axes(self):
        self.ax_reward.set_ylabel("Total Reward")
        self.ax_reward.set_title("Reward per Episode")
        self.ax_reward.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        self.ax_reward.grid(True, alpha=0.3)
 
        self.ax_steps.set_ylabel("Steps")
        self.ax_steps.set_title("Steps per Episode")
        self.ax_steps.grid(True, alpha=0.3)
 
        self.ax_success.set_ylabel("Success Rate")
        self.ax_success.set_xlabel("Episode")
        self.ax_success.set_title("Learning Curve (Cumulative Success Rate)")
        self.ax_success.set_ylim(0, 1)
        self.ax_success.grid(True, alpha=0.3)
 
    def _moving_avg(self, values: list) -> list:
        avgs = []
        for i in range(1, len(values) + 1):
            window_vals = values[max(0, i - self.window):i]
            avgs.append(sum(window_vals) / len(window_vals))
        return avgs
 
    # ── CSV helpers ───────────────────────────────────────────
 
    def _ensure_csv(self, action_counts: list, action_names: dict):
        """Open CSV file and write header on first call."""
        if self._csv_ready:
            return
        n_actions   = len(action_counts)
        action_cols = [f"action_{i}_count" for i in range(n_actions)]
        self._fieldnames = (
            ["episode", "total_reward", "step_count", "success", "success_rate",
             "duration_s", "epsilon"]
            + action_cols
            + ["most_used_action", "scenario", "loss", "q_mean", "q_max"]
        )
        csv_exists       = os.path.isfile(self.csv_path)
        self._csv_file   = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        if not csv_exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()
        self._csv_ready = True
 
    def _write_csv_row(self, episode, reward, steps, success, success_rate,
                       duration_s, epsilon, action_counts, action_names,
                       scenario="Unknown", loss=None, q_mean=None, q_max=None):
        most_used_idx    = int(max(range(len(action_counts)), key=lambda i: action_counts[i]))
        most_used_action = action_names.get(most_used_idx, str(most_used_idx))
        row = {
            "episode":          episode,
            "total_reward":     round(reward, 4),
            "step_count":       steps,
            "success":          success,
            "success_rate":     round(success_rate, 4),
            "duration_s":       round(duration_s, 2),
            "epsilon":          round(epsilon, 5),
            "most_used_action": most_used_action,
            "scenario":         scenario,
            "loss":             round(loss,   6) if loss   is not None else "",
            "q_mean":           round(q_mean, 6) if q_mean is not None else "",
            "q_max":            round(q_max,  6) if q_max  is not None else "",
        }
        for i, count in enumerate(action_counts):
            row[f"action_{i}_count"] = count
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        return most_used_action
 
    # ── Public API ────────────────────────────────────────────
 
    def update(
        self,
        episode: int,
        reward: float,
        steps: int,
        success: int = 0,
        epsilon: float = 0.0,
        duration_s: float = 0.0,
        action_counts: list = None,
        action_names: dict = None,
        scenario: str = "Unknown",
        loss: float = None,
        q_mean: float = None,
        q_max: float = None,
    ):
        """Call after each episode to record data, write CSV row, and refresh the live plot."""
        action_counts = action_counts or []
        action_names  = action_names  or {}
 
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.successes.append(success)
        self.losses.append(loss    if loss   is not None else float("nan"))
        self.q_means.append(q_mean if q_mean is not None else float("nan"))
        self.q_maxes.append(q_max  if q_max  is not None else float("nan"))
        self.durations.append(duration_s)
        self.epsilons.append(epsilon)
        self.scenarios.append(scenario)
        self.action_counts_history.append(list(action_counts))
 
        total_successes = self._prior_successes + sum(self.successes[self._prior_episodes:])
        total_episodes  = self._prior_episodes  + len(self.successes[self._prior_episodes:])
        cumulative_rate = total_successes / total_episodes if total_episodes else 0.0
        self.success_rates.append(cumulative_rate)
 
        # CSV
        if action_counts:
            self._ensure_csv(action_counts, action_names)
            most_used = self._write_csv_row(
                episode, reward, steps, success, cumulative_rate,
                duration_s, epsilon, action_counts, action_names, scenario,
                loss, q_mean, q_max,
            )
        else:
            most_used = "—"
        self.most_used_acts.append(most_used)

        # Save a checkpoint CSV every 10 episodes so progress is not lost
        if episode % 10 == 0:
            self._save_csv_checkpoint(episode)

        self._redraw()
 
        loss_str  = f"{loss:.5f}"  if loss   is not None else "n/a"
        qmean_str = f"{q_mean:.3f}" if q_mean is not None else "n/a"
        qmax_str  = f"{q_max:.3f}"  if q_max  is not None else "n/a"
        print(
            f"  [GRAPH] ep={episode} | reward={reward:+.1f} | steps={steps} | "
            f"success={success} ({cumulative_rate:.0%}) | "
            f"time={duration_s:.1f}s "
        )
 
    def _save_csv_checkpoint(self, episode: int):
        """Flush a full-fidelity snapshot of the entire log to a checkpoint CSV."""
        checkpoint_dir = "csv_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"training_log_ep{episode}.csv")

        # Determine max number of action columns seen so far
        max_actions = max((len(ac) for ac in self.action_counts_history if ac), default=0)
        action_cols = [f"action_{i}_count" for i in range(max_actions)]

        fieldnames = (
            ["episode", "total_reward", "step_count", "success", "success_rate",
             "duration_s", "epsilon"]
            + action_cols
            + ["most_used_action", "scenario", "loss", "q_mean", "q_max"]
        )

        def _fmt(val, digits):
            """Round val if it is a real number, else return empty string."""
            try:
                if val != val:   # NaN check
                    return ""
                return round(val, digits)
            except (TypeError, ValueError):
                return ""

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, ep in enumerate(self.episodes):
                ac = self.action_counts_history[i] if i < len(self.action_counts_history) else []
                row = {
                    "episode":          ep,
                    "total_reward":     _fmt(self.rewards[i], 4),
                    "step_count":       self.steps[i],
                    "success":          self.successes[i],
                    "success_rate":     _fmt(self.success_rates[i], 4),
                    "duration_s":       _fmt(self.durations[i] if i < len(self.durations) else float("nan"), 2),
                    "epsilon":          _fmt(self.epsilons[i]  if i < len(self.epsilons)  else float("nan"), 5),
                    "most_used_action": self.most_used_acts[i] if i < len(self.most_used_acts) else "",
                    "scenario":         self.scenarios[i]      if i < len(self.scenarios)      else "",
                    "loss":             _fmt(self.losses[i]    if i < len(self.losses)          else float("nan"), 6),
                    "q_mean":           _fmt(self.q_means[i]   if i < len(self.q_means)         else float("nan"), 6),
                    "q_max":            _fmt(self.q_maxes[i]   if i < len(self.q_maxes)         else float("nan"), 6),
                }
                for j in range(max_actions):
                    row[f"action_{j}_count"] = ac[j] if j < len(ac) else ""
                writer.writerow(row)
        print(f"[GRAPH] CSV checkpoint saved → {path}")

    def close(self):
        """Close the CSV file. Call when training ends."""
        if self._csv_file:
            self._csv_file.close()
            print(f"[GRAPH] CSV saved → {self.csv_path}")
 
    def save(self):
        """Call after training ends — saves the final plot as a PNG."""
        self.close()
        plt.ioff()
        self._redraw()
        self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAPH] Final plot saved → {self.save_path}")
        plt.show(block=True)
 
    # ── Internal ──────────────────────────────────────────────
 
    def _redraw(self):
        eps = self.episodes
 
        # ── Reward panel ──────────────────────────────────────
        self.ax_reward.cla()
        self.ax_reward.set_ylabel("Total Reward")
        self.ax_reward.set_title("Reward per Episode")
        self.ax_reward.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        self.ax_reward.grid(True, alpha=0.3)
        self.ax_reward.plot(eps, self.rewards,
                            color="steelblue", linewidth=1, alpha=0.6, label="Reward")
        if len(eps) >= 2:
            self.ax_reward.plot(eps, self._moving_avg(self.rewards),
                                color="orange", linewidth=2,
                                label=f"{self.window}-ep avg")
        if self.rewards:
            best  = max(self.rewards)
            worst = min(self.rewards)
            self.ax_reward.set_ylim(worst - abs(worst) * 0.1,
                                    best  + abs(best)  * 0.1 + 1)
        self.ax_reward.legend(fontsize=8)
 
        # ── Steps panel ───────────────────────────────────────
        self.ax_steps.cla()
        self.ax_steps.set_ylabel("Steps")
        self.ax_steps.set_title("Steps per Episode")
        self.ax_steps.grid(True, alpha=0.3)
        self.ax_steps.plot(eps, self.steps,
                           color="seagreen", linewidth=1, alpha=0.6, label="Steps")
        if len(eps) >= 2:
            self.ax_steps.plot(eps, self._moving_avg(self.steps),
                               color="orange", linewidth=2,
                               label=f"{self.window}-ep avg")
        self.ax_steps.legend(fontsize=8)
 
        # ── Success rate panel ────────────────────────────────
        self.ax_success.cla()
        self.ax_success.set_ylabel("Success Rate")
        self.ax_success.set_xlabel("Episode")
        self.ax_success.set_title("Learning Curve (Cumulative Success Rate)")
        self.ax_success.set_ylim(0, 1)
        self.ax_success.grid(True, alpha=0.3)
        self.ax_success.plot(eps, self.success_rates,
                             color="crimson", linewidth=2, label="Success Rate")
        self.ax_success.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        self.ax_success.legend(fontsize=8)
 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
 
    @staticmethod
    def load_from_file(path: str = "training_log.csv") -> "TrainingGraph":
        """
        Reload a previous training run from CSV and show its final plot.
 
            graph = TrainingGraph.load_from_file("training_log.csv")
            graph.save()
        """
        graph = TrainingGraph()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                graph.episodes.append(int(row["episode"]))
                graph.rewards.append(float(row["total_reward"]))
                graph.steps.append(int(row["step_count"]))
                graph.successes.append(int(row["success"]))
                graph.success_rates.append(float(row["success_rate"]))
                graph.losses.append(float(row["loss"])   if row.get("loss")   not in (None, "") else float("nan"))
                graph.q_means.append(float(row["q_mean"]) if row.get("q_mean") not in (None, "") else float("nan"))
                graph.q_maxes.append(float(row["q_max"])  if row.get("q_max")  not in (None, "") else float("nan"))
        graph._redraw()
        return graph