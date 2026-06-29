import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TrainingGraphs import TrainingGraph

from Env    import CMOEnv
from Qlearning  import QLearningAgent
from Config import ACTION_NAMES

# Training loop 

def train(n_episodes: int = 100, scenario_xml: str = "",
          load_model: str = None, checkpoint: int = 10):

    env       = CMOEnv()
    agent     = QLearningAgent()
    graph     = TrainingGraph()
    n_actions = CMOEnv.N_ACTIONS

    if load_model:
        agent.load(load_model)
        print(f"[TRAIN] Resuming from: {load_model}")

    successes = 0

    for episode in range(1, n_episodes + 1):
        obs, _        = env.reset()
        total_reward  = 0.0
        terminated    = False
        step_count    = 0
        success       = 0
        action_counts = [0] * n_actions
        ep_start      = time.time()
        current_scenario_name = getattr(env, "scenario_name", f"{episode}")

        while not terminated:
            action = agent.act(obs)
            action_counts[action] += 1
            next_obs, reward, terminated, _, info = env.step(action)
            step_count  += 1
            agent.update(obs, action, reward, next_obs, terminated)
            obs          = next_obs
            total_reward += reward
            if info.get("success"):
                success = 1

        duration_s = time.time() - ep_start
        successes += success

        agent.decay()
        graph.update(
            episode       = episode,
            reward        = total_reward,
            steps         = step_count,
            success       = success,
            epsilon       = agent.epsilon,
            duration_s    = duration_s,
            action_counts = action_counts,
            action_names  = ACTION_NAMES,
            scenario      = current_scenario_name,
        )

        if episode % checkpoint == 0:
            agent.save_checkpoint(episode)

    env.close()
    agent.save()
    graph.save()
    return agent


if __name__ == "__main__":
    train(n_episodes=1000, load_model=None, checkpoint=10)