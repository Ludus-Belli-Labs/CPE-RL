import time
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TrainingGraphs import TrainingGraph

from Env    import CPEEnv
from DQN  import DQNAgent
from Config import ACTION_NAMES
from Utils  import parse_start_episode


# Training loop 

def train(n_episodes: int, scenario_xml: str = "", load_model: str = None, checkpoint: int = 10):
    env   = CPEEnv()
    agent = DQNAgent(env=env)
    
    n_actions = CPEEnv.N_ACTIONS
    start_episode = 0

    if load_model:                 
        agent.load(load_model)
        start_episode= parse_start_episode(load_model)
        print(f"[TRAIN] Resuming from: {load_model}")
 
    graph = TrainingGraph(start_episode=start_episode)
    successes = 0
    # epsilon_restarts     = 0
    # MAX_EPSILON_RESTARTS = 3
    # EPSILON_RESET_VALUE  = 0.3
 
    for episode in range(start_episode + 1,start_episode + n_episodes + 1):
        obs, _        = env.reset()
        total_reward  = 0.0
        terminated    = False
        step_count    = 0
        success       = 0
        action_counts = [0] * n_actions
        ep_start      = time.time()
        current_scenario_name = getattr(env, "scenario_name", f"{episode}")
        agent._ep_td_errors = []
        agent._ep_q_values  = []
        
 
        while not terminated:
            action = agent.act(obs)
            action_counts[action] += 1
            next_obs, reward, terminated, _, info = env.step(action)
            step_count   += 1
            agent.update(obs, action, reward, next_obs, terminated)
            obs           = next_obs
            total_reward  += reward
            if info.get("success"):
                success = 1

        td = np.array(agent._ep_td_errors, dtype=np.float32) if agent._ep_td_errors else np.zeros(1)
        qv = np.array(agent._ep_q_values,  dtype=np.float32) if agent._ep_q_values  else np.zeros(1)
        ep_loss   = float(np.mean(td ** 2))
        ep_q_mean = float(np.mean(qv))
        ep_q_max  = float(np.max(qv))
 
        duration_s = time.time() - ep_start
        successes += success

        agent.on_episode_end() 
        agent.decay()

        # if agent.epsilon <= agent.epsilon_min:
        #     if epsilon_restarts >= MAX_EPSILON_RESTARTS:
        #         print(f"[END] Episode {episode} — Reached epsilon minimum after {epsilon_restarts} restarts. Training complete.")
        #         agent.save_checkpoint(episode)
        #         break
        #     epsilon_restarts += 1
        #     agent.epsilon = EPSILON_RESET_VALUE
        #     print(f"[EPSILON RESTART #{epsilon_restarts}/{MAX_EPSILON_RESTARTS}] Episode {episode} — "
        #           f"resetting epsilon to {EPSILON_RESET_VALUE} and continuing training.")

        graph.update(
            episode      = episode,
            reward       = total_reward,
            steps        = step_count,
            success      = success,
            epsilon      = agent.epsilon,
            duration_s   = duration_s,
            action_counts= action_counts,
            action_names = ACTION_NAMES,
            scenario = current_scenario_name,
            loss         = ep_loss,
            q_mean       = ep_q_mean,
            q_max        = ep_q_max,
        )

        if episode % checkpoint == 0:
            agent.save_checkpoint(episode)
 
    env.close()
    agent.save("model.pt")
    graph.save()
    return agent

if __name__ == "__main__":
    train(n_episodes=100, checkpoint=10, load_model=None)