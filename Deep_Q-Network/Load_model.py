from Env   import CPEEnv
from DQN import DQNAgent

# ── Inference / evaluation ────────────────────────────────────────────────────
def run(model_path: str, n_episodes: int = 10):
    """Run a trained model in the sim — no exploration, no learning."""
    env   = CPEEnv()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0  # pure exploitation, no random actions

    for episode in range(1, n_episodes + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        terminated   = False
        step_count   = 0

        while not terminated:
            action = agent.act(obs)         
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            step_count   += 1

        print(f"[RUN] Episode {episode} — reward: {total_reward:.2f}, steps: {step_count}, success: {info.get('success')}")

    env.close()


if __name__ == "__main__":
    run(model_path="checkpoints/model_ep10.json", n_episodes=5)