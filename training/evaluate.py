import numpy as np
from simulator.environment import GPUEnvironment
from stable_baselines3 import PPO


def evaluate_rl(model, steps=500):
    env = GPUEnvironment()
    obs, _ = env.reset()

    total_reward = 0
    temps = []
    powers = []
    jobs_completed = 0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        temps.append(env.temperature)
        powers.append(env.power_draw)

        if reward > 0:
            jobs_completed += 1

    return {
        "total_reward": total_reward,
        "jobs_completed": jobs_completed,
        "avg_temperature": np.mean(temps),
        "avg_power": np.mean(powers),
    }


def evaluate_fifo(steps=500):
    env = GPUEnvironment()
    obs, _ = env.reset()

    total_reward = 0
    temps = []
    powers = []
    jobs_completed = 0

    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(0)

        total_reward += reward
        temps.append(env.temperature)
        powers.append(env.power_draw)

        if reward > 0:
            jobs_completed += 1

    return {
        "total_reward": total_reward,
        "jobs_completed": jobs_completed,
        "avg_temperature": np.mean(temps),
        "avg_power": np.mean(powers),
    }


if __name__ == "__main__":
    # Load trained model
    model = PPO.load("models/ppo_gpu_scheduler")

    rl_results = evaluate_rl(model)
    fifo_results = evaluate_fifo()

    print("\n=== RL Policy ===")
    for k, v in rl_results.items():
        print(f"{k}: {v}")

    print("\n=== FIFO Baseline ===")
    for k, v in fifo_results.items():
        print(f"{k}: {v}")