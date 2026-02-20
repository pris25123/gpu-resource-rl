import os
import mlflow
import numpy as np
from simulator.environment import GPUEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def evaluate_model(model, steps=500):
    env = GPUEnvironment()
    obs, _ = env.reset()

    total_reward = 0
    total_power = []
    jobs_completed = 0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        total_power.append(env.power_draw)

        if reward > 0:
            jobs_completed += 1

    return total_reward, np.mean(total_power), jobs_completed


def main():

    mlflow.set_experiment("GPU_RL_Scheduler")

    with mlflow.start_run():

        env = GPUEnvironment()
        check_env(env)

        learning_rate = 3e-4
        total_timesteps = 100000

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=64,
            gamma=0.99
        )

        model.learn(total_timesteps=total_timesteps)

        os.makedirs("models", exist_ok=True)
        model_path = "models/ppo_gpu_scheduler"
        model.save(model_path)

        # Evaluate after training
        total_reward, avg_power, jobs_completed = evaluate_model(model)

        # Log hyperparameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("total_timesteps", total_timesteps)

        # Log metrics
        mlflow.log_metric("eval_total_reward", total_reward)
        mlflow.log_metric("eval_avg_power", avg_power)
        mlflow.log_metric("eval_jobs_completed", jobs_completed)

        # Log model artifact
        mlflow.log_artifact(model_path + ".zip")

        print("\nTraining complete and logged to MLflow.")


if __name__ == "__main__":
    main()