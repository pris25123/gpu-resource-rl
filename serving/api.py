from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO

# Load trained model once at startup
model = PPO.load("models/ppo_gpu_scheduler")

app = FastAPI(title="GPU RL Scheduler API")


# Define request schema
class StateInput(BaseModel):
    state: list  # 6-dimensional state


# Health check
@app.get("/")
def read_root():
    return {"message": "GPU RL Scheduler API is running."}


# Prediction endpoint
@app.post("/predict")
def predict_action(input_data: StateInput):

    state_array = np.array(input_data.state, dtype=np.float32)

    action, _ = model.predict(state_array, deterministic=True)

    return {
        "action": int(action)
    }