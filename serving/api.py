from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
import os

app = FastAPI(title="GPU RL Scheduler API")

# ---- Model Loading ----
MODEL_PATH = os.path.join("models", "ppo_gpu_scheduler")

model = PPO.load(MODEL_PATH)


# ---- Request Schema ----
class StateInput(BaseModel):
    state: list  # expected 6-dimensional state


# ---- Health Check ----
@app.get("/")
def read_root():
    return {"status": "GPU RL Scheduler API is running."}


# ---- Prediction Endpoint ----
@app.post("/predict")
def predict_action(input_data: StateInput):

    state_array = np.array(input_data.state, dtype=np.float32)

    if state_array.shape[0] != 6:
        return {"error": "State must be 6-dimensional."}

    action, _ = model.predict(state_array, deterministic=True)

    return {"action": int(action)}