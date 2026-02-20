import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator.job_generator import generate_job


class GPUEnvironment(gym.Env):

    def __init__(self):
        super(GPUEnvironment, self).__init__()

        # System constants
        self.vram = 16
        self.max_power = 250
        self.thermal_limit = 85
        self.base_power = 50
        self.idle_temp = 30

        # Episode control
        self.max_steps = 200

        # Action space
        # 0 = FIFO
        # 1 = Highest Compute
        # 2 = Increase Power Mode
        # 3 = Decrease Power Mode
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([16, 16, 1, 100, 250, 5], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.memory_used = 0.0
        self.temperature = self.idle_temp
        self.power_draw = self.base_power
        self.utilization = 0.0
        self.power_mode = 1
        self.active_job = None
        self.job_queue = [generate_job() for _ in range(5)]

        self.current_step = 0

        return self._get_state(), {}

    def step(self, action):
        reward = 0
        jobs_completed = 0

        # ---- Power Mode Actions ----
        if action == 2:
            self.power_mode = min(2, self.power_mode + 1)
        elif action == 3:
            self.power_mode = max(0, self.power_mode - 1)

        # ---- Scheduling ----
        elif self.active_job is None:
            if action == 0:
                selected_job = self.job_queue[0]
            elif action == 1:
                selected_job = max(self.job_queue, key=lambda j: j["compute"])
            else:
                selected_job = None

            if selected_job:
                if selected_job["memory"] <= (self.vram - self.memory_used):
                    self.active_job = selected_job
                    self.memory_used = selected_job["memory"]
                    self.job_queue.remove(selected_job)
                else:
                    reward -= 10  # OOM penalty

        # ---- Job Update ----
        if self.active_job:
            self.active_job["remaining_time"] -= 1
            if self.active_job["remaining_time"] <= 0:
                jobs_completed = 1
                self.memory_used = 0
                self.active_job = None
                self.job_queue.append(generate_job())

        # ---- Utilization ----
        if self.active_job:
            multiplier = [0.8, 1.0, 1.2][self.power_mode]
            self.utilization = min(1.0, self.active_job["compute"] * multiplier)
        else:
            self.utilization = 0.0

        # ---- Power ----
        self.power_draw = self.base_power + 150 * self.utilization
        self.power_draw = min(self.max_power, self.power_draw)

        # ---- Temperature ----
        cooling = 0.1 * (self.temperature - self.idle_temp)
        self.temperature += 0.01 * self.power_draw - cooling
        self.temperature = max(self.idle_temp, min(100, self.temperature))

        # ---- Reward ----
        reward += jobs_completed
        reward -= 0.002 * self.power_draw

        if self.temperature > self.thermal_limit:
            reward -= 2

        # ---- Episode Step Tracking ----
        self.current_step += 1

        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_state(), reward, terminated, truncated, {}

    def _get_state(self):
        return np.array([
            self.memory_used,
            self.vram - self.memory_used,
            self.utilization,
            self.temperature,
            self.power_draw,
            len(self.job_queue)
        ], dtype=np.float32)