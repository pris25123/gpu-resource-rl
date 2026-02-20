import streamlit as st
import requests
import numpy as np
import pandas as pd
from simulator.environment import GPUEnvironment
import plotly.graph_objects as go


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="GPU RL Console", layout="wide")

st.title("⚡ GPU Energy-Aware RL Scheduler Console")
st.caption("Multi-objective comparison of RL-based GPU scheduling vs FIFO baseline")

steps = st.slider("Simulation Steps", 50, 1000, 300)

if st.button("🚀 Run Simulation"):

    # ================= RL SIMULATION =================
    env_rl = GPUEnvironment()
    obs_rl, _ = env_rl.reset()

    rl_powers, rl_temps = [], []
    rl_jobs = 0

    for _ in range(steps):
        response = requests.post(API_URL, json={"state": obs_rl.tolist()})
        action = response.json()["action"]

        obs_rl, reward, terminated, truncated, _ = env_rl.step(action)

        rl_powers.append(env_rl.power_draw)
        rl_temps.append(env_rl.temperature)

        if reward > 0:
            rl_jobs += 1

        if truncated:
            obs_rl, _ = env_rl.reset()

    # ================= FIFO SIMULATION =================
    env_fifo = GPUEnvironment()
    obs_fifo, _ = env_fifo.reset()

    fifo_powers, fifo_temps = [], []
    fifo_jobs = 0

    for _ in range(steps):
        action = 0  # FIFO policy
        obs_fifo, reward, terminated, truncated, _ = env_fifo.step(action)

        fifo_powers.append(env_fifo.power_draw)
        fifo_temps.append(env_fifo.temperature)

        if reward > 0:
            fifo_jobs += 1

        if truncated:
            obs_fifo, _ = env_fifo.reset()

    # ================= METRICS =================
    rl_avg_power = np.mean(rl_powers)
    fifo_avg_power = np.mean(fifo_powers)

    rl_avg_temp = np.mean(rl_temps)
    fifo_avg_temp = np.mean(fifo_temps)

    power_savings = ((fifo_avg_power - rl_avg_power) / fifo_avg_power) * 100
    throughput_delta = rl_jobs - fifo_jobs

    # ================= KPI SECTION =================
    st.markdown("## 📊 System Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "⚡ RL Avg Power (W)",
        f"{rl_avg_power:.2f}",
        f"{power_savings:.2f}% vs FIFO"
    )

    col2.metric(
        "🌡 RL Avg Temp (°C)",
        f"{rl_avg_temp:.2f}",
        f"{(fifo_avg_temp - rl_avg_temp):.2f}°C vs FIFO"
    )

    col3.metric(
        "✅ Jobs Completed",
        f"{rl_jobs}",
        f"{throughput_delta} vs FIFO"
    )

    # ================= TRADEOFF ANALYSIS =================
    st.markdown("## ⚖ Energy - Throughput Trade-off Analysis")

    st.write(
        f"The RL scheduler reduced average power consumption by **{power_savings:.2f}%**, "
        f"while completing **{abs(throughput_delta)} fewer jobs** compared to FIFO "
        f"over {steps} simulation steps."
    )

    st.info(
        "This demonstrates reward-driven multi-objective optimization. "
        "The RL agent prioritizes energy efficiency while maintaining competitive throughput. "
        "Adjusting reward weights enables explicit control over this trade-off."
    )

    # ================= ENERGY BAR =================
    st.markdown("## 🔋 Power Optimization Impact")
    st.progress(min(max(power_savings / 20, 0), 1))
    st.write(f"Net Energy Reduction: **{power_savings:.2f}%** relative to FIFO baseline.")

    # ================= TRADEOFF SCATTER =================
    st.markdown("## 📌 Power vs Throughput Trade-off")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[rl_avg_power],
        y=[rl_jobs],
        mode='markers+text',
        marker=dict(size=18, color='#00CC96'),
        text=["RL Policy"],
        textposition="top center",
        name="RL Policy"
    ))

    fig.add_trace(go.Scatter(
        x=[fifo_avg_power],
        y=[fifo_jobs],
        mode='markers+text',
        marker=dict(size=18, color='#EF553B'),
        text=["FIFO Baseline"],
        textposition="top center",
        name="FIFO Baseline"
    ))

    fig.update_layout(
        xaxis_title="Average Power (W)",
        yaxis_title="Jobs Completed",
        template="plotly_dark",
        height=500,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= TIME SERIES VISUALS =================
    st.markdown("## 📈 Power Consumption Over Time")
    st.line_chart({
        "RL Policy": rl_powers,
        "FIFO Baseline": fifo_powers
    })

    st.markdown("## 🌡 Temperature Over Time")
    st.line_chart({
        "RL Policy": rl_temps,
        "FIFO Baseline": fifo_temps
    })

    # ================= THERMAL STATUS =================
    st.markdown("## 🖥 Thermal Status")

    if rl_avg_temp < 70:
        st.success("Thermal levels stable and within safe operating range.")
    elif rl_avg_temp < 85:
        st.warning("Temperature approaching thermal threshold.")
    else:
        st.error("Thermal limit exceeded — risk of throttling.")

    # ================= FOOTER =================
    st.markdown("---")
    st.caption(
        "Deployed PPO RL Inference via FastAPI | "
        "Energy-aware reward shaping | "
        "Multi-objective GPU scheduling optimization"
    )