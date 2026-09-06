import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.benchmark import run_recall_test

st.set_page_config(page_title="SSM vs. BDH Memory Explainer", layout="wide")

st.title("SSM Continuous Decay vs. BDH Synaptic Plasticity")
st.caption("Testing associative recall degradation under increasing token noise.")

col1, col2, col3 = st.columns(3)
with col1:
    noise_tokens = st.slider("Distractor Noise Steps (N)", min_value=0, max_value=300, value=50, step=10)
with col2:
    delta_step = st.slider("SSM Discretization Step (Δ)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
with col3:
    eta_rate = st.slider("BDH Plasticity Rate (η)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

# Run live benchmark
ssm_acc, bdh_acc = run_recall_test(dim=16, noise_steps=noise_tokens, delta=delta_step, eta=eta_rate)

m1, m2 = st.columns(2)
m1.metric("SSM State Recall Fidelity", f"{ssm_acc:.4f}")
m2.metric("BDH-GPU Synaptic Recall Fidelity", f"{bdh_acc:.4f}")

# Sweep plot across noise tokens to show live decay curves
noise_range = list(range(0, noise_tokens + 1, max(1, noise_tokens // 10)))
ssm_curve = []
bdh_curve = []

for n in noise_range:
    s, b = run_recall_test(dim=16, noise_steps=n, delta=delta_step, eta=eta_rate)
    ssm_curve.append(s)
    bdh_curve.append(b)

fig = go.Figure()
fig.add_trace(go.Scatter(x=noise_range, y=ssm_curve, mode="lines+markers", name="SSM (Continuous Decay)"))
fig.add_trace(go.Scatter(x=noise_range, y=bdh_curve, mode="lines+markers", name="BDH-GPU (Synaptic Memory)"))
fig.update_layout(
    xaxis_title="Noise Steps Injected",
    yaxis_title="Cosine Similarity to Target Value",
    yaxis_range=[0, 1],
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)