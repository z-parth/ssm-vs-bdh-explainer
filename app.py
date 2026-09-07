import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.benchmark import run_recall_test

# Hide sidebar by default and keep wide layout for the chart
st.set_page_config(page_title="Memory vs. Plasticity", layout="wide", initial_sidebar_state="collapsed")

# --- 1. CUSTOM CSS INJECTION ---
# Overrides default Streamlit styling for a polished, web-app feel
st.markdown("""
<style>
    /* Soften the typography and spacing */
    h1 { font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.2rem; color: #888; margin-bottom: 2rem; }
    
    /* Custom highlight box for the core claim instead of default blockquote */
    .claim-box {
        background-color: rgba(29, 161, 242, 0.1);
        border-left: 4px solid #1DA1F2;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Subtle footnote styling for the disclaimer */
    .footnote { 
        font-size: 0.85rem; 
        color: #666; 
        font-style: italic; 
        margin-top: 3rem;
        border-top: 1px solid #333;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. THE NARRATIVE GUIDE (Centered) ---
# Using columns to restrict text width makes it much easier to read
_, center_col, _ = st.columns([1, 3, 1])

with center_col:
    st.title("State-Space Memory vs. Synaptic Plasticity")
    st.markdown('<p class="subtitle">An interactive exploration of sequence recall under noise interference.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="claim-box">
        <strong>The Core Claim:</strong> When exposed to an increasing stream of distractor noise tokens, a continuous-time SSM's fixed hidden state decays associative recall accuracy faster than BDH-GPU's additive synaptic weight matrix.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **The Mechanism**  
    Standard State-Space Models (like Mamba) compress history into an algebraic state vector that continuously decays via exponential transition ($e^{-\Delta A}$). 
    In contrast, Pathway's BDH-GPU architecture relies on linear attention mapped as Hebbian synaptic plasticity ($\Delta W = \eta x x^T$). Rather than algebraically diluting a state vector, BDH physically writes memory into synaptic fast-weights, retaining specific semantic links even when flooded with unrelated noise tokens.
    """)

st.write("") # Spacer

# --- 3. INTERACTIVE SANDBOX (Bounded Container) ---
# Grouping the controls and metrics inside a visual box separates theory from practice
with st.container(border=True):
    st.subheader("Interactive Sandbox")
    st.markdown("Adjust the variables below to simulate how each architecture handles an influx of meaningless distractor tokens between a cue and its retrieval.")
    st.write("")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        # Added help tooltips to sliders
        noise_tokens = st.slider("Distractor Noise Steps (N)", 0, 300, 50, 10, help="Number of random vectors injected after the target.")
    with c2:
        delta_step = st.slider("SSM Discretization (Δ)", 0.01, 0.20, 0.05, 0.01, help="Controls the rate of exponential state decay in the SSM.")
    with c3:
        eta_rate = st.slider("BDH Plasticity (η)", 0.01, 0.50, 0.10, 0.01, help="Controls the Hebbian write strength in BDH.")

    # Run live test for metrics
    ssm_acc, bdh_acc = run_recall_test(dim=16, noise_steps=noise_tokens, delta=delta_step, eta=eta_rate)
    
    st.write("")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("SSM State Recall Fidelity", f"{ssm_acc:.4f}")
    m_col2.metric("BDH-GPU Synaptic Recall Fidelity", f"{bdh_acc:.4f}")

# --- 4. VISUALIZATION (Cleaned up Plotly theme) ---
noise_range = list(range(0, noise_tokens + 1, max(1, noise_tokens // 10)))
ssm_curve, bdh_curve = [], []

for n in noise_range:
    s_trials, b_trials = [], []
    for _ in range(5):
        s, b = run_recall_test(dim=16, noise_steps=n, delta=delta_step, eta=eta_rate)
        s_trials.append(s)
        b_trials.append(b)
    ssm_curve.append(np.mean(s_trials))
    bdh_curve.append(np.mean(b_trials))

fig = go.Figure()
# Thicker lines and specific colors
fig.add_trace(go.Scatter(x=noise_range, y=ssm_curve, mode="lines+markers", name="SSM (Continuous Decay)", line=dict(color="#FF4B4B", width=3)))
fig.add_trace(go.Scatter(x=noise_range, y=bdh_curve, mode="lines+markers", name="BDH-GPU (Synaptic Memory)", line=dict(color="#0068C9", width=3)))

# Transparent backgrounds and modern legend placement
fig.update_layout(
    xaxis_title="Noise Steps Injected (N)",
    yaxis_title="Cosine Similarity (5-trial Avg)",
    yaxis_range=[-0.05, 1.05],
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified"
)
# Subtle gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig, use_container_width=True)

# --- 5. DISCLOSURE FOOTNOTE ---
# Moved to the bottom to prevent it from ruining the intro hook
st.markdown("""
<p class="footnote">
* Mandatory Disclosure: This dashboard executes a simplified 16-dimensional toy mathematical analogue to isolate and demonstrate the Hebbian/sparse memory mechanisms. It is an independent educational reimplementation, NOT the official production BDH or BDH-CQ model.
</p>
""", unsafe_allow_html=True)