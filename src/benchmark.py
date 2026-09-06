import numpy as np
from scipy.spatial.distance import cosine
from src.ssm_engine import SSMEngine
from src.bdh_engine import BDHEngine

def run_recall_test(dim: int, noise_steps: int, delta: float, eta: float):
    """Encodes a target associative pair, injects distractor noise, and returns recall scores."""
    ssm = SSMEngine(state_dim=dim)
    bdh = BDHEngine(dim=dim)

    # Ground truth pair: Cue -> Value
    cue = np.random.uniform(0.0, 1.0, size=dim).astype(np.float32)
    val = np.random.uniform(0.0, 1.0, size=dim).astype(np.float32)

    # Encode target association
    ssm.step(cue + val, delta=delta)
    bdh.step(cue + val, eta=eta)

    # Inject random distractor noise
    for _ in range(noise_steps):
        noise = np.random.uniform(0.0, 1.0, size=dim).astype(np.float32)
        ssm.step(noise, delta=delta)
        bdh.step(noise, eta=eta)

    # Probe recall
    ssm_pred = ssm.h
    bdh_pred = bdh.query(cue)

    # Cosine fidelity (1 - cosine distance)
    ssm_score = float(1.0 - cosine(val, ssm_pred)) if np.any(ssm_pred) else 0.0
    bdh_score = float(1.0 - cosine(val, bdh_pred)) if np.any(bdh_pred) else 0.0

    return max(0.0, ssm_score), max(0.0, bdh_score)