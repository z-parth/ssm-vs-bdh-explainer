import numpy as np

class SSMEngine:
    """Continuous-time State Space Model discretized via Zero-Order Hold (ZOH)."""
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.h = np.zeros(state_dim, dtype=np.float32)
        # Diagonal state matrix A and input projection B
        self.A = np.random.uniform(0.1, 1.0, size=state_dim).astype(np.float32)
        self.B = np.random.uniform(0.5, 1.0, size=state_dim).astype(np.float32)

    def reset(self):
        self.h = np.zeros(self.state_dim, dtype=np.float32)

    def step(self, x: np.ndarray, delta: float) -> np.ndarray:
        """Computes discrete recurrence step: h_t = exp(-delta * A) * h_{t-1} + delta * B * x."""
        A_bar = np.exp(-delta * self.A)
        B_bar = self.B * delta
        self.h = (A_bar * self.h) + (B_bar * x)
        return self.h.copy()