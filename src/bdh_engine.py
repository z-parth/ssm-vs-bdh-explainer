import numpy as np

class BDHEngine:
    """Linear attention via low-rank Hebbian synaptic plasticity."""
    def __init__(self, dim: int, sparsity_threshold: float = 0.5):
        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        # Synaptic weight matrix acting as working memory
        self.synapses = np.zeros((dim, dim), dtype=np.float32)

    def reset(self):
        self.synapses = np.zeros((self.dim, self.dim), dtype=np.float32)

    def step(self, x: np.ndarray, eta: float) -> np.ndarray:
        """Additive Hebbian write with non-negative sparse activation."""
        # ReLU + thresholding for biological sparse non-negative activation
        activations = np.maximum(0.0, x - self.sparsity_threshold)
        # Synaptic fast-weight update
        self.synapses += eta * np.outer(activations, activations)
        return self.synapses.copy()

    def query(self, cue: np.ndarray) -> np.ndarray:
        """Associative readout via matrix-vector multiplication."""
        activations = np.maximum(0.0, cue - self.sparsity_threshold)
        return self.synapses @ activations