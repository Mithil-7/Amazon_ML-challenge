"""
Minimal, dependency-free (pure NumPy) neural-net building blocks.

Why NumPy instead of torch/tensorflow: this sandbox has ~1 CPU core, ~3GB
free disk and no GPU, and `pip install torch` pulls the CUDA build (several
GB) which does not fit here and isn't needed for a CPU demo anyway. Rather
than skip the DL piece, this module hand-implements the handful of ops the
architecture needs (dense layers, sigmoid/ReLU, embedding lookup+pool,
Adam) with explicit forward/backward passes, vectorized over batches with
NumPy. On a real GPU box, `dl_matcher.py` will transparently use torch
instead if it's importable -- see the `_HAS_TORCH` branch there -- so
nothing here is wasted work; it's the CPU-only fallback path.
"""
import numpy as np

RNG = np.random.default_rng(42)


def glorot(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return RNG.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(x, 0.0)


class Dense:
    """y = act(x @ W + b). Caches what's needed for backward()."""

    def __init__(self, fan_in, fan_out, act="linear"):
        self.W = glorot(fan_in, fan_out)
        self.b = np.zeros((fan_out,), dtype=np.float32)
        self.act = act
        self._cache = None

    def forward(self, x):
        z = x @ self.W + self.b
        if self.act == "relu":
            a = relu(z)
        elif self.act == "sigmoid":
            a = sigmoid(z)
        else:
            a = z
        self._cache = (x, z, a)
        return a

    def backward(self, d_out):
        x, z, a = self._cache
        if self.act == "relu":
            dz = d_out * (z > 0)
        elif self.act == "sigmoid":
            dz = d_out * a * (1 - a)
        else:
            dz = d_out
        dW = x.T @ dz / x.shape[0]
        db = dz.mean(axis=0)
        dx = dz @ self.W.T
        return dx, {"W": dW, "b": db}

    def params(self):
        return {"W": self.W, "b": self.b}

    def set_params(self, p):
        self.W, self.b = p["W"], p["b"]


class Adam:
    """Adam optimizer over a flat dict of {name: (param_array, grad_array)}."""

    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, named_params_and_grads):
        self.t += 1
        for name, (param, grad) in named_params_and_grads.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * grad
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (grad ** 2)
            m_hat = self.m[name] / (1 - self.b1 ** self.t)
            v_hat = self.v[name] / (1 - self.b2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def hash_trigrams(text: str, n_buckets: int) -> np.ndarray:
    """Hash every character trigram of `text` into [0, n_buckets)."""
    text = f"^{text}$"  # boundary markers so prefixes/suffixes matter
    if len(text) < 3:
        grams = [text]
    else:
        grams = [text[i:i + 3] for i in range(len(text) - 2)]
    return np.array([hash(g) % n_buckets for g in grams], dtype=np.int64)


def embed_and_pool(indices_list, table: np.ndarray):
    """
    indices_list: list of 1D int arrays (variable length per sample) of
    trigram bucket ids. table: (V, d) embedding matrix.
    Returns: pooled (N, d) mean-pooled embeddings ("Deep Averaging Network"
    encoder), plus the indices_list itself (needed for the sparse backward).
    """
    d = table.shape[1]
    n = len(indices_list)
    pooled = np.zeros((n, d), dtype=np.float32)
    for i, idx in enumerate(indices_list):
        if len(idx) == 0:
            continue
        pooled[i] = table[idx].mean(axis=0)
    return pooled


def embed_pool_backward(indices_list, table_shape, d_pooled):
    """Scatter-add the pooled gradient back to the (sparse) embedding rows
    that were averaged, scaled by 1/len for each sample (mean-pool grad)."""
    grad_table = np.zeros(table_shape, dtype=np.float32)
    for i, idx in enumerate(indices_list):
        if len(idx) == 0:
            continue
        grad_table[idx] += (d_pooled[i] / len(idx))
    return grad_table
