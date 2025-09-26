# SO(2)–SPD–SO(2) Polar Decomposition with Linear RNN Scan

## Overview
This document describes a method for combining **rotary positional embeddings (RoPE)** with a **linear recurrent scan** using **polar decomposition**. The key idea is to represent sequence position updates as a structured operator:

$$
h_t = R_t \, S_t \, R_t^\top h_{t-1} + B_t x_t
$$

where:
- **SO(2):** rotation matrices encoding positional shifts (RoPE).
- **SPD:** diagonal positive-definite scaling matrices (learned gates/decays).
- **Linear RNN scan:** associative recurrence enabling parallelism in TSLIB.

---

## Mathematical Formulation

### 1. Inputs
Sequence: 
$$
X = (x_1, x_2, \dots, x_T), \quad x_t \in \mathbb{R}^d
$$

### 2. RoPE (SO(2))
Rotary positional encoding applies a rotation in 2D subspaces:
$$
\text{RoPE}(x_{t}[2i], x_{t}[2i+1]) = (x_{t}[2i]\cos \theta_t - x_{t}[2i+1]\sin \theta_t,\; x_{t}[2i]\sin \theta_t + x_{t}[2i+1]\cos \theta_t)
$$

with frequency-based angle:
$$
\theta_t = t / (10000^{2i/d})
$$

### 3. SPD Gating
Introduce a diagonal gating matrix:
$$
S_t = \text{diag}(\sigma(g_t))
$$

where \(g_t = W_g x_t\).

### 4. Polar Decomposition
The combined operator is:
$$
A_t = R_t S_t R_t^\top
$$

yielding the recurrence:
\[
h_t = A_t h_{t-1} + B_t x_t
\]

This is linear, associative, and compatible with scan parallelization.

---

## Attention Compatibility

Queries and keys are modified as:
\[
q_t = (x_t W_q) R_t S_t, \quad k_i = (x_i W_k) R_i S_i
\]

Attention score:
\[
\langle q_t, k_i \rangle = (x_t W_q R_t S_t)(x_i W_k R_i S_i)^\top
\]

- **SO(2):** introduces relative positional phase (as in RoPE).
- **SPD:** adds per-dimension decay/gating.

---

## TSLIB Implementation Sketch

```python
import torch
import torch.nn as nn
from tslib.scan import associative_scan

def apply_rope(x, t):
    d = x.size(-1)
    half = d // 2
    freq = torch.arange(half, device=x.device) / half
    theta = t[:, None] / (10000 ** freq[None])
    cos, sin = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

class SO2SPDCell(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)  # SPD diagonal

    def forward(self, x, t):
        q = apply_rope(self.W_q(x), t)
        k = apply_rope(self.W_k(x), t)
        S = torch.sigmoid(self.gate(x))  # SPD diag entries
        q, k = q * S, k * S
        return q, k, self.W_v(x)

class RopePolarScan(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cell = SO2SPDCell(d_model)

    def forward(self, x):
        B, T, D = x.shape
        times = torch.arange(T, device=x.device)

        def transition(h, x_t_tuple):
            x_t, t = x_t_tuple
            q, k, v = self.cell(x_t, t)
            return h + torch.einsum("bd,bd->b", q, k)[:, None] * v

        h0 = torch.zeros(B, D, device=x.device)
        h_seq = associative_scan(transition, h0, list(zip(x.transpose(0,1), times)))
        return h_seq
```

---

## Intuition
- **SO(2):** rotate vectors to encode sequence order.
- **SPD:** scale/decay certain feature dimensions dynamically.
- **SO(2)–SPD–SO(2):** rotate → filter → rotate back.
- The scan makes it efficient: O(T) time with GPU parallelism.

---