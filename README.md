# Rotary Position Embedding with Polar Decomposition (SO(2) × SPD)

## 1) Attention from First Principles  
Scaled dot-product attention uses inner products of learned queries and keys. Without positional information, the mechanism is permutation-invariant.  

$$
q_t = x_t W_Q,\quad k_i = x_i W_K,\quad v_i = x_i W_V,\qquad
o_t=\sum_{i=1}^n \mathrm{softmax}_i\!\left(\tfrac{q_t^\top k_i}{\sqrt d}\right)\, v_i
$$

```python
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
attn   = scores.softmax(dim=-1)
O      = attn @ V
```

---

## 2) Classic Rotary Position Embedding (RoPE)  
RoPE encodes position by rotating queries and keys in 2-D subspaces of the embedding.  
The dot product after rotation depends only on relative positions.  

$$
q_t^{(\mathrm{rope})}=(x_t W_Q)R_t,\quad k_i^{(\mathrm{rope})}=(x_i W_K)R_i,\quad
(q_t^{\mathrm{rope}})^\top k_i^{\mathrm{rope}}=(x_t W_Q)^\top R_{i-t}(x_i W_K)
$$

```python
def rotary_apply(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return xr.flatten(-2)
```

---

## 3) Polar Decomposition of the Position Operator  
The positional operator can be generalized as a polar decomposition.  
Each operator is a product of a rotation U_t ∈ SO(2) and a symmetric positive definite gate P_t ∈ SPD.  

$$
M_t = U_t P_t,\quad
q_t=(x_t W_Q)\,U_t P_t,\quad k_i=(x_i W_K)\,U_i P_i
$$

---

## 4) Symmetric Positive Definite (SPD) Gate  
The SPD component acts as a diagonal scaling of each embedding dimension.  
This allows position-dependent modulation of contributions.  

$$
P_t=\mathrm{diag}(\exp(g(t)))
$$

```python
# SPD gate (diagonal, positive via exp)
gate_params = torch.zeros(num_heads, d)   # learnable
gate = gate_params.exp()
q = q_rot * gate
k = k_rot * gate
```

---

## 5) Linear Scan View  
With polar decomposition, attention scores resemble a linear recurrence with gating.  
Rotations preserve relative encoding, while SPD introduces time-aware weighting.  

$$
q_t^\top k_i = (x_t W_Q)^\top P_t U_{i-t} P_i (x_i W_K)
$$

---

## 6) End-to-End PyTorch Head  
A single attention head with rotation and SPD gating.  

```python
import math, torch, torch.nn as nn

def build_freqs_cis(seq_len, d):
    theta = torch.exp(-torch.arange(0, d, 2, dtype=torch.float32) * (math.log(10000.0) / d))
    pos   = torch.arange(seq_len, dtype=torch.float32)[:, None]
    ang   = pos * theta[None, :]
    return torch.cos(ang), torch.sin(ang)

def rotary_apply(x, cos, sin):  # x: (B,H,T,D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)

class PolarRoPEHead(nn.Module):
    def __init__(self, d_model, d_head, use_spd=True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.use_spd = use_spd
        if use_spd:
            self.gamma = nn.Parameter(torch.zeros(d_head))  # log-scale

    def forward(self, x, cos, sin):  # x: (B,T,d_model)
        B, T, _ = x.shape
        Q = self.q_proj(x).unsqueeze(1)
        K = self.k_proj(x).unsqueeze(1)
        V = self.v_proj(x).unsqueeze(1)

        cos_b, sin_b = cos[None, None, :, :], sin[None, None, :, :]
        Qr = rotary_apply(Q, cos_b, sin_b)
        Kr = rotary_apply(K, cos_b, sin_b)

        if self.use_spd:
            gate = self.gamma.exp()
            Qr = Qr * gate
            Kr = Kr * gate

        scores = (Qr @ Kr.transpose(-2, -1)) / math.sqrt(Qr.size(-1))
        attn   = scores.softmax(dim=-1)
        O      = (attn @ V).squeeze(1)
        return O
```

---

## 7) Multi-Head Wrapper  

```python
class PolarRoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, use_spd=True):
        super().__init__()
        self.heads = nn.ModuleList([PolarRoPEHead(d_model, d_head, use_spd) for _ in range(num_heads)])
        self.out = nn.Linear(num_heads * d_head, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        cos, sin = build_freqs_cis(T, self.heads[0].q_proj.out_features)
        cos, sin = cos.to(x.device), sin.to(x.device)
        outs = [h(x, cos, sin) for h in self.heads]
        O = torch.cat(outs, dim=-1)
        return self.out(O)

# sanity check
B,T,d_model,H,d_head = 2,128,256,4,64
x = torch.randn(B,T,d_model)
attn = PolarRoPEAttention(d_model,H,d_head,use_spd=True)
y = attn(x)  # (B,T,d_model)
```
