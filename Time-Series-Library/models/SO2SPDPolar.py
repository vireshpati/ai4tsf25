import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from einops import einsum


class Model(nn.Module):
    """
    SO(2)-SPD-SO(2) Polar Decomposition with Linear RNN Scan
    Combines rotary positional embeddings (RoPE) with linear recurrent scan
    using polar decomposition for efficient sequence modeling.

    Paper implementation based on METHOD.md specification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        # Embedding layer
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # SO2-SPD Polar RNN layers
        self.layers = nn.ModuleList([
            SO2SPDPolarLayer(configs.d_model, configs.dropout)
            for _ in range(configs.e_layers)
        ])

        # Output projections for different tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Embedding
        x = self.embedding(x_enc, x_mark_enc)

        # SO2-SPD Polar layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        output = self.projection(x)

        # Denormalization
        output = output * std_enc + mean_enc
        return output

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        x = self.embedding(x_enc, x_mark_enc)

        # SO2-SPD Polar layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        output = self.projection(x)
        return output

    def anomaly_detection(self, x_enc):
        # Embedding
        x = self.embedding(x_enc, None)

        # SO2-SPD Polar layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        output = self.projection(x)
        return output

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        x = self.embedding(x_enc, None)

        # SO2-SPD Polar layers
        for layer in self.layers:
            x = layer(x)

        # Output
        output = self.act(x)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class SO2SPDPolarLayer(nn.Module):
    """
    Single SO2-SPD Polar layer implementing:
    h_t = R_t S_t R_t^T h_{t-1} + B_t x_t
    """

    def __init__(self, d_model, dropout=0.1):
        super(SO2SPDPolarLayer, self).__init__()
        self.d_model = d_model

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # SPD gating projection
        self.gate_proj = nn.Linear(d_model, d_model)

        # Input projection for scan recurrence
        self.B_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        residual = x

        # Apply layer normalization
        x = self.norm(x)

        # Generate time positions
        times = torch.arange(T, device=x.device, dtype=x.dtype)

        # Apply RoPE and SPD to Q, K
        q = self.apply_rope_spd(self.W_q(x), times)  # [B, T, D]
        k = self.apply_rope_spd(self.W_k(x), times)  # [B, T, D]
        v = self.W_v(x)  # [B, T, D]

        # Compute polar decomposition scan
        output = self.polar_scan(q, k, v, x)

        # Output projection and residual connection
        output = self.out_proj(output)
        output = self.dropout(output)

        return output + residual

    def apply_rope_spd(self, x, times):
        """
        Apply RoPE (SO(2)) rotations and SPD gating
        """
        B, T, D = x.shape

        # Apply RoPE rotations
        x_rope = apply_rope(x, times)

        # Apply SPD gating (diagonal positive scaling)
        gates = torch.sigmoid(self.gate_proj(x))  # Positive gates via sigmoid
        x_spd = x_rope * gates

        return x_spd

    def polar_scan(self, q, k, v, x):
        """
        Implement the polar decomposition scan:
        h_t = A_t h_{t-1} + B_t x_t
        where A_t = R_t S_t R_t^T
        """
        B, T, D = q.shape

        # Initialize hidden state
        h = torch.zeros(B, D, device=q.device, dtype=q.dtype)
        outputs = []

        # Sequential scan (can be parallelized using associative scan)
        for t in range(T):
            # Compute attention-like interaction
            attention_weight = torch.sum(q[:, t] * k[:, t], dim=-1, keepdim=True) / math.sqrt(D)
            attention_weight = torch.softmax(attention_weight, dim=0)

            # Update hidden state with polar decomposition structure
            B_t = self.B_proj(x[:, t])  # Input projection
            h = attention_weight * h + B_t  # Linear recurrence

            # Output computation
            output_t = h * v[:, t]  # Element-wise product with values
            outputs.append(output_t)

        return torch.stack(outputs, dim=1)  # [B, T, D]


def apply_rope(x, times):
    """
    Apply Rotary Position Embedding (RoPE) to input tensor

    Args:
        x: Input tensor [B, T, D]
        times: Time positions [T]

    Returns:
        Rotated tensor [B, T, D]
    """
    B, T, D = x.shape
    half_dim = D // 2

    # Compute frequencies
    freqs = torch.arange(half_dim, device=x.device, dtype=x.dtype) / half_dim
    theta = times[:, None] / (10000 ** freqs[None, :])  # [T, D//2]

    # Compute cos and sin
    cos_theta = torch.cos(theta)  # [T, D//2]
    sin_theta = torch.sin(theta)  # [T, D//2]

    # Split input into pairs
    x1 = x[..., :half_dim]  # [B, T, D//2]
    x2 = x[..., half_dim:2*half_dim]  # [B, T, D//2]

    # Apply rotation
    x1_rot = x1 * cos_theta[None, :, :] - x2 * sin_theta[None, :, :]
    x2_rot = x1 * sin_theta[None, :, :] + x2 * cos_theta[None, :, :]

    # Concatenate rotated pairs
    if D % 2 == 0:
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
    else:
        # Handle odd dimensions
        x_odd = x[..., 2*half_dim:]
        x_rot = torch.cat([x1_rot, x2_rot, x_odd], dim=-1)

    return x_rot


def build_freqs_cis(seq_len, d_model):
    """
    Build frequency components for RoPE
    """
    theta = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float32) * (math.log(10000.0) / d_model))
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    freqs = pos * theta[None, :]
    return torch.cos(freqs), torch.sin(freqs)


class PolarAttentionHead(nn.Module):
    """
    Single attention head with polar decomposition (RoPE + SPD)
    """

    def __init__(self, d_model, d_head, use_spd=True):
        super(PolarAttentionHead, self).__init__()
        self.d_head = d_head
        self.use_spd = use_spd

        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)

        if use_spd:
            self.gamma = nn.Parameter(torch.zeros(d_head))  # SPD log-scale parameters

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).unsqueeze(1)  # [B, 1, T, d_head]
        K = self.k_proj(x).unsqueeze(1)  # [B, 1, T, d_head]
        V = self.v_proj(x).unsqueeze(1)  # [B, 1, T, d_head]

        # Apply RoPE
        cos_b, sin_b = cos[None, None, :, :], sin[None, None, :, :]
        Qr = rotary_apply_tensor(Q, cos_b, sin_b)
        Kr = rotary_apply_tensor(K, cos_b, sin_b)

        # Apply SPD gating
        if self.use_spd:
            gate = self.gamma.exp()  # Positive diagonal scaling
            Qr = Qr * gate
            Kr = Kr * gate

        # Compute attention
        scores = (Qr @ Kr.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        output = (attn @ V).squeeze(1)  # [B, T, d_head]

        return output


def rotary_apply_tensor(x, cos, sin):
    """
    Apply rotary embedding to tensor with cos/sin precomputed
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)