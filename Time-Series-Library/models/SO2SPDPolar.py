import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class SO2SPDPolarLayer(nn.Module):
    """
    Single SO2-SPD Polar layer implementing:
    h_t = R_t S_t R_t^T h_{t-1} + B_t x_t
    """

    def __init__(self, d_model, dropout=0.1, n_head=1):
        super(SO2SPDPolarLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate = dropout

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # SPD gating projection
        self.gate_proj = nn.Linear(d_model, d_model)

        # Input projection for scan recurrence
        self.p_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)

        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_mlp = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.attn_type = 'softmax'

        self.clock = nn.Linear(d_model, n_head)

    def forward(self, x):
        B, T, D = x.shape
        residual = x

        # Apply layer normalization
        x_norm = self.ln_attn(x)

        # Generate time positions (cache this if possible)
        times = torch.arange(T, device=x.device, dtype=x.dtype)

        # Fuse QKV projections for better memory access
        q_out = self.W_q(x_norm).view(B, T, self.n_head, -1)
        k_out = self.W_k(x_norm).view(B, T, self.n_head, -1)
        clock = F.softplus(self.clock(x_norm)).view(B, T, self.n_head, -1)

        k_out = F.normalize(k_out, dim=-1, p=2)
        q_out = F.normalize(q_out, dim=-1, p=2)

        v = self.W_v(x_norm).view(B, T, self.n_head, -1)

        # Apply RoPE and SPD to Q, K
        q = self.apply_rope(q_out, times)     # [B, T, H, D]
        k = self.apply_rope(k_out, times)     # [B, T, H, D]
    #
        gj = -F.softplus(self.gate_proj(x_norm)).view(B, T, self.n_head, -1)
        gj_cumsum = gj.cumsum(dim=1).clip(-60, 50)     # [B, T, H, D]
        gj_cumprod = torch.exp(gj_cumsum)
        q = q * gj_cumprod
        k = k / (gj_cumprod + 1e-8)

        p = self.p_proj(x_norm).view(B, T, self.n_head, -1)
        p_max = p.max(dim=1, keepdim=True).values
        p_exp = torch.exp(p - p_max) * clock
        p_exp_cumsum = p_exp.cumsum(dim=1)
        q = q / (p_exp_cumsum + 1e-8)
        k = k * p_exp

        if self.attn_type == 'softmax':
            output = self.softmax_attention_scan(q, k, v)
        elif self.attn_type == 'linear':
            output = self.linear_attention_scan(q, k, v)

        # Output projection and residual connection
        output = output.reshape(B, T, -1)
        output = self.dropout(self.out_proj(output))

        x = output + residual

        x = x + self.mlp(self.ln_mlp(x))
        return x

    def softmax_attention_scan(self, q, k, v):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, dropout_p=self.dropout_rate if self.training else 0.0).transpose(1, 2)
        return out

    def linear_attention_scan(self, q, k, v, associative_scan=False):
        if associative_scan:
            kv = torch.einsum('blhd,blhe->blhde', k, v)
            kv = kv.cumsum(dim=1)
            out = torch.einsum('blhd,blhde->blhe', q, kv)
        else:
            qk = torch.einsum('blhd,bihd->bhli', q, k)
            qk = torch.tril(qk)
            out = torch.einsum('bhli,bihd->blhd', qk, v)
        return out

    def apply_rope(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE) to input tensor with multi-head support

        Args:
            x: Input tensor [B, T, H, D_head]
            times: Time positions [T]

        Returns:
            Rotated tensor with same shape as input
        """
        input_shape = x.shape

        # Handle multi-head case: [B, T, H, D_head] -> [B*H, T, D_head]
        if len(input_shape) == 4:
            B, T, H, D_head = input_shape
            x = x.reshape(B, T * H, D_head)
            D = D_head
        else:
            # Single-head case: [B, T, D]
            B, T, D = input_shape

        half_dim = D // 2

        # Compute frequencies - use more efficient exp-based approach
        freqs = torch.exp(
            -torch.arange(half_dim, device=x.device, dtype=x.dtype) * (math.log(10000.0) / half_dim)
        )
        theta = times[:, None] * freqs[None, :]  # [T, D//2]

        # Compute cos and sin in one pass
        cos_theta = torch.cos(theta)  # [T, D//2]
        sin_theta = torch.sin(theta)  # [T, D//2]

        # Reshape for efficient rotation - use view instead of split
        x_reshaped = x[..., :2*half_dim].view(x.shape[0], -1, half_dim, 2)
        x1 = x_reshaped[..., 0]  # [B, T(*H), D//2]
        x2 = x_reshaped[..., 1]  # [B, T(*H), D//2]

        # Apply rotation with fused operations
        x1_rot = x1 * cos_theta - x2 * sin_theta
        x2_rot = x1 * sin_theta + x2 * cos_theta

        # Stack and reshape efficiently
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).view(x.shape[0], -1, 2*half_dim)

        # Handle odd dimensions if needed
        if D % 2 != 0:
            x_rot = torch.cat([x_rot, x[..., -1:]], dim=-1)

        # Restore original shape if multi-head
        if len(input_shape) == 4:
            x_rot = x_rot.reshape(B, T, H, D)

        return x_rot




class Model(nn.Module):
    """
    SO(2)-SPD-SO(2)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_head = configs.n_heads

        self.input_norm = nn.LayerNorm(configs.d_model)
        self.output_norm = nn.LayerNorm(configs.d_model)

        # Embedding layer
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # SO2-SPD Polar layers
        self.layers = nn.ModuleList([
            SO2SPDPolarLayer(configs.d_model, configs.dropout, configs.n_heads)
            for _ in range(configs.e_layers)
        ])

        # Output projections for different tasks
        self.projection = nn.Parameter(torch.zeros(configs.d_model, configs.c_out)) # nn.Linear(configs.d_model, configs.c_out, bias=True)

        # Forecast layer
        self.forecast_layer = nn.Sequential(nn.Linear(configs.seq_len, 512),
                                            nn.GELU(),
                                            nn.Dropout(configs.dropout),
                                            nn.Linear(512, configs.pred_len))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = (x_enc.std(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Embedding
        x = self.embedding(x_enc, x_mark_enc)
        x = self.input_norm(x)

        # SO2-SPD Polar layers
        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        # Output projection
        output = x @ self.projection # self.projection(x)

        output = self.forecast_layer((x_enc + output).permute(0, 2, 1)).permute(0, 2, 1)

        # Denormalization
        output = output * std_enc + mean_enc
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
