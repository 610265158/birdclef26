import math

import torch
import torch.nn as nn


def precompute_2d_rope_freqs(h, w, head_dim, theta=10000.0, device=None):
    assert head_dim % 4 == 0, 'head_dim must be divisible by 4 for 2D RoPE'
    quarter = head_dim // 4

    freqs = 1.0 / (theta ** (torch.arange(0, quarter, device=device).float() / quarter))

    rows = torch.arange(h, device=device).float()
    cols = torch.arange(w, device=device).float()

    freqs_h = torch.outer(rows, freqs)
    freqs_w = torch.outer(cols, freqs)

    freqs_h = freqs_h.unsqueeze(1).expand(-1, w, -1).reshape(h * w, quarter)
    freqs_w = freqs_w.unsqueeze(0).expand(h, -1, -1).reshape(h * w, quarter)

    freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)
    freqs_2d = torch.cat([freqs_2d, freqs_2d], dim=-1)

    return freqs_2d.cos(), freqs_2d.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class MLP(nn.Module):
    def __init__(self, hidden_size=512, ratio=4, bias=False, proj_drop=0.0):
        super().__init__()
        intermediate_size = hidden_size * ratio
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.init_weights()

    def forward(self, hidden_state):
        gate_output = self.act_fn(self.gate_proj(hidden_state))
        up_output = self.up_proj(hidden_state)
        output = self.down_proj(gate_output * up_output)
        return self.proj_drop(output)

    def init_weights(self):
        std = 0.02
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=std / math.sqrt(2))
        for bias in (self.gate_proj.bias, self.up_proj.bias, self.down_proj.bias):
            if bias is not None:
                nn.init.zeros_(bias)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
