import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.base_trainer.models.perch_classifier import (
    RMSNorm,
    MLP,
    rotate_half,
)


class SEDSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop
        self.init_weights()

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_cos is not None and rope_sin is not None:
            cos = rope_cos.unsqueeze(0).unsqueeze(0)
            sin = rope_sin.unsqueeze(0).unsqueeze(0)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, S, D)
        x = self.proj(x)
        return self.proj_drop(x)

    def init_weights(self):
        std = 0.02
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=std / math.sqrt(2))
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class SEDBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, mlp_ratio=4, bias=False,
                 eps=1e-6, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps=eps)
        self.attention = SEDSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(
            hidden_size=hidden_size,
            ratio=mlp_ratio,
            bias=bias,
            proj_drop=proj_drop,
        )

    def forward(self, hidden_states, rope_cos=None, rope_sin=None):
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states), rope_cos, rope_sin
        )
        hidden_states = hidden_states + self.mlp(self.ffn_norm(hidden_states))
        return hidden_states


class SEDHead(nn.Module):
    def __init__(self, dim, num_classes, drop=0.5):
        super().__init__()
        self.fc_att = nn.Linear(dim, num_classes, bias=True)
        self.fc_logit = nn.Linear(dim, num_classes, bias=True)
        self.drop = drop
        nn.init.trunc_normal_(self.fc_att.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc_logit.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = F.dropout(x, p=self.drop, training=self.training)
        frame_logits = self.fc_logit(x)
        att_weights = torch.softmax(self.fc_att(x), dim=1)
        att_clip_logits = (att_weights * frame_logits).sum(dim=1)
        max_clip_logits = frame_logits.max(dim=1)[0]
        return att_clip_logits, max_clip_logits
