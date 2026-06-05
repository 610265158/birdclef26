import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_config import config as cfg
from lib.core.base_trainer.models.perch_classifier import precompute_2d_rope_freqs
from lib.core.base_trainer.models.perch_sed import SEDBlock, SEDHead
from perch_torch.perch_pytorch import PerchV2, SimpleMelspec


class PerchContinuousSED(nn.Module):
    def __init__(self, num_classes=None, emb_dim=None, transformer_dim=None,
                 weights_path=None, num_heads=None, num_layers=None,
                 mlp_ratio=None, proj_drop=None, attn_drop=None, cls_drop=None,
                 num_context=None):
        super().__init__()

        num_classes = num_classes or cfg.DATA.num_classes
        emb_dim = emb_dim or cfg.MODEL.emb_dim
        transformer_dim = transformer_dim or cfg.MODEL.transformer_dim
        weights_path = weights_path or cfg.MODEL.weights_path
        num_heads = num_heads or cfg.MODEL.num_heads
        num_layers = num_layers or cfg.MODEL.num_layers
        mlp_ratio = mlp_ratio or cfg.MODEL.mlp_ratio
        proj_drop = proj_drop if proj_drop is not None else cfg.MODEL.proj_drop
        attn_drop = attn_drop if attn_drop is not None else cfg.MODEL.attn_drop
        cls_drop = cls_drop if cls_drop is not None else cfg.MODEL.cls_drop
        self.num_heads = num_heads
        self.num_context = num_context or cfg.MODEL.get('num_context', 4)

        perch = PerchV2()
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu')
            perch.load_state_dict(state, strict=True)
            print(f'[PerchContinuousSED] loaded pretrained weights from {weights_path}')
        else:
            print(f'[PerchContinuousSED] no pretrained weights loaded (weights_path={weights_path})')

        self.backbone = perch.backbone
        self.frontend = SimpleMelspec(
            features=256, stride=320, kernel_size=640, nfft=1024,
            sample_rate=32000, freq_range=(60, 16000), power=1.0,
            log_floor=1e-5, log_offset=0.0, log_scalar=0.1,
        )
        self.fc1 = nn.Linear(emb_dim, transformer_dim, bias=True)
        self.temporal_block = nn.ModuleList([
            SEDBlock(
                hidden_size=transformer_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
            for _ in range(num_layers)
        ])
        self.sed_head = SEDHead(transformer_dim, num_classes, drop=cls_drop)

    def apply_spec_aug(self, spec):
        if not self.training:
            return spec

        B, T, F = spec.shape

        if cfg.TRAIN.spec_time_mask_prob > 0 and T > 1:
            max_w = max(1, int(T * cfg.TRAIN.spec_time_mask_ratio))
            apply = torch.rand(B, device=spec.device) < cfg.TRAIN.spec_time_mask_prob
            widths = torch.randint(1, max_w + 1, (B,), device=spec.device)
            starts = torch.randint(0, max(1, T - max_w + 1), (B,), device=spec.device)
            t_idx = torch.arange(T, device=spec.device).unsqueeze(0)
            mask = (t_idx >= starts.unsqueeze(1)) & (t_idx < (starts + widths).unsqueeze(1))
            mask = mask & apply.unsqueeze(1)
            spec = spec.masked_fill(mask.unsqueeze(2), 0.0)

        if cfg.TRAIN.spec_freq_mask_prob > 0 and F > 1:
            max_w = max(1, int(F * cfg.TRAIN.spec_freq_mask_ratio))
            apply = torch.rand(B, device=spec.device) < cfg.TRAIN.spec_freq_mask_prob
            widths = torch.randint(1, max_w + 1, (B,), device=spec.device)
            starts = torch.randint(0, max(1, F - max_w + 1), (B,), device=spec.device)
            f_idx = torch.arange(F, device=spec.device).unsqueeze(0)
            mask = (f_idx >= starts.unsqueeze(1)) & (f_idx < (starts + widths).unsqueeze(1))
            mask = mask & apply.unsqueeze(1)
            spec = spec.masked_fill(mask.unsqueeze(1), 0.0)

        return spec

    def forward(self, audio, skip_frontend=False):
        B = audio.shape[0]
        nc = self.num_context

        if skip_frontend:
            spec = audio
        else:
            with torch.cuda.amp.autocast(enabled=False):
                spec = self.frontend(audio.float())

        spec = self.apply_spec_aug(spec)
        spatial = self.backbone(spec.unsqueeze(1))

        feat_h, feat_w = spatial.shape[2], spatial.shape[3]
        remainder = feat_h % nc
        if remainder != 0:
            pad_h = nc - remainder
            spatial = F.pad(spatial, (0, 0, 0, pad_h))
            feat_h = feat_h + pad_h

        tokens = spatial.flatten(2).transpose(1, 2)
        tokens = self.fc1(tokens)

        head_dim = tokens.size(-1) // self.num_heads
        rope_cos, rope_sin = precompute_2d_rope_freqs(
            feat_h, feat_w, head_dim, device=tokens.device
        )
        rope_cos = rope_cos.to(tokens.dtype)
        rope_sin = rope_sin.to(tokens.dtype)

        for block in self.temporal_block:
            tokens = block(tokens, rope_cos, rope_sin)

        chunk_h = feat_h // nc
        tokens_per_chunk = chunk_h * feat_w
        tokens_chunked = tokens.reshape(B, nc, tokens_per_chunk, -1)
        tokens_flat = tokens_chunked.reshape(B * nc, tokens_per_chunk, -1)

        att_logits, max_logits = self.sed_head(tokens_flat)
        att_logits = att_logits.reshape(B, nc, -1)
        max_logits = max_logits.reshape(B, nc, -1)

        return att_logits, max_logits
