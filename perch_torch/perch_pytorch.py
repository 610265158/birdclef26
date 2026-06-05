import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 32000

STEM_FEATURES_BASE = 32
HEAD_FEATURES_BASE = 1280
REDUCTION_RATIO = 4

STAGES_B0 = [
    (1, 16, 3, 3, 1, 1),
    (2, 24, 3, 3, 2, 6),
    (2, 40, 5, 5, 2, 6),
    (3, 80, 3, 3, 2, 6),
    (3, 112, 5, 5, 1, 6),
    (4, 192, 5, 5, 2, 6),
    (1, 320, 3, 3, 1, 6),
]

WIDTH_COEFF = 1.2
DEPTH_COEFF = 1.4

_MEL_HIGH_FREQUENCY_Q = 1127.0
_MEL_BREAK_FREQUENCY_HERTZ = 700.0


def round_features(features, width_coeff, depth_divisor=8):
    features *= width_coeff
    new_features = max(
        depth_divisor,
        int(features + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    if new_features < 0.9 * features:
        new_features += depth_divisor
    return int(new_features)


def round_num_blocks(num_blocks, depth_coeff):
    return int(math.ceil(depth_coeff * num_blocks))


def _mel_weight_matrix(
    num_mel_bins=128, num_spectrogram_bins=513,
    sample_rate=32000, lower_edge_hertz=60.0, upper_edge_hertz=16000.0,
):
    hertz_to_mel = lambda f: _MEL_HIGH_FREQUENCY_Q * np.log1p(f / _MEL_BREAK_FREQUENCY_HERTZ)
    bands_to_zero = 1
    nyquist = sample_rate / 2.0
    linear_freqs = np.linspace(0.0, nyquist, num_spectrogram_bins)[bands_to_zero:]
    spec_mel = hertz_to_mel(linear_freqs)[:, None]

    edges = np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    lower, center, upper = edges[None, :-2], edges[None, 1:-1], edges[None, 2:]

    weights = np.maximum(0.0, np.minimum(
        (spec_mel - lower) / (center - lower),
        (upper - spec_mel) / (upper - center),
    ))
    return np.pad(weights, ((bands_to_zero, 0), (0, 0))).astype(np.float32)


class SimpleMelspec(nn.Module):
    def __init__(self, features=128, stride=320, kernel_size=640, nfft=1024,
                 sample_rate=32000, freq_range=(60, 16000), power=1.0,
                 log_floor=1e-5, log_offset=0.0, log_scalar=0.1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.nfft = nfft
        self.power = power
        self.log_floor = log_floor
        self.log_offset = log_offset
        self.log_scalar = log_scalar

        window = np.hanning(kernel_size).astype(np.float32)
        window /= window.sum()
        self.register_buffer("window", torch.from_numpy(window))

        mel = _mel_weight_matrix(features, nfft // 2 + 1, sample_rate,
                                 float(freq_range[0]), float(freq_range[1]))
        self.register_buffer("mel_matrix", torch.from_numpy(mel))

    def forward(self, audio):
        batch_shape = audio.shape[:-1]
        T = audio.shape[-1]
        x = audio.reshape(-1, T)

        out_len = (T + self.stride - 1) // self.stride
        total_pad = max((out_len - 1) * self.stride + self.kernel_size - T, 0)
        x = F.pad(x, (total_pad // 2, total_pad - total_pad // 2))

        frames = x.unfold(1, self.kernel_size, self.stride)
        windowed = frames * self.window
        stfts = torch.fft.rfft(windowed, n=self.nfft, dim=-1)

        mags = stfts.real ** 2 + stfts.imag ** 2
        if self.power == 1.0:
            mags = torch.sqrt(mags)
        elif self.power != 2.0:
            mags = mags ** (self.power / 2.0)

        out = torch.matmul(mags, self.mel_matrix)
        out = self.log_scalar * torch.log(torch.clamp(out, min=self.log_floor) + self.log_offset)
        return out.reshape(batch_shape + out.shape[-2:])


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        self.reduce = nn.Linear(channels, channels // reduction_ratio)
        self.expand = nn.Linear(channels // reduction_ratio, channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = F.silu(self.reduce(s))
        s = torch.sigmoid(self.expand(s))
        return x * s[:, :, None, None]


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio,
                 reduction_ratio=REDUCTION_RATIO):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        expanded = in_ch * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_ch, expanded, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded, eps=1e-5, momentum=0.1)

        self.depthwise_conv = nn.Conv2d(expanded, expanded, kernel_size,
                                        stride=stride, padding=0,
                                        groups=expanded, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded, eps=1e-5, momentum=0.1)
        self.se = SqueezeAndExcitation(expanded, reduction_ratio * expand_ratio)
        self.project_conv = nn.Conv2d(expanded, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)

    def _pad(self, x):
        kh, kw = self.kernel_size
        if self.stride == 2:
            h, w = x.shape[2], x.shape[3]
            x = F.pad(x, (
                (kw // 2) - (1 - w % 2), kw // 2,
                (kh // 2) - (1 - h % 2), kh // 2,
            ))
        else:
            x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2))
        return x

    def forward(self, x):
        if self.expand_ratio != 1:
            x = F.silu(self.expand_bn(self.expand_conv(x)))
        x = F.silu(self.depthwise_bn(self.depthwise_conv(self._pad(x))))
        x = self.project_bn(self.project_conv(self.se(x)))
        return x


class EfficientNetB3(nn.Module):
    def __init__(self, include_top=False, survival_prob=1.):
        super().__init__()
        self.include_top = include_top
        self.survival_prob = survival_prob

        stem_ch = round_features(STEM_FEATURES_BASE, WIDTH_COEFF)
        head_ch = round_features(HEAD_FEATURES_BASE, WIDTH_COEFF)

        self.stem_conv = nn.Conv2d(1, stem_ch, 3, stride=2, padding=0, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_ch, eps=1e-5, momentum=0.1)

        self.blocks = nn.ModuleList()
        self._block_cfgs = []
        prev = stem_ch
        for nb0, f0, kh, kw, stride, er in STAGES_B0:
            nb = round_num_blocks(nb0, DEPTH_COEFF)
            feat = round_features(f0, WIDTH_COEFF)
            for bi in range(nb):
                s = stride if bi == 0 else 1
                self.blocks.append(MBConv(prev, feat, (kh, kw), s, er))
                self._block_cfgs.append(bi)
                prev = feat

        self.head_conv = nn.Conv2d(prev, head_ch, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(head_ch, eps=1e-5, momentum=0.1)

    def forward(self, x):
        x = F.silu(self.stem_bn(self.stem_conv(x)))
        for i, block in enumerate(self.blocks):
            bi = self._block_cfgs[i]
            y = block(x)
            if bi > 0 and self.survival_prob < 1.0 and self.training:
                if torch.rand(1).item() > self.survival_prob:
                    y = torch.zeros_like(y)
            x = y if bi == 0 else y + x
        x = F.silu(self.head_bn(self.head_conv(x)))
        return x.mean(dim=(2, 3)) if self.include_top else x


class PerchV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = SimpleMelspec(
            features=128, stride=320, kernel_size=640, nfft=1024,
            sample_rate=SAMPLE_RATE, freq_range=(60, 16000), power=1.0,
            log_floor=1e-5, log_offset=0.0, log_scalar=0.1,
        )
        self.backbone = EfficientNetB3(include_top=False)
