"""PyTorch implementation sketch for the CortiFormer method.

This file is intentionally a logic-level reference. Some details are not
recoverable from the manuscript's text export (e.g. exact projection widths,
normalization placement and several formula constants); comments mark those
places for verification against the typeset equations and RT-DETR source.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def conv_bn(in_ch: int, out_ch: int, kernel: int, stride: int = 1,
            groups: int = 1, activation: bool = False) -> nn.Sequential:
    padding = kernel // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if activation:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init_value: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class Lifting1D(nn.Module):
    """Learnable reversible even/odd lifting along one spatial axis."""

    def __init__(self, channels: int, axis: str):
        super().__init__()
        self.axis = axis
        self.predict = nn.Conv2d(channels, channels, (3, 1) if axis == "h" else (1, 3),
                                 padding=(1, 0) if axis == "h" else (0, 1),
                                 groups=channels, bias=False)
        self.update = nn.Conv2d(channels, channels, (3, 1) if axis == "h" else (1, 3),
                                padding=(1, 0) if axis == "h" else (0, 1),
                                groups=channels, bias=False)
        # Haar-like initialization; boundaries use replicate padding through the
        # convolutions' ordinary zero padding here and can be matched to the source.
        with torch.no_grad():
            self.predict.weight.zero_()
            self.update.weight.zero_()
            if axis == "h":
                self.predict.weight[:, 0, 1, 0] = 1.0
                self.update.weight[:, 0, 1, 0] = 0.5
            else:
                self.predict.weight[:, 0, 0, 1] = 1.0
                self.update.weight[:, 0, 0, 1] = 0.5

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.axis == "h":
            return x[:, :, 0::2, :], x[:, :, 1::2, :]
        return x[:, :, :, 0::2], x[:, :, :, 1::2]

    def _interleave(self, even: Tensor, odd: Tensor) -> Tensor:
        dim = 2 if self.axis == "h" else 3
        stacked = torch.stack((even, odd), dim=dim + 1)
        shape = list(even.shape)
        shape[dim] *= 2
        return stacked.flatten(dim, dim + 1).reshape(shape)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        even, odd = self._split(x)
        detail = odd - self.predict(even)
        low = even + self.update(detail)
        return low, detail

    def inverse(self, low: Tensor, detail: Tensor) -> Tensor:
        even = low - self.update(detail)
        odd = detail + self.predict(even)
        return self._interleave(even, odd)


class SubbandAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(),
                                 nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x * self.mlp(pooled).unsqueeze(-1).unsqueeze(-1)


class LLWD(nn.Module):
    """Learnable lifting-wavelet downsampling with subband channel attention."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.height_lift = Lifting1D(in_ch, "h")
        self.width_lift_low = Lifting1D(in_ch, "w")
        self.width_lift_high = Lifting1D(in_ch, "w")
        self.attention = SubbandAttention(in_ch * 4)
        self.project = conv_bn(in_ch * 4, out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        low_h, high_h = self.height_lift(x)
        ll, lh = self.width_lift_low(low_h)
        hl, hh = self.width_lift_high(high_h)
        bands = torch.cat((ll, lh, hl, hh), dim=1)
        return self.project(self.attention(bands))


def haar2d(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fixed Haar decomposition; input H/W must be even."""
    x00, x01 = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    x10, x11 = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def inverse_haar2d(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    b, c, h, w = ll.shape
    out = ll.new_empty((b, c, h * 2, w * 2))
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out


def window_partition(x: Tensor, window: int) -> tuple[Tensor, tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (window - h % window) % window
    pad_w = (window - w % window) % window
    x = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.reshape(b, c, hp // window, window, wp // window, window)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window, c)
    return x, (b, h, w, hp)


def window_reverse(tokens: Tensor, shape: tuple[int, int, int, int],
                   channels: int, window: int) -> Tensor:
    b, h, w, hp = shape
    wp = math.ceil(w / window) * window
    x = tokens.reshape(b, hp // window, wp // window, window, window, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, channels, hp, wp)
    return x[:, :, :h, :w]


class TokenMixer(nn.Module):
    def __init__(self, channels: int, mode: str, heads: int = 4, window: int = 8):
        super().__init__()
        self.mode, self.window = mode, window
        if mode == "conv":
            self.mixer: nn.Module = nn.Conv2d(channels, channels, 3, padding=1,
                                               groups=channels, bias=False)
        else:
            self.mixer = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "conv":
            return self.mixer(x)
        b, c, h, w = x.shape
        if self.mode == "window":
            tokens, meta = window_partition(x, self.window)
            mixed, _ = self.mixer(tokens, tokens, tokens, need_weights=False)
            return window_reverse(mixed, meta, c, self.window)
        tokens = x.flatten(2).transpose(1, 2)
        mixed, _ = self.mixer(tokens, tokens, tokens, need_weights=False)
        return mixed.transpose(1, 2).reshape(b, c, h, w)


class SOCR(nn.Module):
    """Structure-tensor-guided orientation continuity residual (logic sketch)."""

    def __init__(self, channels: int, lengths: Sequence[int] = (7, 11)):
        super().__init__()
        self.channels = channels
        self.gradient = nn.Conv2d(channels, 2, 1, bias=False)
        self.gaussian = nn.Conv2d(3, 3, 5, padding=2, groups=3, bias=False)
        with torch.no_grad():
            self.gaussian.weight.fill_(1.0 / 25.0)
        self.lengths = tuple(lengths)
        self.direction_kernels = nn.ParameterList()
        self.direction_mask_names: list[str] = []
        for length in self.lengths:
            kernel = torch.zeros(4, 1, length, length)
            masks = torch.zeros_like(kernel)
            center = length // 2
            # Four rasterized unoriented lines: horizontal, vertical, and diagonals.
            offsets = [
                [(0, dx) for dx in range(-center, center + 1) if dx != 0],
                [(dy, 0) for dy in range(-center, center + 1) if dy != 0],
                [(d, d) for d in range(-center, center + 1) if d != 0],
                [(d, -d) for d in range(-center, center + 1) if d != 0],
            ]
            for direction, coords in enumerate(offsets):
                for dy, dx in coords:
                    if 0 <= center + dy < length and 0 <= center + dx < length:
                        masks[direction, 0, center + dy, center + dx] = 1.0
                # Normalize each predictor's initial contribution.
                denom = masks[direction].sum().clamp_min(1.0)
                kernel[direction] = masks[direction] / denom
            self.direction_kernels.append(nn.Parameter(kernel))
            name = f"direction_mask_{length}"
            self.register_buffer(name, masks, persistent=False)
            self.direction_mask_names.append(name)
        self.predict_fuse = nn.Conv2d(channels * len(self.lengths), channels, 1)
        self.evidence = nn.Sequential(nn.Conv2d(2, 1, 1), nn.Sigmoid())
        self.enhance = nn.Conv2d(channels * 2, channels, 1)

    def _structure(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Haar high bands approximate directional first differences; this compact
        # implementation estimates gradients from the feature map itself.
        gray = x.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        jxx = self.gaussian(dx.square().expand(-1, 3, -1, -1))[:, :1]
        jyy = self.gaussian(dy.square().expand(-1, 3, -1, -1))[:, :1]
        jxy = self.gaussian((dx * dy).expand(-1, 3, -1, -1))[:, :1]
        denom = torch.sqrt((jxx - jyy).square() + 4 * jxy.square() + 1e-8)
        coherence = denom / (jxx + jyy + 1e-8)
        # Double-angle representation avoids computing an angle explicitly.
        cos2 = (jxx - jyy) / (denom + 1e-8)
        sin2 = 2.0 * jxy / (denom + 1e-8)
        preset = x.new_tensor([0.0, math.pi / 2, math.pi / 4, -math.pi / 4])
        score = (cos2 * torch.cos(2 * preset)[None, :, None, None]
                 + sin2 * torch.sin(2 * preset)[None, :, None, None])
        routing = torch.softmax(4.0 * coherence * score, dim=1)
        return coherence.clamp(0, 1), routing, dx, dy

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        coherence, routing, _, _ = self._structure(x)
        predictions: list[Tensor] = []
        for length, parameter, mask_name in zip(self.lengths, self.direction_kernels,
                                                 self.direction_mask_names):
            mask = getattr(self, mask_name)
            # The center is masked after every parameter update, preserving the
            # non-identity predictor constraint from the manuscript.
            weights = parameter * mask.to(parameter.dtype)
            by_direction = []
            for d in range(4):
                weight = weights[d:d + 1].expand(self.channels, 1, -1, -1)
                by_direction.append(F.conv2d(x, weight, padding=length // 2,
                                              groups=self.channels))
            stacked = torch.stack(by_direction, dim=1)
            predictions.append((stacked * routing.unsqueeze(2)).sum(dim=1))
        predicted = self.predict_fuse(torch.cat(predictions, dim=1))
        residual = x - predicted
        gate = self.evidence(torch.cat((residual.abs().mean(1, keepdim=True), coherence), dim=1))
        enhanced = x + self.enhance(torch.cat((x, gate * residual), dim=1))
        return enhanced, residual, coherence, gate


class ConvFFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.net = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                 nn.GELU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FDCPBlock(nn.Module):
    """Frequency-decoupled continuity perception block."""

    def __init__(self, channels: int, stage: int, heads: int = 4):
        super().__init__()
        mode = "conv" if stage == 1 else "window" if stage == 2 else "global"
        self.position = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.low_norm = nn.LayerNorm(channels)
        self.low_mixer = TokenMixer(channels, mode, heads=heads)
        self.socr = SOCR(channels)
        self.low_to_high = nn.Conv2d(channels, channels, 1)
        self.high_to_low = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(channels, channels), nn.Sigmoid())
        self.high_project = nn.Conv2d(channels * 3, channels, 1)
        self.reconstruct = nn.Conv2d(channels, channels * 3, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = ConvFFN(channels)
        self.scale1 = LayerScale(channels)
        self.scale2 = LayerScale(channels)

    @staticmethod
    def _ln(x: Tensor, norm: nn.LayerNorm) -> Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        ll, lh, hl, hh = haar2d(x)
        low = ll + self.position(ll)
        low = low + self.low_mixer(self._ln(low, self.low_norm))
        high, residual, coherence, gate = self.socr(self.high_project(torch.cat((lh, hl, hh), 1)))

        # Cross-frequency mutual modulation (CFMM): low semantic map gates
        # high-frequency evidence; high-frequency energy recalibrates low channels.
        spatial_prior = torch.sigmoid(self.low_to_high(low).mean(1, keepdim=True))
        high = high * (1.0 + spatial_prior)
        channel_gate = self.high_to_low(high).view(x.shape[0], -1, 1, 1)
        low = low * (1.0 + channel_gate)

        delta = self.reconstruct(high)
        d_lh, d_hl, d_hh = delta.chunk(3, dim=1)
        out = inverse_haar2d(low, lh + d_lh, hl + d_hl, hh + d_hh)
        out = x + self.scale1(self.norm2(out))
        return out + self.scale2(self.ffn(out))


class WCPNet(nn.Module):
    """Four-stage backbone returning S3/S4/S5 RT-DETR feature maps."""

    def __init__(self, in_channels: int = 3,
                 channels: Sequence[int] = (64, 128, 256, 512),
                 depths: Sequence[int] = (2, 2, 4, 2)):
        super().__init__()
        # Two stem reductions give the standard stride-4 input to Stage 1;
        # LLWD then yields Stage2/3/4 outputs at strides 8/16/32.
        self.stem = nn.Sequential(conv_bn(in_channels, channels[0] // 2, 3, 2, activation=True),
                                  conv_bn(channels[0] // 2, channels[0], 3, 2, activation=True))
        self.stages = nn.ModuleList([
            nn.Sequential(*(FDCPBlock(channels[i], i + 1) for _ in range(depths[i])))
            for i in range(4)
        ])
        self.downsample = nn.ModuleList([
            LLWD(channels[i], channels[i + 1]) for i in range(3)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:  # stage 2, 3, 4 map to RT-DETR S3, S4, S5
                features.append(x)
            if i < 3:
                x = self.downsample[i](x)
        return tuple(features)  # type: ignore[return-value]


class PrecisionEstimator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.net = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1, groups=1),
                                 nn.SiLU(), nn.Conv2d(hidden, hidden, 1),
                                 nn.SiLU(), nn.Conv2d(hidden, 1, 1), nn.Softplus())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + 1e-6


class DCUFStage(nn.Module):
    def __init__(self, channels: int, sparse_channels: int | None = None):
        super().__init__()
        sparse_channels = sparse_channels or channels
        self.sparse_in = nn.Sequential(nn.Conv2d(channels, sparse_channels, 3, padding=1,
                                                 groups=channels if sparse_channels == channels else 1),
                                       nn.Conv2d(sparse_channels, sparse_channels, 1))
        self.sparse_out = nn.Sequential(nn.Conv2d(sparse_channels, sparse_channels, 1),
                                        nn.Conv2d(sparse_channels, channels, 3, padding=1))
        nn.init.zeros_(self.sparse_out[-1].weight)
        nn.init.zeros_(self.sparse_out[-1].bias)
        self.threshold = nn.Parameter(torch.full((1, sparse_channels, 1, 1), 0.01))
        self.step = nn.Parameter(torch.tensor(0.5))

    def proximal(self, corrected: Tensor) -> Tensor:
        sparse = self.sparse_in(corrected)
        shrunk = torch.sign(sparse) * F.relu(sparse.abs() - F.softplus(self.threshold))
        return corrected + self.sparse_out(shrunk)


class DCUF(nn.Module):
    """Dual-consistency proximal-gradient unfolding fusion node.

    For a top-down node, pass deep features at half the shallow resolution.
    Bottom-up nodes use `bottom_up=True` and reverse those roles.
    """

    def __init__(self, deep_channels: int, shallow_channels: int, out_channels: int,
                 stages: int = 2, bottom_up: bool = False):
        super().__init__()
        self.bottom_up = bottom_up
        self.deep_project = nn.Conv2d(deep_channels, out_channels, 1)
        self.shallow_project = nn.Conv2d(shallow_channels, out_channels, 1)
        self.spatial = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1,
                                 groups=out_channels, bias=False)
        semantic_groups = 8 if out_channels % 8 == 0 else 1
        self.semantic = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                                  groups=semantic_groups, bias=False)
        self.precision_deep = PrecisionEstimator(out_channels)
        self.precision_shallow = PrecisionEstimator(out_channels)
        self.stages = nn.ModuleList([DCUFStage(out_channels) for _ in range(stages)])
        self.post = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels), nn.SiLU())

    def _spatial_forward(self, z: Tensor) -> Tensor:
        if not self.bottom_up:
            return self.spatial(z)
        # The bottom-up dual uses the transpose of the top-down stride-2
        # operator as its observation operator.
        return F.conv_transpose2d(z, self.spatial.weight, stride=2, padding=1,
                                  output_padding=1, groups=self.spatial.groups)

    def _spatial_adjoint(self, value: Tensor, output_size: tuple[int, int]) -> Tensor:
        if self.bottom_up:
            return F.conv2d(value, self.spatial.weight, stride=2, padding=1,
                            groups=self.spatial.groups)
        # Choose output_padding so the transpose maps exactly to the unknown's
        # current resolution, including odd sizes if the data pipeline has them.
        base_h = (value.shape[-2] - 1) * 2 - 2 + 2 + 1
        base_w = (value.shape[-1] - 1) * 2 - 2 + 2 + 1
        op_h = output_size[0] - base_h
        op_w = output_size[1] - base_w
        return F.conv_transpose2d(value, self.spatial.weight, stride=2, padding=1,
                                  output_padding=(op_h, op_w),
                                  groups=self.spatial.groups)

    def _semantic_forward(self, z: Tensor) -> Tensor:
        return F.conv2d(z, self.semantic.weight, padding=1, groups=self.semantic.groups)

    def _semantic_adjoint(self, value: Tensor) -> Tensor:
        return F.conv_transpose2d(value, self.semantic.weight, padding=1,
                                  groups=self.semantic.groups)

    def forward(self, deep: Tensor, shallow: Tensor) -> Tensor:
        deep = self.deep_project(deep)
        shallow = self.shallow_project(shallow)
        # Top-down: deep is low-resolution, shallow is high-resolution, and z
        # lives on the shallow grid. Bottom-up: deep is the high-resolution
        # observation, shallow is the lower-resolution current-level feature,
        # and z lives on the shallow grid. The operators below are adjoint pairs.
        size_z = shallow.shape[-2:]
        z = self._spatial_adjoint(deep, size_z) + self._semantic_adjoint(shallow)
        wd, ws = self.precision_deep(deep), self.precision_shallow(shallow)
        for stage in self.stages:
            ed = self._spatial_forward(z) - deep
            es = self._semantic_forward(z) - shallow
            correction = (self._spatial_adjoint(wd * ed, z.shape[-2:])
                          + self._semantic_adjoint(ws * es))
            z = stage.proximal(z - stage.step * correction)
        return self.post(z)


class HORep(nn.Module):
    """Hotelling-observer re-parameterized convolution logic sketch.

    `update_background_statistics` is called during training with a feature map,
    a background-position mask, encoder scores and a foreground mask. The
    deploy path folds the fixed observer templates into the ordinary Conv-BN
    branches. Exact Ultralytics RepConv integration is adapter-specific.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8,
                 prototypes: int = 3, hard_weight: float = 2.0,
                 momentum: float = 0.99, shrinkage: float = 0.1):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.groups, self.num_prototypes = groups, prototypes
        self.hard_weight, self.momentum, self.shrinkage = hard_weight, momentum, shrinkage
        self.base3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.base1 = nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.patch_dim = (in_channels // groups) * 9
        self.out_per_group = out_channels // groups
        self.direction = nn.Parameter(torch.randn(out_channels, self.patch_dim) * 0.01)
        self.pwt_scale = nn.Parameter(torch.zeros(out_channels, 1))
        self.pwt_bias = nn.Parameter(torch.zeros(out_channels))
        self.hp_coeff = nn.Parameter(torch.zeros(out_channels, prototypes))
        self.register_buffer("bg_mean", torch.zeros(groups, self.patch_dim))
        self.register_buffer("bg_cov", torch.eye(self.patch_dim).repeat(groups, 1, 1))
        self.register_buffer("prototypes", torch.zeros(groups, prototypes, self.patch_dim))
        self.register_buffer("prototype_ready", torch.zeros(groups, prototypes, dtype=torch.bool))
        self.register_buffer("stats_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("stats_frozen", torch.tensor(False))
        self.act = nn.SiLU(inplace=True)
        self._deploy_conv: nn.Conv2d | None = None

    @torch.no_grad()
    def update_background_statistics(self, x: Tensor, background: Tensor,
                                     encoder_scores: Tensor, foreground: Tensor) -> None:
        if bool(self.stats_frozen):
            return
        b, _, h, w = x.shape
        patches = F.unfold(x, kernel_size=3, padding=1)
        patches = patches.reshape(b, self.groups, self.patch_dim, h * w)
        bg_mask = background.reshape(b, 1, h * w).expand(-1, self.groups, -1).bool()
        fg_mask = foreground.reshape(b, 1, h * w).expand(-1, self.groups, -1).bool()
        scores = encoder_scores.detach().reshape(b, 1, h * w).clamp(0, 1)
        weights = (scores + 1e-4).pow(self.hard_weight).expand(-1, self.groups, -1)

        for group in range(self.groups):
            values = patches[:, group].transpose(1, 2).reshape(-1, self.patch_dim)
            w_bg = weights[:, group].reshape(-1) * bg_mask[:, group].reshape(-1)
            valid = w_bg > 0
            if valid.sum() > 1:
                v, wv = values[valid], w_bg[valid]
                mean = (v * wv[:, None]).sum(0) / wv.sum().clamp_min(1e-8)
                centered = v - mean
                cov = (centered * wv[:, None]).T @ centered / wv.sum().clamp_min(1e-8)
                eye = torch.eye(self.patch_dim, device=x.device, dtype=x.dtype)
                cov = (1.0 - self.shrinkage) * cov + self.shrinkage * eye
                self.bg_mean[group].lerp_(mean, 1.0 - self.momentum)
                self.bg_cov[group].lerp_(cov, 1.0 - self.momentum)

            fg_values = patches[:, group].transpose(1, 2).reshape(-1, self.patch_dim)
            fg_values = fg_values[fg_mask[:, group].reshape(-1)]
            if fg_values.numel() == 0:
                continue
            # Online cosine k-means in whitened coordinates (small deterministic
            # initialization followed by EMA updates is a replaceable detail).
            cov_inv_sqrt = self._inverse_sqrt(self.bg_cov[group])
            white = F.normalize((fg_values - self.bg_mean[group]) @ cov_inv_sqrt, dim=-1)
            ready = self.prototype_ready[group]
            if not ready.all():
                take = min(int((~ready).sum()), white.shape[0])
                slots = torch.where(~ready)[0][:take]
                self.prototypes[group, slots] = fg_values[:take]
                self.prototype_ready[group, slots] = True
                continue
            pwhite = F.normalize((self.prototypes[group] - self.bg_mean[group]) @ cov_inv_sqrt, dim=-1)
            assignment = (white @ pwhite.T).argmax(dim=1)
            for prototype_id in range(self.num_prototypes):
                selected = fg_values[assignment == prototype_id]
                if selected.numel():
                    center = selected.mean(dim=0)
                    self.prototypes[group, prototype_id].lerp_(center, 1.0 - self.momentum)
        self.stats_updates.add_(1)

    @staticmethod
    def _inverse_sqrt(covariance: Tensor) -> Tensor:
        values, vectors = torch.linalg.eigh(covariance.float())
        values = values.clamp_min(1e-5)
        return (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T

    def _observer_weights(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weights_pwt, biases_pwt, weights_hp, biases_hp = [], [], [], []
        for group in range(self.groups):
            q = self._inverse_sqrt(self.bg_cov[group]).to(self.direction.dtype)
            mean = self.bg_mean[group].to(self.direction.dtype)
            start = group * self.out_per_group
            end = start + self.out_per_group
            direction = F.normalize(self.direction[start:end], dim=-1)
            w = direction @ q
            scale = self.pwt_scale[start:end]
            wp = scale * w
            bp = self.pwt_bias[start:end] - (wp * mean).sum(-1)
            weights_pwt.append(wp)
            biases_pwt.append(bp)

            proto = self.prototypes[group].to(self.direction.dtype)
            proto_white = (proto - mean) @ q
            template_white = F.normalize(proto_white, dim=-1)
            hp = template_white @ q
            coeff = self.hp_coeff[start:end]
            # Each output channel learns a mixture over the morphology prototypes.
            mixed_hp = coeff @ hp
            bhp = -(mixed_hp * mean).sum(-1)
            weights_hp.append(mixed_hp)
            biases_hp.append(bhp)
        wp = torch.cat(weights_pwt).reshape(self.out_channels, self.in_channels // self.groups, 3, 3)
        wh = torch.cat(weights_hp).reshape(self.out_channels, self.in_channels // self.groups, 3, 3)
        return wp, torch.cat(biases_pwt), wh, torch.cat(biases_hp)

    def forward(self, x: Tensor) -> Tensor:
        if self._deploy_conv is not None:
            return self.act(self._deploy_conv(x))
        y = self.bn3(self.base3(x))
        y = y + self.bn1(self.base1(x))
        wp, bp, wh, bh = self._observer_weights()
        y = y + F.conv2d(x, wp, bp, padding=1, groups=self.groups)
        y = y + F.conv2d(x, wh, bh, padding=1, groups=self.groups)
        return self.act(y)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        weight = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros(
            conv.out_channels, device=weight.device, dtype=weight.dtype
        )
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_weight = weight * scale[:, None, None, None]
        fused_bias = (bias - bn.running_mean) * scale + bn.bias
        return fused_weight, fused_bias

    @torch.no_grad()
    def freeze_statistics(self) -> None:
        self.stats_frozen.fill_(True)

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Fold fixed observer templates and Conv-BN branches into one 3x3 conv."""
        w3, b3 = self._fuse_conv_bn(self.base3, self.bn3)
        w1, b1 = self._fuse_conv_bn(self.base1, self.bn1)
        w1 = F.pad(w1, (1, 1, 1, 1))
        wp, bp, wh, bh = self._observer_weights()
        weight = w3 + w1 + wp + wh
        bias = b3 + b1 + bp + bh
        conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1,
                         groups=self.groups, bias=True).to(weight.device, weight.dtype)
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
        self._deploy_conv = conv


@torch.no_grad()
def horep_position_masks(boxes_per_image: Sequence[Tensor], feature_hw: tuple[int, int],
                         stride: float, device: torch.device,
                         expansion: float = 1.5) -> tuple[Tensor, Tensor]:
    """Project GT boxes to HORep feature positions.

    Foreground positions have centers inside an erosion box. Background
    positions are outside every box enlarged around its center by `expansion`.
    Inputs are xyxy boxes in original-image pixel coordinates.
    """
    height, width = feature_hw
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    foreground = torch.zeros((len(boxes_per_image), height, width),
                             dtype=torch.bool, device=device)
    background = torch.ones_like(foreground)

    for batch_index, boxes in enumerate(boxes_per_image):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1), (y2 - y1)
        for i in range(boxes.shape[0]):
            inside = ((xx >= x1[i]) & (xx <= x2[i])
                      & (yy >= y1[i]) & (yy <= y2[i]))
            enlarged = ((xx >= cx[i] - expansion * bw[i] / 2)
                        & (xx <= cx[i] + expansion * bw[i] / 2)
                        & (yy >= cy[i] - expansion * bh[i] / 2)
                        & (yy <= cy[i] + expansion * bh[i] / 2))
            foreground[batch_index] |= inside
            background[batch_index] &= ~enlarged
    return foreground, background


class CortiFormer(nn.Module):
    """Composition adapter around an existing RT-DETR implementation.

    The baseline owns RT-DETR's AIFI, query selection, decoder, matcher and
    criterion. Replace only its backbone, CCFM resize/concat nodes and RepC3
    RepConv modules as described in the manuscript.
    """

    def __init__(self, rtdetr_detector: nn.Module, backbone: nn.Module | None = None):
        super().__init__()
        self.detector = rtdetr_detector
        self.backbone = backbone or WCPNet()
        # Integration is intentionally explicit: find/replace the backbone and
        # four CCFM nodes in the chosen Ultralytics 8.3.78 RT-DETR implementation.
        # The module paths differ across releases; see README.md before wiring.

    def forward(self, images: Tensor, targets: list[dict] | None = None):
        # Adapter pseudocode: supply WCPNet's S3/S4/S5 features to the original
        # hybrid encoder and pass its outputs to the unchanged RT-DETR decoder.
        features = self.backbone(images)
        return self.detector(features, targets=targets)


def iter_horep_modules(module: nn.Module) -> Iterable[HORep]:
    for child in module.modules():
        if isinstance(child, HORep):
            yield child
