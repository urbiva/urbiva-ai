"""UVA-Net S10: Urban Vision Accessibility network, small, under 10M parameters.

Architecture: an efficient convolutional backbone built from inverted residual
blocks with depthwise convolutions, squeeze-and-excitation and hard-swish
activations, followed by a compact classification head over the seven
categories of the app taxonomy. The width multiplier scales every stage; the
constructor refuses any configuration above the parameter budget.

This module defines the architecture only. No pretrained or trained weights
are shipped with this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("urbiva_ai.models requires PyTorch: pip install 'urbiva-ai[torch]'") from exc

from urbiva_ai.config import MAX_PARAMETERS
from urbiva_ai.labels import NUM_CLASSES


def make_divisible(v: float, divisor: int = 8) -> int:
    """Round channels to a multiple of divisor without dropping more than 10 %."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@dataclass(frozen=True)
class BlockSpec:
    kernel: int
    expand: int
    out: int
    se: bool
    hswish: bool
    stride: int


# kernel, expanded channels, output channels, SE, hard-swish, stride
S10_BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec(3, 16, 16, False, False, 1),
    BlockSpec(3, 64, 24, False, False, 2),
    BlockSpec(3, 72, 24, False, False, 1),
    BlockSpec(5, 72, 40, True, False, 2),
    BlockSpec(5, 120, 40, True, False, 1),
    BlockSpec(5, 120, 40, True, False, 1),
    BlockSpec(3, 240, 80, False, True, 2),
    BlockSpec(3, 200, 80, False, True, 1),
    BlockSpec(3, 184, 80, False, True, 1),
    BlockSpec(3, 184, 80, False, True, 1),
    BlockSpec(3, 480, 112, True, True, 1),
    BlockSpec(3, 672, 112, True, True, 1),
    BlockSpec(5, 672, 160, True, True, 2),
    BlockSpec(5, 960, 160, True, True, 1),
    BlockSpec(5, 960, 160, True, True, 1),
)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = make_divisible(channels / reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * s


def conv_bn_act(cin: int, cout: int, k: int, stride: int = 1, groups: int = 1, act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
    ]
    if act is not None:
        layers.append(act)
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, spec: BlockSpec, width: float):
        super().__init__()
        exp = make_divisible(spec.expand * width)
        cout = make_divisible(spec.out * width)
        act = nn.Hardswish if spec.hswish else nn.ReLU
        layers: list[nn.Module] = []
        if exp != cin:
            layers.append(conv_bn_act(cin, exp, 1, act=act(inplace=True)))
        layers.append(conv_bn_act(exp, exp, spec.kernel, spec.stride, groups=exp, act=act(inplace=True)))
        if spec.se:
            layers.append(SqueezeExcite(exp))
        layers.append(conv_bn_act(exp, cout, 1, act=None))
        self.block = nn.Sequential(*layers)
        self.use_residual = spec.stride == 1 and cin == cout
        self.out_channels = cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class UVANetS10(nn.Module):
    """Image classifier: (N, 3, H, W) normalised RGB to (N, num_classes) logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, width_multiplier: float = 1.0, dropout: float = 0.2,
                 head_channels: int = 1024):
        super().__init__()
        stem = make_divisible(16 * width_multiplier)
        self.stem = conv_bn_act(3, stem, 3, 2, act=nn.Hardswish(inplace=True))
        blocks: list[nn.Module] = []
        cin = stem
        for spec in S10_BLOCKS:
            b = InvertedResidual(cin, spec, width_multiplier)
            blocks.append(b)
            cin = b.out_channels
        self.blocks = nn.Sequential(*blocks)
        last = make_divisible(6 * cin)
        self.last = conv_bn_act(cin, last, 1, act=nn.Hardswish(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(last, head_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(head_channels, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.last(self.blocks(self.stem(x)))).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(num_classes: int = NUM_CLASSES, width_multiplier: float = 1.0, dropout: float = 0.2,
                max_parameters: int = MAX_PARAMETERS) -> UVANetS10:
    """Build UVA-Net S10 and enforce the parameter budget."""
    model = UVANetS10(num_classes=num_classes, width_multiplier=width_multiplier, dropout=dropout)
    n = count_parameters(model)
    limit = min(max_parameters, MAX_PARAMETERS)
    if n > limit:
        raise ValueError(f"UVA-Net S10 with width {width_multiplier} has {n:,} parameters; limit is {limit:,}")
    return model
