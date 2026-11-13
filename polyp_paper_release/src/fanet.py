"""
FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation
Tomar et al., arXiv:2103.17235 / IEEE TNNLS 2022.

Faithful PyTorch port of the official implementation:
    https://github.com/nikhilroxtomar/FANet  (model.py + blocks.py, retrieved
    2026-10; the repository carries NO LICENSE file - code is re-implemented
    here with minimal, behavior-preserving adaptations for integration into
    this project; cite the paper above when used).

Adaptations (semantics preserved):
  1. Device-agnostic: the official code hard-codes `torch.cuda.FloatTensor`;
     replaced by dtype/device of the input tensors (enables CPU testing).
  2. `forward(image, mask)` takes the feedback mask as an explicit argument
     instead of a list input, and returns raw logits (B,1,H,W).
  3. MixPool logical_or now operates on explicit booleans
     ((fmask > 0.5) | (mask > 0.5)), identical to the official behavior.

Feedback protocol (handled by train.py / eval.py, mirroring the official
train.py / test.py):
  - Training: a per-image "previous epoch" mask (zeros at epoch 1) is fed as
    hard attention; after each epoch the binarized predictions replace it.
  - Testing: iterative rectification - `num_iter` passes where the model's
    own previous prediction is fed back (zeros in the first pass).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FANet"]


class SELayer(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """3x3 conv -> 3x3 conv residual block with 1x1 SE-gated shortcut."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.se(self.bn3(self.conv3(x)))
        return self.relu(x2 + x3)


class MixPool(nn.Module):
    """Merges feature maps with the feedback mask (hard attention)."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, m):
        m = m.to(torch.float32)   # accept bool/float masks; official used float32
        fmask = self.fmask(x) > 0.5
        kh, kw = m.shape[2] // x.shape[2], m.shape[3] // x.shape[3]
        m = F.max_pool2d(m, kernel_size=(max(kh, 1), max(kw, 1))) if (kh > 1 or kw > 1) else m
        x1 = x * ((fmask | (m > 0.5)).to(x.dtype))
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.r2(self.r1(inputs))
        p = self.p1(x, masks)
        return self.pool(p), x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c + in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.r2(self.r1(x))
        return self.p1(x, masks)


class FANet(nn.Module):
    """
    Feedback Attention Network (official widths). Input (B,3,H,W) plus a
    feedback mask (B,1,H,W); returns single-channel logits (B,1,H,W).
    H, W must be divisible by 16 (352 works).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.e1 = EncoderBlock(in_channels, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
        self.d4 = DecoderBlock(32, 16)

        self.output = nn.Conv2d(16 + 1, 1, kernel_size=1, padding=0)

    def forward(self, image, mask):
        p1, s1 = self.e1(image, mask)
        p2, s2 = self.e2(p1, mask)
        p3, s3 = self.e3(p2, mask)
        p4, s4 = self.e4(p3, mask)

        d1 = self.d1(p4, s4, mask)
        d2 = self.d2(d1, s3, mask)
        d3 = self.d3(d2, s2, mask)
        d4 = self.d4(d3, s1, mask)

        return self.output(torch.cat([d4, mask], dim=1))
