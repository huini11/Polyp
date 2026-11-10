"""
LightPolySegNet: a lightweight Boundary-Aware Transformer-CNN hybrid network
for real-time polyp segmentation.

Design summary
--------------
Encoder (hybrid, < ~3M params):
    CNN stem (3x3, stride 2) -> 4 stages. Each stage = MobileNetV2-style
    inverted residual blocks (local feature extraction) followed, for stages
    2-4, by a MobileViT-style block: a small ViT over non-overlapping patches
    that captures long-range context at a fraction of the cost of a full ViT.
    Stage 1 is pure CNN (high resolution, transformer would be wasteful).

Decoder (lightweight boundary-aware):
    Progressive top-down fusion (upsample + concat + depthwise-separable
    convs). A boundary enhancement branch computes an edge-attention map from
    fixed Sobel gradients of the shallow fused feature and re-weights it
    before the segmentation head. Two output heads:
        - main segmentation map (352x352, sigmoid)
        - boundary map          (352x352, sigmoid)
    plus coarse auxiliary segmentation maps for deep supervision.

The architecture is fully self-contained (no timm / pretrained backbones),
which keeps the reproducibility package dependency-free and the parameter
budget explicit. All widths are configurable in configs/default.yaml; if the
parameter count exceeds 10M, reduce stage_channels / tf_dims / tf_depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LightPolySegNet", "build_model_from_cfg"]


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """(Depthwise-separable optional) conv + BN + activation."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def dwconv(in_ch, out_ch, k=3, stride=1):
    """Depthwise-separable conv block: depthwise 3x3 + pointwise 1x1 (+BN+act)."""
    return nn.Sequential(
        ConvBNAct(in_ch, in_ch, k, stride, groups=in_ch),
        ConvBNAct(in_ch, out_ch, 1),
    )


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual with linear bottleneck."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [ConvBNAct(in_ch, mid, 1)]
        layers += [
            ConvBNAct(mid, mid, 3, stride, groups=mid),   # depthwise
            nn.Conv2d(mid, out_ch, 1, bias=False),        # linear projection
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Lightweight transformer (MobileViT-style block)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Pre-norm MHSA + MLP, GELU. Weight-compatible with timm's TransformerBlock
    when mlp_ratio=2.0 and dim matches (timm uses qkv_bias=True, num_heads=4)."""

    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT-style block: local CNN processing -> patch unfolding -> small
    transformer over patches -> folding back -> fusion conv.
    Input/output: (B, C, H, W) with H, W divisible by patch_size.
    """

    def __init__(self, in_ch, tf_dim, depth, patch_size=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch_size
        # Local branch
        self.local = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            ConvBNAct(in_ch, in_ch, 1),
        )
        # Global branch: project patch tokens (C*p*p) -> tf_dim
        self.proj_in = nn.Linear(in_ch * patch_size * patch_size, tf_dim)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(tf_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(tf_dim)
        self.proj_out = nn.Linear(tf_dim, in_ch * patch_size * patch_size)
        # Fusion of global re-encoded features with the input
        self.fusion = ConvBNAct(in_ch * 2, in_ch, 3)

    @staticmethod
    def _unfold(x, p):
        """(B,C,H,W) -> (B, N, C*p*p) non-overlapping patch tokens."""
        B, C, H, W = x.shape
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(B, (H // p) * (W // p), C * p * p)

    @staticmethod
    def _fold(x, p, H, W):
        """(B, N, C*p*p) -> (B, C, H, W)."""
        B, _, D = x.shape
        C = D // (p * p)
        x = x.view(B, H // p, W // p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        # Pad H, W up to a multiple of patch_size (e.g. 11x11 -> 12x12),
        # run the block, then crop back — keeps the module size-agnostic.
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]
        local = self.local(x)
        tokens = self._unfold(local, p)
        tokens = self.proj_in(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        tokens = self.proj_out(tokens)
        glob = self._fold(tokens, p, Hp, Wp)
        if pad_h or pad_w:
            glob = glob[:, :, :H, :W]
            x = x[:, :, :H, :W]
        return self.fusion(torch.cat([x, glob], dim=1))


class MobileViTBlockTimm(nn.Module):
    """
    timm-compatible MobileViT block ("pixel-quadrant" tokenization, conv
    projections). Module-for-module weight-compatible with timm's
    MobileVitBlock, enabling direct ImageNet weight transfer from
    timm `mobilevit_xxs` (see src/pretrained.py):
        local    <-> timm conv_kxk          (conv3x3 + BN + act)
        proj_in  <-> timm conv_1x1          (conv1x1, no bias)
        blocks   <-> timm transformer.*     (pre-norm MHSA + MLP r=2.0)
        norm     <-> timm norm              (LayerNorm)
        proj_out <-> timm conv_proj         (conv1x1 + BN + act)
        fusion   <-> timm conv_fusion       (conv3x3 on concat + BN + act)
    Tokenization mirrors timm exactly: patches are split into p*p quadrant
    sub-tokens; attention runs over patch positions per quadrant (B*P, N, C).
    """

    def __init__(self, in_ch, tf_dim, depth, patch_size=2, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch = patch_size
        self.local = ConvBNAct(in_ch, in_ch, 3)                 # timm conv_kxk
        self.proj_in = nn.Conv2d(in_ch, tf_dim, 1, bias=False)  # timm conv_1x1
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(tf_dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(tf_dim)                        # timm norm
        self.proj_out = ConvBNAct(tf_dim, in_ch, 1)             # timm conv_proj
        self.fusion = ConvBNAct(in_ch * 2, in_ch, 3)            # timm conv_fusion

    def _to_tokens(self, x):
        """(B,C,H,W) -> (B*p*p, N, C) with timm quadrant tokenization."""
        B, C, H, W = x.shape
        p = self.patch
        t = x.view(B, C, H // p, p, W // p, p)            # B,C,Hp,ph,Wp,pw
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()      # B,C,Hp,Wp,ph,pw
        t = t.view(B, C, (H // p) * (W // p), p * p)      # B,C,N,P
        t = t.permute(0, 3, 2, 1).contiguous()            # B,P,N,C
        return t.view(B * p * p, (H // p) * (W // p), C)

    def _from_tokens(self, t, B, Hp, Wp):
        """(B*p*p, N, C) -> (B,C,H,W)."""
        p = self.patch
        C = t.shape[-1]
        t = t.view(B, p * p, Hp * Wp, C)                  # B,P,N,C
        t = t.permute(0, 3, 2, 1).contiguous()            # B,C,N,P
        t = t.view(B, C, Hp, Wp, p, p)                    # B,C,Hp,Wp,ph,pw
        t = t.permute(0, 1, 2, 4, 3, 5).contiguous()      # B,C,Hp,ph,Wp,pw
        return t.view(B, C, Hp * p, Wp * p)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]
        local = self.local(x)
        tokens = self._to_tokens(self.proj_in(local))
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        glob = self._from_tokens(tokens, B, Hp // p, Wp // p)   # (B, tf_dim, H, W)
        glob = self.proj_out(glob)                              # (B, in_ch, H, W)
        if pad_h or pad_w:
            glob = glob[:, :, :H, :W]
            x = x[:, :, :H, :W]
        return self.fusion(torch.cat([x, glob], dim=1))


# ---------------------------------------------------------------------------
# Boundary enhancement module
# ---------------------------------------------------------------------------

class BoundaryEnhanceAttention(nn.Module):
    """
    Edge-attention re-weighting of the shallow decoder feature.
    Uses FIXED (non-learnable) Sobel filters to extract gradients, fuses them
    with the feature map, and predicts a spatial attention map in [0, 1].
    """

    def __init__(self, ch):
        super().__init__()
        self.reduce = dwconv(ch * 3, ch)          # concat(x, gx, gy) -> ch
        self.attn = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, 1, 1),
            nn.Sigmoid(),
        )
        # Fixed Sobel kernels (registered as buffers, never updated)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x.expand(x.size(1), 1, 3, 3),
                      padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y.expand(x.size(1), 1, 3, 3),
                      padding=1, groups=x.size(1))
        feat = self.reduce(torch.cat([x, gx, gy], dim=1))
        return x * self.attn(feat)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class LightPolySegNet(nn.Module):
    """
    Args (defaults match configs/default.yaml):
        stem_channels : stem width (stride-2 conv, 352 -> 176)
        stage_channels: encoder stage widths [s1, s2, s3, s4]
        mv2_blocks    : inverted-residual blocks per stage
        tf_dims       : transformer dims for stages 2-4
        tf_depths     : transformer depths for stages 2-4
        tf_patch_size : ViT patch size (feature cells)
        decoder_channels: fused widths at 1/16, 1/8, 1/4 (deep -> shallow)
        use_boundary_head : if True, also output a boundary map
    Returns dict:
        seg     : (B,1,352,352) main segmentation logits (pre-sigmoid? -> logits)
        boundary: (B,1,352,352) boundary logits (if use_boundary_head)
        aux     : list of coarse seg logits (1/4 and 1/8 scale) for deep supervision
    """

    def __init__(
        self,
        stem_channels: int = 16,
        stage_channels=(32, 56, 112, 176),
        mv2_blocks=(1, 2, 2, 2),
        mv2_expand: int = 4,
        tf_dims=(72, 128, 176),
        tf_depths=(2, 2, 2),
        tf_patch_size: int = 2,
        vit_style: str = "patch",       # "patch" (original) | "timm" (timm-aligned)
        tf_mlp_ratio: float = 4.0,      # transformer MLP ratio (timm xxs uses 2.0)
        decoder_channels=(96, 56, 32),
        use_boundary_head: bool = True,
        use_tf_blocks: bool = True,        # ablation: disable MobileViT blocks
        use_sobel_attention: bool = True,  # ablation: disable edge-attention branch
        encoder_mode: str = "mobilevit",   # "mobilevit" (ours) | "timm" (pretrained timm backbone control)
        timm_encoder: str = "mobilenetv2_050",  # timm model name when encoder_mode="timm"
        decoder_mode: str = "progressive", # "progressive" (ours) | "fpn" (minimal FPN baseline)
        in_channels: int = 3,
    ):
        super().__init__()
        c1, c2, c3, c4 = stage_channels
        self.use_boundary_head = use_boundary_head
        self.use_tf_blocks = use_tf_blocks
        self.use_sobel_attention = use_sobel_attention
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        vit_cls = MobileViTBlockTimm if vit_style == "timm" else MobileViTBlock

        # ---------------- Encoder ----------------
        if encoder_mode == "timm":
            # Capacity-matched control (exp9): pretrained timm backbone feeding
            # the SAME decoder. ImageNet weights are loaded by timm itself.
            import timm
            self.timm_enc = timm.create_model(
                timm_encoder, pretrained=True, features_only=True,
                out_indices=(1, 2, 3, 4))
            feat_ch = list(self.timm_enc.feature_info.channels())
            assert len(feat_ch) == 4
            self.timm_reduce = nn.ModuleList(
                [nn.Conv2d(f, o, 1) for f, o in zip(feat_ch, stage_channels)])
        else:
            self.stem = ConvBNAct(in_channels, stem_channels, 3, stride=2)   # 176x176

            self.stage1 = self._make_stage(stem_channels, c1, mv2_blocks[0],
                                           stride=2, expand_ratio=mv2_expand)  # 88
            self.stage2 = self._make_stage(c1, c2, mv2_blocks[1],
                                           stride=2, expand_ratio=mv2_expand)  # 44
            self.stage3 = self._make_stage(c2, c3, mv2_blocks[2],
                                           stride=2, expand_ratio=mv2_expand)  # 22
            self.stage4 = self._make_stage(c3, c4, mv2_blocks[3],
                                           stride=2, expand_ratio=mv2_expand)  # 11

            if use_tf_blocks:
                self.vit2 = vit_cls(c2, tf_dims[0], tf_depths[0], tf_patch_size,
                                    mlp_ratio=tf_mlp_ratio)
                self.vit3 = vit_cls(c3, tf_dims[1], tf_depths[1], tf_patch_size,
                                    mlp_ratio=tf_mlp_ratio)
                self.vit4 = vit_cls(c4, tf_dims[2], tf_depths[2], tf_patch_size,
                                    mlp_ratio=tf_mlp_ratio)

        # ---------------- Decoder ----------------
        if decoder_mode == "fpn":
            # Minimal FPN baseline (exp10): lateral 1x1 + top-down add + 3x3
            # smooth, no boundary head, no Sobel attention, no progressive fusion.
            self.fpn_dim = 64
            self.fpn_lat2 = nn.Conv2d(c2, self.fpn_dim, 1)
            self.fpn_lat3 = nn.Conv2d(c3, self.fpn_dim, 1)
            self.fpn_lat4 = nn.Conv2d(c4, self.fpn_dim, 1)
            self.fpn_smooth3 = dwconv(self.fpn_dim, self.fpn_dim)
            self.fpn_smooth2 = dwconv(self.fpn_dim, self.fpn_dim)
            self.fpn_lat1 = nn.Conv2d(c1, 32, 1)
            self.seg_head_fpn = nn.Sequential(
                dwconv(self.fpn_dim + 32, 32),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                dwconv(32, 16),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(16, 1, 1),
            )
            self.aux2 = nn.Conv2d(self.fpn_dim, 1, 1)   # 1/8
            self.aux3 = nn.Conv2d(self.fpn_dim, 1, 1)   # 1/16
            return

        d3, d2, d1 = decoder_channels
        # lateral 1x1 projections to keep fusion cheap
        self.lat3 = ConvBNAct(c3, d3, 1)
        self.lat2 = ConvBNAct(c2, d2, 1)
        self.lat1 = ConvBNAct(c1, d1, 1)
        # top-down fusion: upsample x2 + concat + depthwise-separable convs
        self.fuse3 = dwconv(c4 + d3, d3)
        self.fuse2 = dwconv(d3 + d2, d2)
        self.fuse1 = dwconv(d2 + d1, d1)

        if use_sobel_attention:
            self.boundary_attn = BoundaryEnhanceAttention(d1)

        # Main segmentation head: d1 (1/2 res) -> 1/1 via two upsample-conv steps
        self.seg_head = nn.Sequential(
            dwconv(d1, d1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            dwconv(d1, d1 // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d1 // 2, 1, 1),
        )
        # Boundary head operates on the same enhanced shallow feature
        if use_boundary_head:
            self.boundary_head = nn.Sequential(
                dwconv(d1, d1 // 2),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                dwconv(d1 // 2, d1 // 2),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(d1 // 2, 1, 1),
            )
        # Deep supervision: coarse seg maps from fuse2 (1/8) and fuse3 (1/16)
        self.aux2 = nn.Conv2d(d2, 1, 1)   # 1/8 resolution
        self.aux3 = nn.Conv2d(d3, 1, 1)   # 1/16 resolution

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride=2, expand_ratio=4):
        layers = [InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=expand_ratio)]
        for _ in range(n_blocks - 1):
            layers.append(InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, _, H, W = x.shape

        # Encoder
        if self.encoder_mode == "timm":
            feats = self.timm_enc(x)
            s1, s2, s3, s4 = [r(f) for r, f in zip(self.timm_reduce, feats)]
        else:
            s1 = self.stage1(self.stem(x))          # (B,c1,88,88)
            if self.use_tf_blocks:
                s2 = self.vit2(self.stage2(s1))     # (B,c2,44,44)
                s3 = self.vit3(self.stage3(s2))     # (B,c3,22,22)
                s4 = self.vit4(self.stage4(s3))     # (B,c4,11,11)
            else:
                s2 = self.stage2(s1)
                s3 = self.stage3(s2)
                s4 = self.stage4(s3)

        # Minimal FPN decoder (exp10 baseline)
        if self.decoder_mode == "fpn":
            p4 = self.fpn_lat4(s4)
            p3 = self.fpn_smooth3(self.fpn_lat3(s3) +
                                  F.interpolate(p4, scale_factor=2, mode="bilinear",
                                                align_corners=False))            # 1/16
            p2 = self.fpn_smooth2(self.fpn_lat2(s2) +
                                  F.interpolate(p3, scale_factor=2, mode="bilinear",
                                                align_corners=False))            # 1/8
            f1 = torch.cat([F.interpolate(p2, scale_factor=2, mode="bilinear",
                                          align_corners=False),
                            self.fpn_lat1(s1)], dim=1)                          # 1/4
            seg = self.seg_head_fpn(f1)
            if seg.shape[-2:] != (H, W):
                seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
            return {"seg": seg, "aux": [self.aux2(p2), self.aux3(p3)]}

        # Decoder (progressive top-down fusion)
        # s4 is 1/32 res; upsample x2 to meet s3 (1/16) for the first fusion
        f3 = self.fuse3(torch.cat([F.interpolate(s4, scale_factor=2,
                                                 mode="bilinear", align_corners=False),
                                   self.lat3(s3)], dim=1))                  # 1/16
        f2 = self.fuse2(torch.cat([F.interpolate(f3, scale_factor=2,
                                                 mode="bilinear", align_corners=False),
                                    self.lat2(s2)], dim=1))              # 1/8
        f1 = self.fuse1(torch.cat([F.interpolate(f2, scale_factor=2,
                                                 mode="bilinear", align_corners=False),
                                    self.lat1(s1)], dim=1))              # 1/4

        # Boundary enhancement attention on the shallow fused feature
        if self.use_sobel_attention:
            f1 = self.boundary_attn(f1)

        # Heads (upsample to full input resolution)
        seg = self.seg_head(f1)
        if seg.shape[-2:] != (H, W):
            seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)

        out = {"seg": seg}
        if self.use_boundary_head:
            bnd = self.boundary_head(f1)
            if bnd.shape[-2:] != (H, W):
                bnd = F.interpolate(bnd, size=(H, W), mode="bilinear", align_corners=False)
            out["boundary"] = bnd
        out["aux"] = [self.aux2(f2), self.aux3(f3)]  # coarse maps for deep supervision
        return out


def build_model_from_cfg(cfg: dict, load_pretrained=None):
    """
    Build a model from a config dict. Dispatches on cfg["model"]["arch"]:
      - "lightpolysegnet" (default): LightPolySegNet (ours)
      - "fanet": baseline FANet (Tomar et al. 2021), no pretrained init,
        forward signature model(image, feedback_mask)
    If model.pretrained is true (or load_pretrained=True is passed explicitly),
    attempt to initialize the encoder from timm `mobilevit_xxs` ImageNet
    weights (requires vit_style='timm' + timm-aligned widths to get high
    coverage); on any failure a warning is printed and random init is kept.
    eval.py passes load_pretrained=False so checkpoint loading never triggers
    a download.
    """
    m = cfg["model"]
    if m.get("arch", "lightpolysegnet") == "fanet":
        from src.fanet import FANet
        return FANet(in_channels=m.get("in_channels", 3))
    net = LightPolySegNet(
        stem_channels=m.get("stem_channels", 16),
        stage_channels=m.get("stage_channels", [32, 56, 112, 176]),
        mv2_blocks=m.get("mv2_blocks", [1, 2, 2, 2]),
        mv2_expand=m.get("mv2_expand", 4),
        tf_dims=m.get("tf_dims", [72, 128, 176]),
        tf_depths=m.get("tf_depths", [2, 2, 2]),
        tf_patch_size=m.get("tf_patch_size", 2),
        vit_style=m.get("vit_style", "patch"),
        tf_mlp_ratio=m.get("tf_mlp_ratio", 4.0),
        decoder_channels=m.get("decoder_channels", [96, 56, 32]),
        use_boundary_head=m.get("use_boundary_head", True),
        use_tf_blocks=m.get("use_tf_blocks", True),
        use_sobel_attention=m.get("use_sobel_attention", True),
        encoder_mode=m.get("encoder_mode", "mobilevit"),
        timm_encoder=m.get("timm_encoder", "mobilenetv2_050"),
        decoder_mode=m.get("decoder_mode", "progressive"),
    )
    want_pretrained = m.get("pretrained", False) if load_pretrained is None else load_pretrained
    if want_pretrained and m.get("encoder_mode", "mobilevit") == "timm":
        print("[pretrained] encoder_mode='timm': ImageNet weights were loaded "
              "inside timm at build time; skipping the MobileViT checkpoint "
              "loader (decoder remains randomly initialized, same protocol "
              "as the main model).")
    elif want_pretrained:
        from src.pretrained import load_mobilevit_xxs_pretrained
        load_mobilevit_xxs_pretrained(net, verbose=True)
    return net
