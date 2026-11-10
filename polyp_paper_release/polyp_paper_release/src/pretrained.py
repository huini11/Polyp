"""
ImageNet pretrained initialization from timm `mobilevit_xxs.cvnets_in1k`.

Strategy (exp3): the encoder config in configs/exp3_pretrain.yaml is aligned
module-for-module with timm's mobilevit_xxs backbone:

  timm mobilevit_xxs                     our LightPolySegNet (vit_style='timm')
  -----------------------------------------------------------------------
  stem (16, 3x3, s2)                     stem
  stages.0.0  IR c16 s1 (1/2 res)        [no counterpart - skipped]
  stages.1.{0,1,2}  IR c24 s2 d=3 br=2   stage1.{0,1,2}   (mv2_expand=2)
  stages.2.0  IR 24->48 br=2             stage2.0
  stages.2.1  MobileVit c48 tf64 d2      vit2
  stages.3.0  IR 48->64 br=2             stage3.0
  stages.3.1  MobileVit c64 tf80 d4      vit3
  stages.4.0  IR 64->80 br=2             stage4.0
  stages.4.1  MobileVit c80 tf96 d3      vit4
  final_conv / head (classifier)         [no counterpart - skipped]

Weight loading is shape-checked key by key; unmatched keys are skipped and
coverage statistics are printed. If the timm download fails (no network /
corrupt cache), a local file `pretrained/mobilevit_xxs.safetensors` (or .pth)
inside the project root is used as fallback; failing both, the function
prints a warning and the model keeps its random initialization (never raises).
"""

import os

import torch

TIMM_VARIANT = "mobilevit_xxs"

# timm stage index -> our encoder module prefix
_IR_STAGE_MAP = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4"}
_VIT_STAGE_MAP = {2: "vit2", 3: "vit3", 4: "vit4"}
# timm IR sub-module -> our InvertedResidual(self.block=Sequential) indices
_IR_SUFFIX_MAP = {
    "conv1_1x1.conv.weight": "block.0.conv.weight",
    "conv1_1x1.bn.weight": "block.0.bn.weight",
    "conv1_1x1.bn.bias": "block.0.bn.bias",
    "conv1_1x1.bn.running_mean": "block.0.bn.running_mean",
    "conv1_1x1.bn.running_var": "block.0.bn.running_var",
    "conv2_kxk.conv.weight": "block.1.conv.weight",
    "conv2_kxk.bn.weight": "block.1.bn.weight",
    "conv2_kxk.bn.bias": "block.1.bn.bias",
    "conv2_kxk.bn.running_mean": "block.1.bn.running_mean",
    "conv2_kxk.bn.running_var": "block.1.bn.running_var",
    "conv3_1x1.conv.weight": "block.2.weight",
    "conv3_1x1.bn.weight": "block.3.weight",
    "conv3_1x1.bn.bias": "block.3.bias",
    "conv3_1x1.bn.running_mean": "block.3.running_mean",
    "conv3_1x1.bn.running_var": "block.3.running_var",
}
# timm MobileVitBlock sub-module -> our MobileViTBlockTimm sub-module
_VIT_SUFFIX_MAP = {
    "conv_kxk.conv.weight": "local.conv.weight",
    "conv_kxk.bn.weight": "local.bn.weight",
    "conv_kxk.bn.bias": "local.bn.bias",
    "conv_kxk.bn.running_mean": "local.bn.running_mean",
    "conv_kxk.bn.running_var": "local.bn.running_var",
    "conv_1x1.weight": "proj_in.weight",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
    "conv_proj.conv.weight": "proj_out.conv.weight",
    "conv_proj.bn.weight": "proj_out.bn.weight",
    "conv_proj.bn.bias": "proj_out.bn.bias",
    "conv_proj.bn.running_mean": "proj_out.bn.running_mean",
    "conv_proj.bn.running_var": "proj_out.bn.running_var",
    "conv_fusion.conv.weight": "fusion.conv.weight",
    "conv_fusion.bn.weight": "fusion.bn.weight",
    "conv_fusion.bn.bias": "fusion.bn.bias",
    "conv_fusion.bn.running_mean": "fusion.bn.running_mean",
    "conv_fusion.bn.running_var": "fusion.bn.running_var",
}


def _map_timm_key(key: str):
    """Map a timm state-dict key to our key; return None if no counterpart."""
    parts = key.split(".")
    if key.startswith("stem."):
        m = {"conv.weight": "conv.weight",
             "bn.weight": "bn.weight", "bn.bias": "bn.bias",
             "bn.running_mean": "bn.running_mean",
             "bn.running_var": "bn.running_var"}.get(key[len("stem."):])
        return f"stem.{m}" if m else None
    if key.startswith("stages."):
        rest = key[len("stages."):]
        s_str, b_str, suffix = rest.split(".", 2)
        s, b = int(s_str), int(b_str)  # timm keys are numeric: stages.2.0...
        if s == 1 and suffix in _IR_SUFFIX_MAP:
            # timm stage1 is a pure IR stage (d=3): all blocks map to stage1
            return f"stage1.{b}.{_IR_SUFFIX_MAP[suffix]}"
        if s in (2, 3, 4) and b == 0 and suffix in _IR_SUFFIX_MAP:
            # each ViT stage starts with a single IR block
            return f"{_IR_STAGE_MAP[s]}.0.{_IR_SUFFIX_MAP[suffix]}"
        if s in _VIT_STAGE_MAP and b == 1:
            tgt = _VIT_STAGE_MAP[s]
            # transformer blocks
            if suffix.startswith("transformer."):
                idx, sub = suffix[len("transformer."):].split(".", 1)
                inner = {"norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
                         "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
                         "attn.qkv.weight": "attn.in_proj_weight",
                         "attn.qkv.bias": "attn.in_proj_bias",
                         "attn.proj.weight": "attn.out_proj.weight",
                         "attn.proj.bias": "attn.out_proj.bias",
                         "mlp.fc1.weight": "mlp.0.weight", "mlp.fc1.bias": "mlp.0.bias",
                         "mlp.fc2.weight": "mlp.3.weight", "mlp.fc2.bias": "mlp.3.bias",
                         }.get(sub)
                return f"{tgt}.blocks.{idx}.{inner}" if inner else None
            return f"{tgt}.{_VIT_SUFFIX_MAP[suffix]}" if suffix in _VIT_SUFFIX_MAP else None
    return None  # stages.0.0, final_conv, head -> skipped


def _get_timm_state_dict():
    """Load timm mobilevit_xxs weights: HF hub first, then local fallback."""
    try:
        import timm
        tm = timm.create_model(TIMM_VARIANT, pretrained=True)
        return tm.state_dict(), f"timm hub ({TIMM_VARIANT})"
    except Exception as e:
        print(f"[pretrained] timm hub download failed ({type(e).__name__}: {e}); "
              f"trying local fallback file...")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ("mobilevit_xxs.safetensors", "mobilevit_xxs.pth"):
        p = os.path.join(root, "pretrained", fname)
        if os.path.isfile(p):
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(p), p
            return torch.load(p, map_location="cpu", weights_only=True), p
    return None, None


def load_mobilevit_xxs_pretrained(model, verbose: bool = True) -> dict:
    """
    Load timm mobilevit_xxs ImageNet weights into `model` (shape-checked).
    Never raises: on any failure the model keeps random initialization.
    Returns {"loaded": n_tensors, "loaded_params": n, "source": str}.
    """
    stats = {"loaded": 0, "loaded_params": 0, "source": None}
    try:
        sd, source = _get_timm_state_dict()
        if sd is None:
            print("[pretrained][WARN] no timm weights available -> keeping random init. "
                  "Fix: `pip install timm` + network, or place the file at "
                  "polyp_paper/pretrained/mobilevit_xxs.safetensors")
            return stats
        own = model.state_dict()
        new_sd = {}
        for k, v in sd.items():
            tgt = _map_timm_key(k)
            if tgt is None or tgt not in own:
                continue
            if own[tgt].shape != v.shape:
                if verbose:
                    print(f"[pretrained] shape mismatch, skipping {k}: "
                          f"{tuple(v.shape)} vs {tuple(own[tgt].shape)}")
                continue
            new_sd[tgt] = v
            stats["loaded"] += 1
            stats["loaded_params"] += v.numel()
        model.load_state_dict(new_sd, strict=False)
        stats["source"] = source
        total = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"[pretrained] source: {source}")
            print(f"[pretrained] loaded {stats['loaded']} tensors "
                  f"({stats['loaded_params']/1e6:.2f}M params) -> "
                  f"{100.0 * stats['loaded_params'] / total:.1f}% of model total "
                  f"({total/1e6:.2f}M)")
    except Exception as e:  # never crash training because of pretraining
        print(f"[pretrained][WARN] pretrained init failed "
              f"({type(e).__name__}: {e}) -> keeping random init.")
    return stats
