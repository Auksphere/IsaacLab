"""Frozen visuo-force encoder for Stage F RL finetune.

Loaded from pretrained checkpoint produced by Stage E pretrain_dynamics.py.
No bottleneck — returns [v_t, f_t] concat (vis_dim + 64).  The environment
computes Δv, Δf externally using ring buffers; VICReg + dynamics gradients
flowed through exactly these features during pretrain.
"""
import torch
import torch.nn as nn


# ── Image preprocessing (must match pretrain_dynamics.py) ──
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def preprocess_rgb(rgb_uint8):
    """uint8 (..., H, W, 3) → float32 (..., 3, IMG_SIZE, IMG_SIZE) ImageNet-normalised.

    Bilinear-resizes to IMG_SIZE if the camera resolution differs (e.g. 112→224).
    """
    import torch.nn.functional as F
    x = rgb_uint8.float() / 255.0
    if x.dim() == 4:  # (N, H, W, 3) → (N, 3, H, W)
        x = x.permute(0, 3, 1, 2)
    elif x.dim() == 3:  # (H, W, 3) → (3, H, W)
        x = x.permute(2, 0, 1)
    if x.shape[-2] != IMG_SIZE or x.shape[-1] != IMG_SIZE:
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    for c in range(3):
        x[:, c] = (x[:, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return x


# ── Backbone factory ─────────────────────────────────────────────────────────

def _build_backbone(name: str):
    """Return (backbone_module, feature_dim, encode_fn)."""
    import torchvision

    name = name.lower()
    if name in ("resnet18", "resnet50"):
        net = getattr(torchvision.models, name)(weights="IMAGENET1K_V1")
        backbone = nn.Sequential(*list(net.children())[:-2])
        dim = 512 if name == "resnet18" else 2048
        gap = nn.AdaptiveAvgPool2d(1)
        for p in backbone.parameters():
            p.requires_grad = False

        def encode_resnet(x):
            with torch.no_grad():
                feats = backbone(x)
            return gap(feats).flatten(1)

        return backbone, dim, encode_resnet

    if name.startswith("dinov2_"):
        import importlib, os, sys
        variant = name.replace("dinov2_", "")
        dim = {"vits14": 384, "vitb14": 768}[variant]

        hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        ckpt_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        ckpt_path = os.path.join(ckpt_dir, f"dinov2_{variant}_pretrain.pth")

        if not os.path.isdir(hub_dir):
            raise RuntimeError(
                f"DINOv2 cache not found at {hub_dir}.  "
                "Run once with network access or copy the cache manually."
            )

        # Load hubconf from local cache (no network)
        old_path = sys.path.copy()
        sys.path.insert(0, hub_dir)
        try:
            hub_module = importlib.import_module("hubconf")
            model_fn = getattr(hub_module, f"dinov2_{variant}")
            model = model_fn(pretrained=False)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        finally:
            sys.path = old_path

        for p in model.parameters():
            p.requires_grad = False

        def encode_dinov2(x):
            with torch.no_grad():
                return model(x)

        return model, dim, encode_dinov2

    raise ValueError(f"Unknown backbone: {name}")


# ── Encoder ──────────────────────────────────────────────────────────────────

class VisualForceEncoder(nn.Module):
    """Frozen visuo-force encoder for RL inference.

    No bottleneck — returns [v_t, f_t] concat (vis_dim + 64).
    Δv, Δf are computed externally by the environment using ring buffers.
    The same features received VICReg + dynamics gradients during pretrain.
    """

    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        self._backbone, self._vis_dim, self._encode_frame_fn = \
            _build_backbone(backbone_name)

        D = self._vis_dim

        self.proj_mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        self.force_mlp = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, 64), nn.GELU(), nn.Linear(64, 64))

    @property
    def feature_dim(self) -> int:
        """RL policy input dimension (= vis_dim + 64)."""
        return self._vis_dim + 64

    def encode_visual(self, img_left, img_right):
        """img_left/right: (B, 3, H, W) single-frame, ImageNet-normalised."""
        fl = self.proj_mlp(self._encode_frame_fn(img_left))
        fr = self.proj_mlp(self._encode_frame_fn(img_right))
        return self.fusion_mlp(torch.cat([fl, fr], dim=-1))   # (B, vis_dim)

    def encode_force(self, ft):
        """ft: (B, 18) — 3-frame force history."""
        return self.force_mlp(ft)                              # (B, 64)

    def forward_features(self, img_left, img_right, ft_3frame):
        """Return [v_t, f_t] concat for RL (env computes Δ externally).

        Returns:
            (B, vis_dim + 64) features — same ones that received
            VICReg + dynamics gradients during pretrain.
        """
        v = self.encode_visual(img_left, img_right)
        f = self.encode_force(ft_3frame)
        return torch.cat([v, f], dim=-1)                       # (B, vis_dim + 64)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, backbone_name: str = "resnet18",
                        device: str = "cuda:0"):
        """Load frozen encoder from Stage E checkpoint.

        Uses strict=False because pretrain checkpoint contains dynamics heads
        (not needed for inference).
        """
        model = cls(backbone_name=backbone_name)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
