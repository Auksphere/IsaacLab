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

    Returns bottleneck(256D) from [v_t, Δf, prev_a, proprio].
    Pretrained with dynamics + task progress + future self-prediction + VICReg.
    """

    def __init__(self, backbone_name: str = "resnet18", mono: bool = False):
        super().__init__()
        self._mono = mono
        self._backbone, self._vis_dim, self._encode_frame_fn = \
            _build_backbone(backbone_name)

        D = self._vis_dim

        self.proj_mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        if not mono:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        else:
            self.fusion_mlp = None
        self.force_mlp = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, 128), nn.GELU(), nn.Linear(128, 64))
        # Bottleneck: [v_t(vis_dim), Δv(vis_dim), Δf(64)] → 256D (motion-aware)
        self.bottleneck = nn.Sequential(
            nn.Linear(D * 2 + 64, 256), nn.GELU(), nn.Linear(256, 256))
        # Task head: [bottleneck(256), proprio(20)] → held_pos_rel_fixed (3D, bypass)
        self.head_task = nn.Sequential(
            nn.Linear(256 + 20, 128), nn.GELU(), nn.Linear(128, 3))

        # Task output normalization (loaded from pretrain checkpoint)
        self.register_buffer("task_mean", torch.zeros(3))
        self.register_buffer("task_std", torch.ones(3))

        # Cache for visual delta (mimics pretrain Δv = v_t - v_{t-k})
        self.register_buffer("_prev_v_t", None)  # (num_envs, vis_dim), set on first call

    @property
    def feature_dim(self) -> int:
        return 256

    def encode_visual(self, img_left, img_right=None):
        """img_left: (B, 3, H, W) single-frame. img_right optional (stereo only)."""
        x = img_left[:, -1, :, :, :] if img_left.dim() == 5 else img_left
        fl = self.proj_mlp(self._encode_frame_fn(x))
        if self.fusion_mlp is not None and img_right is not None:
            xr = img_right[:, -1, :, :, :] if img_right.dim() == 5 else img_right
            fr = self.proj_mlp(self._encode_frame_fn(xr))
            return self.fusion_mlp(torch.cat([fl, fr], dim=-1))  # (B, vis_dim)
        return fl  # mono: single-frame direct output

    def encode_force(self, ft):
        """ft: (B, 18) — 3-frame force history."""
        return self.force_mlp(ft)                              # (B, 64)

    def forward_features(self, img_left, img_right, ft_3frame, ft_3frame_k,
                          prev_action, proprio):
        """Return (bottleneck(256D), task_pred(3D)) for RL policy."""
        v_t = self.encode_visual(img_left, img_right)
        f_t = self.encode_force(ft_3frame)
        f_tk = self.encode_force(ft_3frame_k)
        Δf = f_t - f_tk

        # Visual delta (cached previous v_t, like pretrain Δv = v_t - v_{t-k})
        if self._prev_v_t is None:
            self._prev_v_t = v_t.detach()  # first call: init cache
        Δv = v_t - self._prev_v_t
        self._prev_v_t = v_t.detach()

        z = self.bottleneck(torch.cat([v_t, Δv, Δf], dim=-1))
        task_pred = self.head_task(torch.cat([z, proprio], dim=-1))  # (B, 3) normalized
        task_pred = task_pred * self.task_std + self.task_mean        # de-normalize to meters
        return z, task_pred

    def predict_task(self, img_l, ft_3frame, img_r=None):
        """Convenience wrapper for eval — matches eval_encoder_features call."""
        B = img_l.shape[0] if img_l.dim() >= 3 else 1
        dev = img_l.device
        ft_3frame_k = torch.zeros_like(ft_3frame)
        prev_action = torch.zeros(B, 6, device=dev)
        proprio = torch.zeros(B, 20, device=dev)
        _, task_pred = self.forward_features(img_l, img_r, ft_3frame, ft_3frame_k,
                                              prev_action, proprio)
        return task_pred

    def reset_cache(self, env_ids=None):
        """Reset cached visual features (call on env reset)."""
        if self._prev_v_t is not None:
            if env_ids is None:
                self._prev_v_t.zero_()
            else:
                self._prev_v_t[env_ids] = 0.0

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, backbone_name: str = "resnet18",
                        device: str = "cuda:0", mono: bool = False):
        model = cls(backbone_name=backbone_name, mono=mono)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)  # fusion_mlp keys skipped if mono
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
