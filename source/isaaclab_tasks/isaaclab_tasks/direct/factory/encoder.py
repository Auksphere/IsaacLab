"""Frozen visuo-force encoder for RL inference. V3 dual-pathway architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 224
FT_FRAMES = 10
FT_DIM = 6


def preprocess_rgb(rgb_uint8):
    x = rgb_uint8.float() / 255.0
    if x.dim() == 4: x = x.permute(0, 3, 1, 2)
    elif x.dim() == 3: x = x.permute(2, 0, 1)
    if x.shape[-2] != IMG_SIZE or x.shape[-1] != IMG_SIZE:
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    for c in range(3):
        x[:, c] = (x[:, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return x


def _build_backbone(name: str):
    import torchvision, importlib, os, sys
    name = name.lower()
    if name in ("resnet18", "resnet50"):
        net = getattr(torchvision.models, name)(weights="IMAGENET1K_V1")
        backbone = nn.Sequential(*list(net.children())[:-2])
        dim = 512 if name == "resnet18" else 2048
        gap = nn.AdaptiveAvgPool2d(1)
        for p in backbone.parameters(): p.requires_grad = False
        def encode_fn(x):
            with torch.no_grad(): feats = backbone(x)
            return gap(feats).flatten(1)
        return backbone, dim, encode_fn
    if name.startswith("dinov2_"):
        variant = name.replace("dinov2_", "").replace("_patches", "").replace("_attn", "")
        dim = {"vits14": 384, "vitb14": 768}[variant]
        hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        ckpt_path = os.path.expanduser(f"~/.cache/torch/hub/checkpoints/dinov2_{variant}_pretrain.pth")
        if not os.path.isdir(hub_dir):
            raise RuntimeError(f"DINOv2 cache not found at {hub_dir}.")
        old_path = sys.path.copy(); sys.path.insert(0, hub_dir)
        try:
            hub_module = importlib.import_module("hubconf")
            model_fn = getattr(hub_module, f"dinov2_{variant}")
            dino = model_fn(pretrained=False)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            dino.load_state_dict(state_dict, strict=True)
        finally: sys.path = old_path

        use_patches = name.endswith("_patches")
        use_attn = name.endswith("_attn")

        if use_attn:
            class PatchAttnEncoder(nn.Module):
                def __init__(self, dino_model, dim):
                    super().__init__()
                    self.dino = dino_model
                    self.attn = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
                    for p in self.dino.parameters():
                        p.requires_grad = False
                def forward(self, x):
                    with torch.no_grad():
                        patches = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)[0]
                    w = F.softmax(self.attn(patches).squeeze(-1), dim=-1)
                    return (patches * w.unsqueeze(-1)).sum(dim=1)
            model = PatchAttnEncoder(dino, dim)
            def encode_fn(x):
                return model(x)
        elif use_patches:
            for p in dino.parameters():
                p.requires_grad = False
            model = dino
            def encode_fn(x):
                with torch.no_grad():
                    out = model.get_intermediate_layers(x, n=1, return_class_token=False)
                    return out[0].mean(dim=1)
        else:
            for p in dino.parameters():
                p.requires_grad = False
            model = dino
            def encode_fn(x):
                with torch.no_grad():
                    return model(x)
        return model, dim, encode_fn
    raise ValueError(f"Unknown backbone: {name}")


class VisualForceEncoder(nn.Module):
    """V3: Dual-pathway — visual-geometry + force-contact, no bottleneck."""

    def __init__(self, backbone_name: str = "resnet18", mono: bool = False):
        super().__init__()
        self._mono = mono
        self._backbone, self._vis_dim, self._encode_frame_fn = _build_backbone(backbone_name)
        D = self._vis_dim

        self.register_buffer("task_mean", torch.zeros(3))
        self.register_buffer("task_std", torch.ones(3))

        self.proj_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        if not mono:
            self.fusion_mlp = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        else:
            self.fusion_mlp = None

        ft_in = FT_FRAMES * FT_DIM
        self.force_mlp = nn.Sequential(
            nn.LayerNorm(ft_in), nn.Linear(ft_in, 128), nn.GELU(), nn.Linear(128, 64))

        self.diff_extract_v = nn.Sequential(
            nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))
        self.diff_extract_f = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 32))

        # Pathway 1: Visual-Geometry
        self.head_vis_fc1 = nn.Linear(D + 32 + 20, 256)
        self.head_vis_fc2 = nn.Linear(256, 128)
        self.head_task = nn.Sequential(
            nn.Linear(128 + 6, 128), nn.GELU(), nn.Linear(128, 2))  # XY direction only

        # Pathway 2: Force-Contact
        self.head_force = nn.Sequential(
            nn.Linear(64 + 32 + 20, 64), nn.GELU(), nn.Linear(64, 128))
        # engagement: z_vis(128)+z_force(128) → eng_pred (visual + force)
        self.head_engagement = nn.Sequential(
            nn.Linear(128 + 128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

        self.register_buffer("_prev_v_t", None)

    @property
    def feature_dim(self) -> int:
        return 128 + 128  # z_vis + z_force

    def encode_visual(self, imgs_left, imgs_right=None):
        xl = imgs_left[:, -1, :, :, :] if imgs_left.dim() == 5 else imgs_left
        fl = self.proj_mlp(self._encode_frame_fn(xl))
        if self.fusion_mlp is not None and imgs_right is not None:
            xr = imgs_right[:, -1, :, :, :] if imgs_right.dim() == 5 else imgs_right
            fr = self.proj_mlp(self._encode_frame_fn(xr))
            return self.fusion_mlp(torch.cat([fl, fr], dim=-1))
        return fl

    def encode_force(self, ft): return self.force_mlp(ft)

    def forward_features(self, img_left, img_right, ft_3frame, ft_3frame_k,
                          prev_action, proprio):
        v_t = self.encode_visual(img_left, img_right)
        f_t = self.encode_force(ft_3frame)
        f_tk = self.encode_force(ft_3frame_k)

        if self._prev_v_t is None or self._prev_v_t.shape[0] != v_t.shape[0]:
            self._prev_v_t = v_t.detach()
        Δv, Δf = v_t - self._prev_v_t, f_t - f_tk
        self._prev_v_t = v_t.detach()

        α_v = self.diff_extract_v(Δv)
        α_f = self.diff_extract_f(Δf)

        # Pathway 1
        h_vis = F.gelu(self.head_vis_fc1(torch.cat([v_t, α_v, proprio], dim=-1)))
        z_vis = F.gelu(self.head_vis_fc2(h_vis))

        # Pathway 2
        h_force = F.gelu(self.head_force(torch.cat([f_t, α_f, proprio], dim=-1)))
        z_force = h_force

        # Task: direction prediction + engagement (visual+force)
        task_norm = self.head_task(torch.cat([z_vis, prev_action], dim=-1))
        eng_pred = self.head_engagement(torch.cat([z_vis, z_force], dim=-1))

        return z_vis, z_force, task_norm, eng_pred

    def predict_task(self, img_l, ft_3frame, img_r=None,
                      prev_action=None, proprio=None, ft_3frame_k=None):
        B = img_l.shape[0] if img_l.dim() >= 3 else 1
        dev = img_l.device
        if ft_3frame_k is None:
            ft_3frame_k = torch.zeros(B, FT_FRAMES * FT_DIM, device=dev)
        if prev_action is None:
            prev_action = torch.zeros(B, 6, device=dev)
        if proprio is None:
            proprio = torch.zeros(B, 20, device=dev)
        z_vis, z_force, task_norm, eng_pred = self.forward_features(
            img_l, img_r, ft_3frame, ft_3frame_k, prev_action, proprio)
        dir_pred = F.normalize(task_norm, dim=-1, eps=1e-8)
        return dir_pred, eng_pred

    def reset_cache(self, env_ids=None):
        if self._prev_v_t is not None:
            if env_ids is None: self._prev_v_t.zero_()
            else: self._prev_v_t[env_ids] = 0.0

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, backbone_name: str = "resnet18",
                        device: str = "cuda:0", mono: bool = False):
        model = cls(backbone_name=backbone_name, mono=mono)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        for p in model.parameters(): p.requires_grad = False
        return model
