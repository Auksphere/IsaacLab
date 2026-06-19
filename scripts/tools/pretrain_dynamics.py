#!/usr/bin/env python3
"""
Stage E: Pretrain symmetric visuo-force dynamics encoder with VICReg regularisation.

Architecture (per idea.md §3 + AFRO differencing):
  Visual:  img_L(t) → ResNet(frozen) → proj_mlp ──┐
           img_R(t) → ResNet(frozen) → proj_mlp ──┤
                        └── FusionMLP → v_single (vis_dim)
  Force:   F(t-2,t-1,t) → cat[18D] → LayerNorm → ForceMLP(18→64→64) → f_t (64D)
  Deltas:  Δv = v_single_t − v_single_{t-k}     Δf = f_stack_t − f_stack_{t-k}
  Gating:  Δp_base = DynamicsHead_vis(Δv), Δp_resid = DynamicsHead_force(Δf)
           α = GateMLP(f_stack_t),  Δp_hat = (1-α)·Δp_base + α·Δp_resid
  VICReg: on [Δv, Δf] concat — invariant to visual+force domain shift

Loss:
  L_dynamics = Huber_fwd + Huber_bwd + 0.3·Huber_cons  (robust to contact outliers)
  L_vicreg   = w_vicreg × (L_inv + 0.33·L_var + 0.33·L_cov)

Runs on the HOST machine (pure PyTorch — no Isaac Sim dependency).
Reads data/ produced by collect_unified_dataset.py.

Usage:
  python pretrain_dynamics.py --data-dir ../data --epochs 100
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
FT_FRAMES = 3       # force history frames per sample
FT_DIM = 6          # force/torque per frame
IMG_SIZE = 224      # camera resolution
K = 3               # frame-skip for Δp computation
PROPRIO_DIM = 20    # Δp dimension: qpos(7)+fingertip_pos(3)+fingertip_quat(4)+ee_linvel(3)+ee_angvel(3)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =============================================================================
# Dataset
# =============================================================================

class StageEDataset(Dataset):
    """Load Isaac Lab labels.jsonl, build 3-frame sliding windows.

    Each sample:
      - img_left_t  : (3, 3, H, W) — 3-frame left camera ending at t
      - img_right_t : (3, 3, H, W) — 3-frame right camera ending at t
      - img_left_tk : (3, 3, H, W) — 3-frame left camera ending at t-K
      - img_right_tk: (3, 3, H, W)
      - ft_t        : (18,) — 3-frame force_torque ending at t
      - ft_tk       : (18,) — 3-frame force_torque ending at t-K
      - p_t         : (20,) — proprio at t
      - p_tk        : (20,) — proprio at t-K
      - ep_id, t    : metadata
    """

    def _load(self) -> None:
        episodes: Dict[int, List[Dict]] = defaultdict(list)
        with self.labels_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episodes[rec["episode_id"]].append(rec)

        min_len = K + 1  # need at least t and t-K
        for ep_id, frames in episodes.items():
            if len(frames) < min_len:
                continue
            # t must have 2 prior frames for 3-frame window
            for t in range(FT_FRAMES - 1, len(frames)):
                t_k = t - K
                if t_k < FT_FRAMES - 1:
                    continue
                self._samples.append({"ep_id": ep_id, "t": t, "t_k": t_k})

    def __len__(self) -> int:
        return len(self._samples)

    def _load_rgb(self, rel_path: str) -> torch.Tensor:
        """Load one camera image, resize to 224×224, normalise with ImageNet stats."""
        path = self.images_dir / Path(rel_path).name  # basename only (images/ is flat)
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # (H, W, 3)
        t = t.permute(2, 0, 1)  # (3, H, W)
        for c in range(3):
            t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return t

    def _load_3frame_rgb(self, episode: List[Dict], idx: int, cam_key: str) -> torch.Tensor:
        """Load 3-frame window ending at idx for one camera. Returns (3, 3, H, W)."""
        frames = []
        for offset in range(-2, 1):
            i = max(0, min(idx + offset, len(episode) - 1))
            frames.append(self._load_rgb(episode[i][cam_key]))
        return torch.stack(frames, dim=0)  # (3, 3, H, W)

    def _load_3frame_ft(self, episode: List[Dict], idx: int) -> torch.Tensor:
        """Load 3-frame force ending at idx → (18,).

        Uses 'force_torque' which is sum(left+right fingertip forces) in raw
        Newtons. ForceMLP LayerNorm learns optimal scaling — no hand-coded
        normalisation.
        """
        vals = []
        for offset in range(-2, 1):
            i = max(0, min(idx + offset, len(episode) - 1))
            ft = episode[i].get("force_torque", [0.0] * FT_DIM)
            vals.extend([float(v) for v in ft[:FT_DIM]])
        return torch.tensor(vals, dtype=torch.float32)

    def _load_proprio(self, episode: List[Dict], idx: int) -> torch.Tensor:
        """Build 20D proprio from Isaac Lab labels."""
        i = max(0, min(idx, len(episode) - 1))
        rec = episode[i]
        vals = []
        vals.extend([float(v) for v in rec.get("joint_pos", [0]*7)[:7]])
        vals.extend([float(v) for v in rec.get("fingertip_pos", [0]*3)[:3]])
        vals.extend([float(v) for v in rec.get("fingertip_quat", [0]*4)[:4]])
        vals.extend([float(v) for v in rec.get("ee_linvel", [0]*3)[:3]])
        vals.extend([float(v) for v in rec.get("ee_angvel", [0]*3)[:3]])
        return torch.tensor(vals, dtype=torch.float32)

    @staticmethod
    def _load_action(episode: List[Dict], idx: int) -> torch.Tensor:
        """Load 6D action at frame idx."""
        i = max(0, min(idx, len(episode) - 1))
        rec = episode[i]
        a = rec.get("action", [0.0] * 6)
        return torch.tensor([float(v) for v in a[:6]], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self._samples[idx]
        ep_id = info["ep_id"]
        ep = self._episodes(ep_id)
        t, t_k = info["t"], info["t_k"]

        # τ_t: 0.95^(steps to first success), then 1.0 after success
        first_succ = next((i for i, r in enumerate(ep) if r.get("keypoint_dist", 1.0) < 0.002), None)
        if first_succ is not None:
            steps_to = max(0, first_succ - t)
            tau = 1.0 if t >= first_succ else 0.80 ** steps_to
        else:
            tau = 0.0

        # Task target: held_pos_rel_fixed (peg→hole 3D offset) from JSONL → mm
        held_rel = ep[t].get("held_pos_rel_fixed", [0.0, 0.0, 0.0])
        held_rel = torch.tensor([float(v) * 1000 for v in held_rel[:3]], dtype=torch.float32)

        return {
            "img_left_t":   self._load_3frame_rgb(ep, t,   "camera_left"),
            "img_right_t":  self._load_3frame_rgb(ep, t,   "camera_right"),
            "img_left_tk":  self._load_3frame_rgb(ep, t_k, "camera_left"),
            "img_right_tk": self._load_3frame_rgb(ep, t_k, "camera_right"),
            "ft_t":         self._load_3frame_ft(ep, t),
            "ft_tk":        self._load_3frame_ft(ep, t_k),
            "p_t":          self._load_proprio(ep, t),
            "p_tk":         self._load_proprio(ep, t_k),
            "a":            self._load_action(ep, max(0, t - 1)),  # prev action (matches RL)
            "tau":          torch.tensor(tau, dtype=torch.float32),
            "held_pos_rel_fixed": held_rel,  # task target (3D)
            "ep_id":        torch.tensor(info["ep_id"]),
            "t":            torch.tensor(t),
        }

    # Episode cache (per-instance — avoids train/val data leak)
    _cache: Dict[int, List[Dict]]

    def __init__(self, labels_path, images_dir):
        super().__init__()
        self.labels_path = Path(labels_path)
        self.images_dir  = Path(images_dir)
        self._cache = {}
        self._samples: List[Dict[str, Any]] = []
        self._load()

    def _episodes(self, ep_id: int) -> List[Dict]:
        if ep_id not in self._cache:
            eps: Dict[int, List[Dict]] = defaultdict(list)
            with self.labels_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    eps[r["episode_id"]].append(r)
            self._cache = eps
        return self._cache[ep_id]


# =============================================================================
# Force noise model
# =============================================================================

def add_force_noise(F_clean: torch.Tensor,
                    sigma_drift: float, sigma_coupled: float, sigma_white: float,
                    ) -> torch.Tensor:
    """Random-walk drift + cross-axis coupling + white noise."""
    B, T, D = F_clean.shape
    dev = F_clean.device
    drift  = torch.cumsum(torch.randn(B, T, D, device=dev) * sigma_drift, dim=1)
    couple = torch.randn(B, T, D, device=dev) * sigma_coupled
    white  = torch.randn(B, T, D, device=dev) * sigma_white
    return F_clean + drift + couple + white


def sim2real_visual_augment(imgs: torch.Tensor) -> torch.Tensor:
    """Apply sim-to-real visual perturbations for VICReg dual-forward.

    Args:
        imgs: (N, 3, H, W) normalised image batch.
    Returns:
        Augmented batch of same shape, with independent perturbations per image.
    """
    import torchvision.transforms.functional as VF
    N = imgs.shape[0]
    # Work in [0, 1] space for augmentation, then re-normalise
    mean_t = torch.tensor(IMAGENET_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=imgs.device).view(1, 3, 1, 1)
    x = imgs * std_t + mean_t  # un-normalise to [0, 1]
    x = x.clamp(0, 1)

    for i in range(N):
        # Color jitter (brightness, contrast, saturation, hue)
        xi = x[i]
        xi = VF.adjust_brightness(xi, float(0.8 + 0.4 * torch.rand(1).item()))
        xi = VF.adjust_contrast(xi,   float(0.8 + 0.4 * torch.rand(1).item()))
        xi = VF.adjust_saturation(xi, float(0.7 + 0.6 * torch.rand(1).item()))
        if torch.rand(1).item() > 0.5:
            xi = VF.gaussian_blur(xi, kernel_size=3)
        # Small Gaussian noise
        if torch.rand(1).item() > 0.5:
            xi = xi + torch.randn_like(xi) * 0.02
        x[i] = xi.clamp(0, 1)

    return (x - mean_t) / std_t  # re-normalise


# =============================================================================
# VICReg loss
# =============================================================================

def vicreg_loss(Z1: torch.Tensor, Z2: torch.Tensor,
                lambda_var: float = 1.0, lambda_cov: float = 1.0,
                eps: float = 1e-4) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, D = Z1.shape
    L_inv = F.mse_loss(Z1, Z2)

    Z = torch.cat([Z1, Z2], dim=0)
    std_Z = torch.sqrt(Z.var(dim=0) + eps)
    L_var = torch.mean(F.relu(1.0 - std_Z))

    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.T @ Zc) / (2 * B - 1)
    off_diag = cov * (1.0 - torch.eye(D, device=Z.device, dtype=cov.dtype))
    L_cov = (off_diag ** 2).sum() / D

    total = L_inv + lambda_var * L_var + lambda_cov * L_cov
    return total, {"inv": float(L_inv.item()), "var": float(L_var.item()), "cov": float(L_cov.item())}


# =============================================================================
# Backbone factory
# =============================================================================

def _build_backbone(name: str, pretrained: bool, freeze: bool):
    """Return (backbone_module, feature_dim, encode_fn).

    ``encode_fn`` takes (B*F, 3, H, W) and returns (B*F, feature_dim).
    """
    import torchvision

    name = name.lower()

    # ── ResNet family ──
    if name in ("resnet18", "resnet50"):
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        backbone = nn.Sequential(*list(net.children())[:-2])  # drop FC+pool
        if name == "resnet18":
            dim = 512
        else:
            dim = 2048

        gap = nn.AdaptiveAvgPool2d(1)

        def encode_resnet(x):
            with torch.no_grad():
                feats = backbone(x)       # (B*F, dim, H', W')
            return gap(feats).flatten(1)  # (B*F, dim)

        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False

        return backbone, dim, encode_resnet

    # ── DINOv2 family (local cache — no network) ──
    if name.startswith("dinov2_"):
        import importlib, os as _os
        variant = name.replace("dinov2_", "")
        if variant == "vits14":
            dim = 384
        elif variant == "vitb14":
            dim = 768
        else:
            raise ValueError(f"Unknown DINOv2 variant: {name}")

        hub_dir = _os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        ckpt_path = _os.path.expanduser(
            f"~/.cache/torch/hub/checkpoints/dinov2_{variant}_pretrain.pth")

        if not _os.path.isdir(hub_dir):
            raise RuntimeError(
                f"DINOv2 cache not found at {hub_dir}.  "
                "Run once with network access or copy the cache manually."
            )

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

        if freeze:
            for p in model.parameters():
                p.requires_grad = False

        def encode_dinov2(x):
            with torch.no_grad():
                return model(x)

        return model, dim, encode_dinov2

    raise ValueError(f"Unknown backbone: {name}. "
                     f"Choices: resnet18, resnet50, dinov2_vits14, dinov2_vitb14")


# =============================================================================
# Model
# =============================================================================

class SymmetricDynamicsEncoder(nn.Module):
    """Visuo-force-action dynamics encoder per idea.md §3.

    Configurable backbone (ResNet-18/50 or DINOv2 ViT-S/B) → ProjMLP →
    FusionMLP → v_single. ForceMLP(18→64→64) → f_t.
    VICReg on [Δv, Δf] concat (vis_dim+64).

    Dynamics heads receive full causal context (idea.md §3.4 modified):
      Δv + a_t + p_t → DynamicsHead_vis → Δp_base
      Δf + a_t + p_t → DynamicsHead_force → Δp_resid
      Δf + a_t      → GateMLP → α
    Action is the cause of Δp; proprio grounds "where we started".
    """

    def __init__(self, backbone_name: str = "resnet18",
                 pretrained: bool = True, freeze_backbone: bool = True,
                 mono: bool = False):
        super().__init__()
        self._mono = mono
        self._backbone, self._vis_dim, self._encode_frame_fn = \
            _build_backbone(backbone_name, pretrained, freeze_backbone)

        # Task output normalization (set externally via set_task_norm)
        self.register_buffer("task_mean", torch.zeros(3))
        self.register_buffer("task_std", torch.ones(3))

        vis_dim = self._vis_dim

        # Per-frame projection: vis_dim → vis_dim
        self.proj_mlp = nn.Sequential(
            nn.Linear(vis_dim, vis_dim),
            nn.GELU(),
            nn.Linear(vis_dim, vis_dim),
        )
        # Dual-camera fusion (stereo only)
        if not mono:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * vis_dim, vis_dim),
                nn.GELU(),
                nn.Linear(vis_dim, vis_dim),
            )
        else:
            self.fusion_mlp = None

        # Force pathway (LayerNorm learns input scaling — no hand-coded normalisation)
        self.force_mlp = nn.Sequential(           # 18 → 128 → 64
            nn.LayerNorm(18),
            nn.Linear(18, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Bottleneck: [v_t, Δv, Δf] → 256D (motion-aware, action-agnostic)
        self.bottleneck = nn.Sequential(
            nn.Linear(vis_dim * 2 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

        # Task head: [bottleneck, proprio] → held_pos_rel_fixed (3D peg→hole offset)
        self.head_task = nn.Sequential(
            nn.Linear(256 + 20, 128), nn.GELU(), nn.Linear(128, 3))

        # Future prediction head: bottleneck → ẑ_{t+K} (BYOL-style)
        self.head_future = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))

        # Dynamics heads:
        #   head_vis:  Δv(vis_dim) + a(6) + p(20) = vis_dim + 26  ← visual delta (like copy)
        #   head_force: Δf(64) + a(6) + p(20) = 90
        self.dynamics_head_vis = nn.Sequential(
            nn.Linear(vis_dim + 26, max(vis_dim // 2, 64)), nn.GELU(),
            nn.Linear(max(vis_dim // 2, 64), 128), nn.GELU(),
            nn.Linear(128, PROPRIO_DIM),
        )
        self.dynamics_head_force = nn.Sequential( # 90 → 64 → 32 → 20
            nn.Linear(90, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, PROPRIO_DIM),
        )

        # Gate — force context + action
        #   Δf(64) + a(6) = 70
        self.gate_mlp = nn.Sequential(            # 70 → 32 → 1
            nn.Linear(70, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def encode_visual(self, imgs_left: torch.Tensor,
                       imgs_right: torch.Tensor = None) -> torch.Tensor:
        """Encode last frame → v_single (vis_dim). Mono: single camera, no fusion."""
        xl = imgs_left[:, -1, :, :, :]           # (B, 3, H, W)
        fl = self.proj_mlp(self._encode_frame_fn(xl))           # (B, vis_dim)
        if self.fusion_mlp is not None and imgs_right is not None:
            xr = imgs_right[:, -1, :, :, :]
            fr = self.proj_mlp(self._encode_frame_fn(xr))
            return self.fusion_mlp(torch.cat([fl, fr], dim=-1))  # (B, vis_dim)
        return fl                                                # mono: direct

    def encode_force(self, ft: torch.Tensor) -> torch.Tensor:
        """Encode 3-frame force → f_stack (64D).

        Args:
            ft: (B, 18) — cat[F(t-2), F(t-1), F(t)]
        """
        return self.force_mlp(ft)               # (B, 64)

    def forward_pretrain(self,
                         img_left_t:  torch.Tensor, img_right_t: torch.Tensor,
                         img_left_tk: torch.Tensor, img_right_tk: torch.Tensor,
                         ft_t: torch.Tensor, ft_tk: torch.Tensor,
                         a: torch.Tensor,
                         p_t: torch.Tensor, p_tk: torch.Tensor,
                         ) -> Dict[str, torch.Tensor]:
        """Forward pass for pretraining — returns Δp, bottleneck, prog/future preds.

        Bottleneck receives [v_t, Δf, a, p] — full causal context for task progress.
        Dynamics heads unchanged (bypass architecture).
        """
        # Encode (single-frame)
        v_t   = self.encode_visual(img_left_t,  img_right_t)
        v_tk  = self.encode_visual(img_left_tk, img_right_tk)
        f_t   = self.encode_force(ft_t)
        f_tk  = self.encode_force(ft_tk)

        # Visual + force deltas (AFRO-style differencing, matches copy)
        Δv = v_t - v_tk                                             # (B, vis_dim)
        Δf = f_t - f_tk                                             # (B, 64)

        # Bottleneck: [visual, visual-delta, force-delta] — motion-aware
        z_t  = self.bottleneck(torch.cat([v_t,  Δv, Δf], dim=-1))   # (B, 256)
        z_tk = self.bottleneck(torch.cat([v_tk, Δv, Δf], dim=-1))   # past bottleneck

        # Task prediction: bottleneck + proprio bypass → held_pos_rel_fixed (3D, normalized)
        task_pred = self.head_task(torch.cat([z_t, p_t], dim=-1))   # (B, 3) — raw output (learns internal norm)
        z_fut = self.head_future(z_tk)                              # predict z_t from z_{t-k}

        # Causal contexts: Δv for visual head (matches copy), Δf for force head
        ctx_vis_fwd   = torch.cat([Δv,   a,  p_t],  dim=-1)
        ctx_vis_bwd   = torch.cat([-Δv, -a,  p_tk], dim=-1)
        ctx_force_fwd = torch.cat([Δf,   a,  p_t],  dim=-1)
        ctx_force_bwd = torch.cat([-Δf, -a,  p_tk], dim=-1)
        ctx_gate_fwd  = torch.cat([Δf,   a], dim=-1)
        ctx_gate_bwd  = torch.cat([-Δf, -a], dim=-1)

        # Dynamics predictions (unchanged)
        Δp_base_fwd = self.dynamics_head_vis(ctx_vis_fwd)
        Δp_base_bwd = self.dynamics_head_vis(ctx_vis_bwd)
        Δp_resid_fwd = self.dynamics_head_force(ctx_force_fwd)
        Δp_resid_bwd = self.dynamics_head_force(ctx_force_bwd)
        α_fwd = self.gate_mlp(ctx_gate_fwd)
        α_bwd = self.gate_mlp(ctx_gate_bwd)
        Δp_hat_fwd = (1 - α_fwd) * Δp_base_fwd + α_fwd * Δp_resid_fwd
        Δp_hat_bwd = (1 - α_bwd) * Δp_base_bwd + α_bwd * Δp_resid_bwd

        return {
            "Δp_hat_fwd": Δp_hat_fwd, "Δp_hat_bwd": Δp_hat_bwd,
            "Δp_base_fwd": Δp_base_fwd, "Δp_base_bwd": Δp_base_bwd,
            "Δp_resid_fwd": Δp_resid_fwd, "Δp_resid_bwd": Δp_resid_bwd,
            "α_fwd": α_fwd, "α_bwd": α_bwd,
            "v_t": v_t, "v_tk": v_tk, "Δv": Δv, "Δf": Δf,
            "z_t": z_t, "task_pred": task_pred, "z_fut": z_fut,
        }

    def get_inference_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state_dict for downstream RL: backbone + ProjMLP + FusionMLP
        + ForceMLP + Bottleneck + head_prog → outputs (256D, 1D)."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("dynamics_head_") and not k.startswith("gate_mlp.")
                and not k.startswith("head_future.")}  # keeps head_task, bottleneck, etc.

    @property
    def feature_dim(self) -> int:
        """Dimension of the bottleneck feature RL receives."""
        return 256


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch,
                dp_scale: torch.Tensor = None,
                w_dynamics=1.0, w_vicreg=0.3, w_task=1.0, w_fut=0.5,
                warmup_inv_cons=5, warmup_vicreg=5, warmup_force_noise=10,
                sigma_drift=0.05, sigma_coupled=0.1, sigma_white=0.1,
                ) -> Dict[str, float]:
    model.train()
    mets = defaultdict(float)
    use_inv = epoch >= warmup_inv_cons
    use_vic = epoch >= warmup_vicreg
    use_fn = epoch >= warmup_force_noise
    inv_r = 0.3 if use_inv else 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}")
    for batch in pbar:
        il_t  = batch["img_left_t"].to(device)
        ir_t  = batch["img_right_t"].to(device)
        il_tk = batch["img_left_tk"].to(device)
        ir_tk = batch["img_right_tk"].to(device)
        ft_t  = batch["ft_t"].to(device)
        ft_tk = batch["ft_tk"].to(device)
        p_t   = batch["p_t"].to(device)
        p_tk  = batch["p_tk"].to(device)
        a     = batch["a"].to(device)
        held_rel_gt = batch["held_pos_rel_fixed"].to(device)        # (B, 3)
        B = il_t.shape[0]
        Δp_gt = (p_t - p_tk) / dp_scale

        # Dynamics
        out = model.forward_pretrain(il_t, ir_t, il_tk, ir_tk, ft_t, ft_tk,
                                     a, p_t, p_tk)
        L_fwd = F.smooth_l1_loss(out["Δp_hat_fwd"] / dp_scale, Δp_gt)
        L_bwd = F.smooth_l1_loss(out["Δp_hat_bwd"] / dp_scale, -Δp_gt)
        L_cons = F.smooth_l1_loss(out["Δp_hat_fwd"] / dp_scale, -out["Δp_hat_bwd"] / dp_scale)
        L_dyn = L_fwd + L_bwd + inv_r * L_cons

        # Task prediction (held_pos_rel_fixed 3D, normalized target)
        held_rel_norm = (held_rel_gt - model.task_mean) / model.task_std
        L_task = F.huber_loss(out["task_pred"], held_rel_norm)
        # Cosine distance: scale-invariant, large signals for small angular changes
        z_fut_n = F.normalize(out["z_fut"], dim=-1)
        z_tgt_n = F.normalize(out["z_t"].detach(), dim=-1)
        L_fut = 2 - 2 * (z_fut_n * z_tgt_n).sum(dim=-1).mean()  # cosine distance ∈ [0, 4]

        # VICReg: bottleneck from two visual+force perturbations
        if use_vic:
            # Force perturbation (t and t-k)
            ft_t_noisy = add_force_noise(
                ft_t.view(B, FT_FRAMES, FT_DIM),
                sigma_drift=sigma_drift if use_fn else 0.0,
                sigma_coupled=sigma_coupled if use_fn else 0.0,
                sigma_white=sigma_white if use_fn else 0.0,
            ).view(B, -1)
            ft_tk_noisy = add_force_noise(
                ft_tk.view(B, FT_FRAMES, FT_DIM),
                sigma_drift=sigma_drift if use_fn else 0.0,
                sigma_coupled=sigma_coupled if use_fn else 0.0,
                sigma_white=sigma_white if use_fn else 0.0,
            ).view(B, -1)

            # Visual perturbation — t-frame only, two augmentations
            il_last = il_t[:, -1, :, :, :];  ir_last = ir_t[:, -1, :, :, :]
            il_a1 = sim2real_visual_augment(il_last)
            ir_a1 = sim2real_visual_augment(ir_last)
            il_a2 = sim2real_visual_augment(il_last)
            ir_a2 = sim2real_visual_augment(ir_last)

            v_t1 = model.encode_visual(il_a1.unsqueeze(1), ir_a1.unsqueeze(1))
            v_t2 = model.encode_visual(il_a2.unsqueeze(1), ir_a2.unsqueeze(1))

            # Force deltas
            f_t   = model.encode_force(ft_t)
            f_tk  = model.encode_force(ft_tk)
            f_t_n = model.encode_force(ft_t_noisy)
            f_tk_n = model.encode_force(ft_tk_noisy)
            Δf1 = f_t - f_tk
            Δf2 = f_t_n - f_tk_n

            # Visual deltas (t-frame augmented, t-k frame clean from forward pass)
            Δv1 = v_t1 - out["v_tk"]
            Δv2 = v_t2 - out["v_tk"]

            b1 = model.bottleneck(torch.cat([v_t1, Δv1, Δf1], dim=-1))
            b2 = model.bottleneck(torch.cat([v_t2, Δv2, Δf2], dim=-1))
            L_vic, vc = vicreg_loss(b1, b2, lambda_var=0.33, lambda_cov=0.33)
        else:
            L_vic = torch.tensor(0.0, device=device)
            vc = {"inv": 0.0, "var": 0.0, "cov": 0.0}

        loss = w_dynamics * L_dyn + w_vicreg * L_vic + w_task * L_task + w_fut * L_fut

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mets["total"] += loss.item()
        mets["fwd"]   += L_fwd.item()
        mets["bwd"]   += L_bwd.item()
        mets["cons"]  += L_cons.item()
        mets["task"]  += L_task.item()
        mets["fut"]   += L_fut.item()
        mets["vic"]   += L_vic.item()
        mets["a_fwd"] += out["α_fwd"].mean().item()
        for k, v in vc.items():
            mets[f"vic_{k}"] += v

        pbar.set_postfix(L=f"{loss.item():.4f}", fwd=f"{L_fwd.item():.4f}",
                         task=f"{L_task.item():.4f}")

    n = max(1, len(dataloader))
    return {k: v / n for k, v in mets.items()}


@torch.no_grad()
def validate(model, dataloader, device, dp_scale: torch.Tensor) -> Dict[str, float]:
    model.eval()
    mets = defaultdict(float)
    for batch in dataloader:
        il_t  = batch["img_left_t"].to(device)
        ir_t  = batch["img_right_t"].to(device)
        il_tk = batch["img_left_tk"].to(device)
        ir_tk = batch["img_right_tk"].to(device)
        ft_t  = batch["ft_t"].to(device)
        ft_tk = batch["ft_tk"].to(device)
        p_t   = batch["p_t"].to(device)
        p_tk  = batch["p_tk"].to(device)
        a     = batch["a"].to(device)
        held_rel_gt = batch["held_pos_rel_fixed"].to(device)  # (B, 3)

        out = model.forward_pretrain(il_t, ir_t, il_tk, ir_tk, ft_t, ft_tk,
                                     a, p_t, p_tk)
        Δp_gt = (p_t - p_tk) / dp_scale
        mets["fwd"] += F.smooth_l1_loss(out["Δp_hat_fwd"] / dp_scale, Δp_gt).item()
        mets["bwd"] += F.smooth_l1_loss(out["Δp_hat_bwd"] / dp_scale, -Δp_gt).item()
        mets["α"]   += out["α_fwd"].mean().item()
        # Task prediction L2 error (mm): de-normalize pred, GT is already mm
        task_pred_mm = out["task_pred"] * model.task_std + model.task_mean
        mets["task_l2_mm"] += torch.sqrt(F.mse_loss(task_pred_mm, held_rel_gt)).item()

    n = max(1, len(dataloader))
    return {k: v / n for k, v in mets.items()}


def sigma_anneal(epoch: int, start: int, end: int, sigma_max: float) -> float:
    if epoch < start:
        return 0.0
    if epoch > end:
        return sigma_max
    return sigma_max * (epoch - start) / (end - start)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage E: pretrain visuo-force dynamics encoder")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / "pretrain_config.yaml"),
                        help="YAML config file (default: pretrain_config.yaml next to this script)")
    # CLI overrides for config values
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args_cli = parser.parse_args()

    # Load config, CLI overrides take precedence
    cfg = {}
    if Path(args_cli.config).exists():
        with open(args_cli.config) as f:
            cfg = yaml.safe_load(f)
    pt = cfg.get("pretraining", cfg)  # support nested "pretraining:" key or flat
    config_dir = Path(args_cli.config).resolve().parent

    def _get(key, default):
        cli_val = getattr(args_cli, key.replace("-", "_"), None)
        if cli_val is not None:
            return cli_val
        return pt.get(key, default)

    def _resolve_path(key, default):
        """Resolve a path value relative to the config file directory."""
        val = _get(key, default)
        p = Path(val)
        if p.is_absolute():
            return str(p)
        return str(config_dir / p)

    args = argparse.Namespace(
        data_dir=_resolve_path("data_dir", "./data"),
        output_dir=_resolve_path("output_dir", "./outputs/pretrain"),
        epochs=_get("epochs", 100),
        batch_size=_get("batch_size", 32),
        lr=_get("lr", 1e-3),
        weight_decay=_get("weight_decay", 1e-4),
        w_dynamics=_get("w_dynamics", 1.0),
        w_vicreg=_get("w_vicreg", 0.3),
        w_task=_get("w_task", 1.0),
        w_fut=_get("w_fut", 0.5),
        sigma_end=_get("sigma_end", 2.0),
        warmup_inv_cons=_get("warmup_inv_cons", 5),
        warmup_vicreg=_get("warmup_vicreg", 5),
        warmup_force_noise=_get("warmup_force_noise", 10),
        sigma_start=_get("sigma_start", 0.1),
        sigma_start_epoch=_get("sigma_start_epoch", 10),
        sigma_end_epoch=_get("sigma_end_epoch", 80),
        device=_get("device", "cuda"),
        num_workers=_get("num_workers", 4),
        seed=_get("seed", 42),
        no_pretrained_resnet=not pt.get("pretrained_resnet", True),
        ablation=pt.get("ablation", {}),
        backbone=pt.get("backbone", "resnet18"),
        pretrained_backbone=pt.get("pretrained_backbone", True),
        freeze_backbone=pt.get("freeze_backbone", True),
        mono=pt.get("mono", False),
    )

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to a log file for debugging
    import builtins
    log_file = open(output_dir / "summary.json", "a", buffering=1)
    _orig_print = builtins.print
    def _tee_print(*args, **kwargs):
        import io
        _orig_print(*args, **kwargs)
        buf = io.StringIO()
        _orig_print(*args, file=buf, **kwargs)
        log_file.write(buf.getvalue())
        log_file.flush()
    builtins.print = _tee_print

    writer = SummaryWriter(output_dir / "tb")

    # Datasets
    ds_train = StageEDataset(data_dir / "train" / "labels.jsonl", data_dir / "train" / "images")
    ds_val   = StageEDataset(data_dir / "val"   / "labels.jsonl", data_dir / "val"   / "images")
    print(f"Train: {len(ds_train)} samples, Val: {len(ds_val)} samples")

    # Per-dim Δp std for loss normalisation (avoids trivial zero-prediction)
    _dps = []
    _held_rels = []
    for i in range(min(500, len(ds_train))):
        s = ds_train[i]
        _dps.append(s["p_t"] - s["p_tk"])
        _held_rels.append(s["held_pos_rel_fixed"])
    dp_scale = torch.stack(_dps).std().to(device) + 1e-8  # scalar RMS
    held_rel = torch.stack(_held_rels)
    task_mean = held_rel.mean(dim=0).to(device)            # (3,) — per-dim mean
    task_std  = held_rel.std(dim=0).clamp(min=1e-4).to(device)  # (3,) — per-dim std
    print(f"Δp overall RMS: {dp_scale.item():.5f}")
    print(f"held_pos_rel_fixed mean: {task_mean.tolist()} std: {task_std.tolist()}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Model
    model = SymmetricDynamicsEncoder(
        backbone_name=args.backbone,
        pretrained=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        mono=args.mono,
    ).to(device)
    model.task_mean.copy_(task_mean)
    model.task_std.copy_(task_std)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {args.backbone} ({model._vis_dim}D), "
          f"trainable: {n_params:,} params (~{n_params/1000:.0f}K)")
    print(f"Task norm: mean={task_mean.tolist()} std={task_std.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    best_val_fwd = float("inf")
    best_combined = float("inf")
    early_stop_patience = pt.get("early_stop_patience", 0)
    no_improve_count = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        s = sigma_anneal(epoch, args.sigma_start_epoch, args.sigma_end_epoch, args.sigma_end)

        train_m = train_epoch(
            model, dl_train, optimizer, device, epoch,
            w_dynamics=args.w_dynamics,
            w_vicreg=args.w_vicreg if epoch >= args.warmup_vicreg else 0.0,
            w_task=args.w_task,
            w_fut=args.w_fut,
            warmup_inv_cons=args.warmup_inv_cons,
            warmup_vicreg=args.warmup_vicreg,
            warmup_force_noise=args.warmup_force_noise,
            sigma_drift=s, sigma_coupled=s, sigma_white=s,
            dp_scale=dp_scale,
        )
        val_m = validate(model, dl_val, device, dp_scale)

        scheduler.step(val_m["fwd"])
        elapsed = time.time() - t0

        # TensorBoard
        for k, v in {**train_m, **{f"val_{k2}": v2 for k2, v2 in val_m.items()}}.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("sigma", s, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        vic_var = train_m.get("vic_var", 0)
        print(f"Epoch {epoch:3d} | Σ {train_m['total']:.4f} fwd={train_m['fwd']:.4f} "
              f"task={train_m.get('task',0):.4f} "
              f"fut={train_m.get('fut',0):.4f} "
              f"val_fwd={val_m['fwd']:.4f} val_task={val_m.get('task_l2_mm',0):.2f}mm "
              f"α={train_m['a_fwd']:.3f} σ={s:.1f} "
              f"vic=[i={train_m.get('vic_inv',0):.3f} v={vic_var:.3f} c={train_m.get('vic_cov',0):.3f}] "
              f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.0f}s")
        if vic_var < 0.01 and epoch > args.warmup_vicreg:
            print(f"  ⚠ VICReg var={vic_var:.4f} → features may be collapsing. "
                  f"Consider increasing w_vicreg.")

        # Early stopping / best checkpoint: combined val_fwd + val_task_l2_mm
        no_improve_count += 1
        val_task = val_m.get("task_l2_mm", float("inf"))
        combined_score = val_m["fwd"] + 0.1 * val_task  # weight task error
        if combined_score < best_combined:
            best_combined = combined_score
            best_val_fwd = val_m["fwd"]
            no_improve_count = 0
            torch.save(model.get_inference_state_dict(), output_dir / "pretrained_encoder_best.pt")
            print(f"  → best (val_fwd={val_m['fwd']:.4f} task_l2={val_task:.2f}mm combined={combined_score:.4f})")
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f"Early stop at epoch {epoch}: no improvement for "
                  f"{early_stop_patience} epochs (best val_fwd={best_val_fwd:.4f})")
            break

        if epoch % 20 == 0:
            torch.save(model.get_inference_state_dict(), output_dir / f"pretrained_encoder_e{epoch:03d}.pt")

    torch.save(model.get_inference_state_dict(), output_dir / "pretrained_encoder_final.pt")
    writer.close()
    print(f"Done. Total time: {time.time() - t0:.0f}s. Output: {output_dir}")


if __name__ == "__main__":
    main()
