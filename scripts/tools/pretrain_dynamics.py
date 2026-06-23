#!/usr/bin/env python3
"""
Stage E V3: Dual-pathway visuo-force pretraining with MAE masked reconstruction.

Pathway1 (Visual-Geometry): head_vis(v_t,alpha_v,p_t)->z_vis(128D), head_task(z_vis,a)->held_pos_rel_fixed
Pathway2 (Force-Contact):   head_force(f_t,alpha_f,p_t)->z_force(128D), head_engagement(z_force)->eng
Dynamics:  head_dyn(alpha_v,alpha_f,p_t)->p_{t+K} (forward), inverse->p_t
MAE:       masked_encoder([z_vis,z_force,p_t], 50pct mask)->reconstruction

Loss: L_dyn = L_fwd + 0.3*L_inv, L_total = w_dyn*L_dyn + w_task*L_task + w_eng*L_eng + w_recon*L_recon
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
FT_FRAMES = 10      # force history frames per sample
FT_DIM = 6          # force/torque per frame
IMG_SIZE = 224
K = 3               # frame-skip for Δp computation
PROPRIO_DIM = 20    # qpos(7)+fingertip_pos(3)+fingertip_quat(4)+ee_linvel(3)+ee_angvel(3)
D_MAX = 5.0             # mm — clip held_rel target; beyond this only direction matters

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =============================================================================
# Dataset
# =============================================================================

class StageEDataset(Dataset):
    """Load Isaac Lab labels.jsonl, build sliding windows."""

    def _load(self) -> None:
        episodes: Dict[int, List[Dict]] = defaultdict(list)
        with self.labels_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episodes[rec["episode_id"]].append(rec)

        min_len = 2 * K + FT_FRAMES  # need t-K and t+K windows
        for ep_id, frames in episodes.items():
            if len(frames) < min_len:
                continue
            for t in range(FT_FRAMES - 1, len(frames)):
                t_k = t - K
                t_fwd = t + K
                if t_k < FT_FRAMES - 1 or t_fwd >= len(frames):
                    continue
                self._samples.append({"ep_id": ep_id, "t": t, "t_k": t_k, "t_fwd": t_fwd})

    def __len__(self) -> int:
        return len(self._samples)

    def _load_rgb(self, rel_path: str) -> torch.Tensor:
        path = self.images_dir / Path(rel_path).name
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        t = t.permute(2, 0, 1)
        for c in range(3):
            t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return t

    def _load_3frame_rgb(self, episode: List[Dict], idx: int, cam_key: str) -> torch.Tensor:
        frames = []
        for offset in range(-2, 1):
            i = max(0, min(idx + offset, len(episode) - 1))
            frames.append(self._load_rgb(episode[i][cam_key]))
        return torch.stack(frames, dim=0)

    def _load_window_ft(self, episode: List[Dict], idx: int) -> torch.Tensor:
        """Load N-frame force window ending at idx → (FT_FRAMES*FT_DIM,)."""
        vals = []
        for offset in range(-(FT_FRAMES - 1), 1):
            i = max(0, min(idx + offset, len(episode) - 1))
            ft = episode[i].get("force_torque", [0.0] * FT_DIM)
            vals.extend([float(v) for v in ft[:FT_DIM]])
        return torch.tensor(vals, dtype=torch.float32)

    def _load_proprio(self, episode: List[Dict], idx: int) -> torch.Tensor:
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
        i = max(0, min(idx, len(episode) - 1))
        rec = episode[i]
        a = rec.get("action", [0.0] * 6)
        return torch.tensor([float(v) for v in a[:6]], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self._samples[idx]
        ep_id = info["ep_id"]
        ep = self._episodes(ep_id)
        t, t_k = info["t"], info["t_k"]

        # Compute fingertip_pos_rel_fixed (world-frame, mm)
        fingertip_pos_v = [float(ep[t].get("fingertip_pos", [0.0,0.0,0.0])[i]) for i in range(3)]
        held_pos_v = [float(ep[t].get("held_pos", [0.0,0.0,0.0])[i]) for i in range(3)]
        held_rel_v = [float(ep[t].get("held_pos_rel_fixed", [0.0,0.0,0.0])[i]) for i in range(3)]
        fingertip_rel = torch.tensor(
            [(fingertip_pos_v[i] - held_pos_v[i] + held_rel_v[i]) * 1000 for i in range(3)],
            dtype=torch.float32)
        # XY direction: engaged → (0,0); otherwise normalize([x, y])
        engaged = float(ep[t].get("engagement_state", 0.0) > 0.5)
        if engaged:
            fingertip_dir_xy = torch.zeros(2)
        else:
            fingertip_dir_xy = fingertip_rel[:2] / (torch.norm(fingertip_rel[:2]) + 1e-8)

        return {
            "img_left_t":   self._load_3frame_rgb(ep, t,   "camera_left"),
            "img_right_t":  self._load_3frame_rgb(ep, t,   "camera_right"),
            "img_left_tk":  self._load_3frame_rgb(ep, t_k, "camera_left"),
            "img_right_tk": self._load_3frame_rgb(ep, t_k, "camera_right"),
            "ft_t":         self._load_window_ft(ep, t),
            "ft_tk":        self._load_window_ft(ep, t_k),
            "p_t":          self._load_proprio(ep, t),
            "p_tk":         self._load_proprio(ep, t_k),
            "p_fwd":        self._load_proprio(ep, info["t_fwd"]),
            "a":            self._load_action(ep, max(0, t - 1)),
            "fingertip_pos_rel_fixed": fingertip_rel,
            "fingertip_dir_xy": fingertip_dir_xy,
            "engagement":   torch.tensor(float(ep[t].get("engagement_state", 0.0) > 0.5)),
            "ep_id":        torch.tensor(info["ep_id"]),
            "t":            torch.tensor(t),
        }

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

# =============================================================================
# Backbone factory
# =============================================================================

def _build_backbone(name: str, pretrained: bool, freeze: bool):
    import torchvision
    name = name.lower()

    if name in ("resnet18", "resnet50"):
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        backbone = nn.Sequential(*list(net.children())[:-2])
        dim = 512 if name == "resnet18" else 2048
        gap = nn.AdaptiveAvgPool2d(1)

        def encode_fn(x):
            with torch.no_grad():
                feats = backbone(x)
            return gap(feats).flatten(1)

        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False
        return backbone, dim, encode_fn

    if name.startswith("dinov2_"):
        import importlib as _il
        variant = name.replace("dinov2_", "").replace("_patches", "").replace("_attn", "")
        dim = {"vits14": 384, "vitb14": 768}[variant]
        hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        ckpt_path = os.path.expanduser(
            f"~/.cache/torch/hub/checkpoints/dinov2_{variant}_pretrain.pth")
        if not os.path.isdir(hub_dir):
            raise RuntimeError(f"DINOv2 cache not found at {hub_dir}.")
        old_path = sys.path.copy()
        sys.path.insert(0, hub_dir)
        try:
            hub_module = _il.import_module("hubconf")
            model_fn = getattr(hub_module, f"dinov2_{variant}")
            dino = model_fn(pretrained=False)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            dino.load_state_dict(state_dict, strict=True)
        finally:
            sys.path = old_path

        use_patches = name.endswith("_patches")
        use_attn = name.endswith("_attn")

        if use_attn:
            # Learnable spatial attention over patch tokens
            class PatchAttnEncoder(nn.Module):
                def __init__(self, dino_model, dim):
                    super().__init__()
                    self.dino = dino_model
                    self.attn = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
                    for p in self.dino.parameters():
                        p.requires_grad = False

                def forward(self, x):
                    with torch.no_grad():
                        patches = self.dino.get_intermediate_layers(
                            x, n=1, return_class_token=False)[0]  # (B, 256, D)
                    w = F.softmax(self.attn(patches).squeeze(-1), dim=-1)  # (B, 256)
                    return (patches * w.unsqueeze(-1)).sum(dim=1)  # (B, D)

            model = PatchAttnEncoder(dino, dim)
            dim_out = dim

            def encode_fn(x):
                return model(x)
        elif use_patches:
            # Mean pool over patch tokens
            for p in dino.parameters():
                p.requires_grad = False
            model = dino
            dim_out = dim

            def encode_fn(x):
                with torch.no_grad():
                    out = model.get_intermediate_layers(x, n=1, return_class_token=False)
                    return out[0].mean(dim=1)
        else:
            # CLS token
            if freeze:
                for p in dino.parameters():
                    p.requires_grad = False
            model = dino
            dim_out = dim

            def encode_fn(x):
                with torch.no_grad():
                    return model(x)

        return model, dim_out, encode_fn

    raise ValueError(f"Unknown backbone: {name}")


# =============================================================================
# Model
# =============================================================================

class SymmetricDynamicsEncoder(nn.Module):
    """V3: Dual-pathway — visual-geometry + force-contact, no bottleneck."""

    def __init__(self, backbone_name: str = "resnet18",
                 pretrained: bool = True, freeze_backbone: bool = True,
                 mono: bool = False):
        super().__init__()
        self._mono = mono
        self._backbone, self._vis_dim, self._encode_frame_fn = \
            _build_backbone(backbone_name, pretrained, freeze_backbone)
        D = self._vis_dim

        # ── Task normalization ──
        self.register_buffer("task_mean", torch.zeros(3))
        self.register_buffer("task_std", torch.ones(3))

        # ── Per-frame projection ──
        self.proj_mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

        # ── Stereo fusion ──
        if not mono:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        else:
            self.fusion_mlp = None

        # ── Force pathway (10 frames × 6 axes = 60D) ──
        ft_in = FT_FRAMES * FT_DIM  # 60
        self.force_mlp = nn.Sequential(
            nn.LayerNorm(ft_in), nn.Linear(ft_in, 128), nn.GELU(),
            nn.Linear(128, 64))

        # ── DiffExtract: 差分 → 潜特征 ──
        self.diff_extract_v = nn.Sequential(
            nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32))
        self.diff_extract_f = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 32))

        # ── Pathway 1: Visual-Geometry ──
        # v_t(D) + α_v(32) + p_t(20) → z_vis(128)
        self.head_vis_fc1 = nn.Linear(D + 32 + 20, 256)
        self.head_vis_fc2 = nn.Linear(256, 128)
        # task head: z_vis(128) + a(6) → held_pos_rel_fixed
        self.head_task = nn.Sequential(
            nn.Linear(128 + 6, 128), nn.GELU(), nn.Linear(128, 2))  # XY direction only

        # ── Pathway 2: Force-Contact ──
        # f_t(64) + α_f(32) + p_t(20) → z_force(128)
        self.head_force = nn.Sequential(
            nn.Linear(64 + 32 + 20, 64), nn.GELU(), nn.Linear(64, 128))
        # engagement: z_vis(128)+z_force(128) → eng_pred (visual + force)
        self.head_engagement = nn.Sequential(
            nn.Linear(128 + 128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

        # ── Unified dynamics: α_v(32)+α_f(32)+p_t(20) → p_{t+K}(20) ──
        self.head_dyn = nn.Sequential(
            nn.Linear(84, 128), nn.GELU(), nn.Linear(128, 20))

        # ── MAE-style masked latent autoencoder ──
        # Concat [z_vis(128), z_force(128), p_t(20)] = 276D,
        # randomly mask 50% dims, MLP reconstructs full vector from survivors.
        self.masked_encoder = nn.Sequential(
            nn.Linear(276, 256), nn.GELU(), nn.Linear(256, 276))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_visual(self, imgs_left: torch.Tensor,
                       imgs_right: torch.Tensor = None) -> torch.Tensor:
        xl = imgs_left[:, -1, :, :, :] if imgs_left.dim() == 5 else imgs_left
        fl = self.proj_mlp(self._encode_frame_fn(xl))
        if self.fusion_mlp is not None and imgs_right is not None:
            xr = imgs_right[:, -1, :, :, :] if imgs_right.dim() == 5 else imgs_right
            fr = self.proj_mlp(self._encode_frame_fn(xr))
            return self.fusion_mlp(torch.cat([fl, fr], dim=-1))
        return fl

    def encode_force(self, ft: torch.Tensor) -> torch.Tensor:
        return self.force_mlp(ft)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_pretrain(self,
                         img_left_t:  torch.Tensor, img_right_t: torch.Tensor,
                         img_left_tk: torch.Tensor, img_right_tk: torch.Tensor,
                         ft_t: torch.Tensor, ft_tk: torch.Tensor,
                         a: torch.Tensor,
                         p_t: torch.Tensor, p_tk: torch.Tensor, p_fwd: torch.Tensor,
                         ) -> Dict[str, torch.Tensor]:
        B = img_left_t.shape[0]

        # 1. Encode
        v_t  = self.encode_visual(img_left_t,  img_right_t)
        v_tk = self.encode_visual(img_left_tk, img_right_tk)
        f_t  = self.encode_force(ft_t)
        f_tk = self.encode_force(ft_tk)

        # 2. Deltas + DiffExtract
        Δv, Δf = v_t - v_tk, f_t - f_tk
        α_v = self.diff_extract_v(Δv)
        α_f = self.diff_extract_f(Δf)

        # 3. Pathway 1: Visual-Geometry
        h_vis = F.gelu(self.head_vis_fc1(torch.cat([v_t, α_v, p_t], dim=-1)))
        z_vis = F.gelu(self.head_vis_fc2(h_vis))                 # (B,128)
        task_pred = self.head_task(torch.cat([z_vis, a], dim=-1))  # (B,3)

        # 4. Pathway 2: Force-Contact
        h_force = F.gelu(self.head_force(torch.cat([f_t, α_f, p_t], dim=-1)))
        z_force = h_force                                        # (B,128)
        eng_pred = self.head_engagement(torch.cat([z_vis, z_force], dim=-1))  # (B,1)

        # 5. MAE masked reconstruction: 50% dims masked, reconstruct all
        feats = torch.cat([z_vis, z_force, p_t], dim=-1)          # (B, 276)
        mask = (torch.rand(B, 276, device=z_vis.device) < 0.5)    # 50% mask
        feats_masked = feats * (~mask)
        feats_recon = self.masked_encoder(feats_masked)           # (B, 276)
        z_vis_cross = feats_recon[:, :128]
        z_force_cross = feats_recon[:, 128:256]

        # 5. Unified dynamics (forward: p_t → p_{t+K}, inverse: p_{t+K} → p_t)
        p_fwd = self.head_dyn(torch.cat([α_v, α_f, p_t], dim=-1))
        α_v_inv = self.diff_extract_v(-Δv)
        α_f_inv = self.diff_extract_f(-Δf)
        p_inv = self.head_dyn(torch.cat([α_v_inv, α_f_inv, p_fwd], dim=-1))

        return {
            "z_vis": z_vis, "z_force": z_force,
            "task_pred": task_pred, "eng_pred": eng_pred,
            "p_fwd": p_fwd, "p_inv": p_inv,
            "v_t": v_t, "v_tk": v_tk, "f_t": f_t, "f_tk": f_tk,
            "Δv": Δv, "Δf": Δf,
            "z_vis_cross": z_vis_cross, "z_force_cross": z_force_cross, "feats_recon": feats_recon,
        }

    def get_inference_state_dict(self) -> Dict[str, torch.Tensor]:
        """Keep encoders + both pathways. Exclude dynamics/aux heads."""
        exclude = ("head_dyn.",)
        return {k: v for k, v in self.state_dict().items()
                if not any(k.startswith(p) for p in exclude)}

    @property
    def feature_dim(self) -> int:
        return 256


# =============================================================================
# Training
# =============================================================================

def sim2real_visual_augment(imgs: torch.Tensor) -> torch.Tensor:
    """Per-image visual augmentation: brightness, contrast, saturation, blur, noise.
    imgs: (N, 3, H, W) normalized by ImageNet stats. Returns same shape."""
    import torchvision.transforms.functional as VF
    mean_t = torch.tensor(IMAGENET_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=imgs.device).view(1, 3, 1, 1)
    x = imgs * std_t + mean_t  # de-normalize
    x = x.clamp(0, 1)
    N = x.shape[0]
    for i in range(N):
        xi = x[i]
        xi = VF.adjust_brightness(xi, float(0.75 + 0.50 * torch.rand(1).item()))
        xi = VF.adjust_contrast(xi,   float(0.75 + 0.50 * torch.rand(1).item()))
        xi = VF.adjust_saturation(xi, float(0.60 + 0.80 * torch.rand(1).item()))
        if torch.rand(1).item() > 0.5:
            xi = VF.gaussian_blur(xi, kernel_size=3)
        if torch.rand(1).item() > 0.5:
            xi = xi + torch.randn_like(xi) * 0.03
        x[i] = xi.clamp(0, 1)
    return (x - mean_t) / std_t


def train_epoch(model, dataloader, optimizer, device, epoch,
                dp_scale: torch.Tensor,
                w_dynamics=1.0, w_task=1.0, w_eng=1.0, w_recon=0.1,
                warmup_inv_cons=5,
                ) -> Dict[str, float]:
    model.train()
    mets = defaultdict(float)
    use_inv = epoch >= warmup_inv_cons
    inv_r = 0.5 if use_inv else 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}")
    for batch in pbar:
        il_t  = batch["img_left_t"].to(device)
        ir_t  = batch["img_right_t"].to(device)
        il_tk = batch["img_left_tk"].to(device)
        ir_tk = batch["img_right_tk"].to(device)
        # Visual augmentation (per-frame)
        B_img, T_img = il_t.shape[0], il_t.shape[1]
        il_t  = sim2real_visual_augment(il_t.reshape(-1, 3, IMG_SIZE, IMG_SIZE)).view(B_img, T_img, 3, IMG_SIZE, IMG_SIZE)
        ir_t  = sim2real_visual_augment(ir_t.reshape(-1, 3, IMG_SIZE, IMG_SIZE)).view(B_img, T_img, 3, IMG_SIZE, IMG_SIZE)
        il_tk = sim2real_visual_augment(il_tk.reshape(-1, 3, IMG_SIZE, IMG_SIZE)).view(B_img, T_img, 3, IMG_SIZE, IMG_SIZE)
        ir_tk = sim2real_visual_augment(ir_tk.reshape(-1, 3, IMG_SIZE, IMG_SIZE)).view(B_img, T_img, 3, IMG_SIZE, IMG_SIZE)
        ft_t  = batch["ft_t"].to(device)
        ft_tk = batch["ft_tk"].to(device)
        p_t   = batch["p_t"].to(device)
        p_tk  = batch["p_tk"].to(device)
        a     = batch["a"].to(device)
        task_gt_pos = batch["fingertip_pos_rel_fixed"].to(device)  # for kd mask only
        task_gt_dir = batch["fingertip_dir_xy"].to(device)          # (B,2) XY unit direction
        eng_gt = batch["engagement"].to(device)
        p_fwd = batch["p_fwd"].to(device)
        B = il_t.shape[0]

        # Forward
        out = model.forward_pretrain(il_t, ir_t, il_tk, ir_tk, ft_t, ft_tk,
                                     a, p_t, p_tk, p_fwd)

        # ── Dynamics: p_{t+K} forward, p_t inverse ──
        L_fwd = F.smooth_l1_loss(out["p_fwd"] / dp_scale, p_fwd / dp_scale)
        L_inv  = F.smooth_l1_loss(out["p_inv"] / dp_scale, p_t / dp_scale) if use_inv else torch.tensor(0.0, device=device)
        L_dynamics = L_fwd + inv_r * L_inv

        # ── Task: cosine loss on XY direction (engaged→(0,0), skip origin) ──
        task_pred_dir = F.normalize(out["task_pred"], dim=-1, eps=1e-8)  # (B,2)
        kd_xy = torch.norm(task_gt_pos[:, :2], dim=-1)
        mask = (kd_xy > 1e-4) & (eng_gt < 0.5)  # skip origin + engaged (cosine undefined for (0,0))
        cos_sim = torch.sum(task_pred_dir * task_gt_dir, dim=-1)
        L_task = (1.0 - cos_sim[mask]).mean() if mask.any() else torch.tensor(0.0, device=device)
        L_eng  = F.binary_cross_entropy(out["eng_pred"].squeeze(-1), eng_gt.float())

        # ── MSDP denoising loss (masked dims only) ──
        feats_gt = torch.cat([out["z_vis"].detach(), out["z_force"].detach(),
                               p_t], dim=-1)
        L_recon = F.mse_loss(out["feats_recon"], feats_gt)

        loss = w_dynamics * L_dynamics + w_task * L_task + w_eng * L_eng + w_recon * L_recon

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mets["total"] += loss.item()
        mets["fwd"] += L_fwd.item()
        mets["inv_dyn"] += L_inv.item()
        mets["task"] += L_task.item()
        mets["eng"]  += L_eng.item()
        eng_acc = ((out["eng_pred"].squeeze(-1) > 0.5).float() == eng_gt.float()).float().mean()
        mets["eng_acc"] += eng_acc.item()
        mets["recon"] += L_recon.item()
        

        pbar.set_postfix(L=f"{loss.item():.4f}", fwd=f"{L_fwd.item():.4f}",
                         task=f"{L_task.item():.4f}", eng=f"{L_eng.item():.3f}")

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
        task_gt_pos = batch["fingertip_pos_rel_fixed"].to(device)  # for kd mask
        task_gt_dir = batch["fingertip_dir_xy"].to(device)          # (B,2)
        eng_gt = batch["engagement"].to(device)
        p_fwd_gt = batch["p_fwd"].to(device)

        out = model.forward_pretrain(il_t, ir_t, il_tk, ir_tk, ft_t, ft_tk,
                                     a, p_t, p_tk, p_fwd_gt)

        mets["fwd"] += F.smooth_l1_loss(
            out["p_fwd"] / dp_scale, p_fwd_gt / dp_scale).item()
        mets["inv"] += F.smooth_l1_loss(
            out["p_inv"] / dp_scale, p_t / dp_scale).item()

        # MAE reconstruction (reuse forward pass output)
        feats_v = torch.cat([out["z_vis"], out["z_force"], p_t], dim=-1)
        L_recon = F.mse_loss(out["feats_recon"], feats_v)

        # Task: cosine similarity on XY direction (skip engaged + origin)
        task_pred_dir = F.normalize(out["task_pred"], dim=-1, eps=1e-8)  # (B,2)
        kd_xy = torch.norm(task_gt_pos[:, :2], dim=-1)
        mask_v = (kd_xy > 1e-4) & (eng_gt < 0.5)  # skip origin + engaged
        if mask_v.any():
            cos_v = torch.sum(task_pred_dir[mask_v] * task_gt_dir[mask_v], dim=-1).mean()
            mets["task_cos"] += cos_v.item()

        # Engagement accuracy
        eng_acc = ((out["eng_pred"].squeeze(-1) > 0.5).float() == eng_gt.float()).float().mean()
        mets["eng_acc"] += eng_acc.item()
        mets["recon"] += L_recon.item()
        

    n = max(1, len(dataloader))
    return {k: v / n for k, v in mets.items()}


def main():
    parser = argparse.ArgumentParser(description="Stage E: pretrain dynamics encoder")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).resolve().parent / "pretrain_config.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args_cli = parser.parse_args()

    cfg = {}
    if Path(args_cli.config).exists():
        with open(args_cli.config) as f:
            cfg = yaml.safe_load(f)
    pt = cfg.get("pretraining", cfg)
    config_dir = Path(args_cli.config).resolve().parent

    def _get(key, default):
        cli_val = getattr(args_cli, key.replace("-", "_"), None)
        if cli_val is not None: return cli_val
        return pt.get(key, default)

    def _resolve_path(key, default):
        val = _get(key, default)
        p = Path(val)
        return str(p) if p.is_absolute() else str(config_dir / p)

    args = argparse.Namespace(
        data_dir=_resolve_path("data_dir", "./data"),
        output_dir=_resolve_path("output_dir", "./outputs/pretrain"),
        epochs=_get("epochs", 100),
        batch_size=_get("batch_size", 32),
        lr=_get("lr", 1e-3),
        weight_decay=_get("weight_decay", 1e-4),
        w_dynamics=_get("w_dyn", 1.0),
        w_task=_get("w_task", 1.0),
        w_eng=_get("w_eng", 1.0),
        w_recon=_get("w_recon", 0.1),
        warmup_inv_cons=_get("warmup_inv_cons", 5),
        device=_get("device", "cuda"),
        num_workers=_get("num_workers", 4),
        seed=_get("seed", 42),
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

    import builtins
    log_file = open(output_dir / "summary.txt", "w", buffering=1)
    _orig_print = builtins.print
    def _tee_print(*a, **kw):
        import io
        _orig_print(*a, **kw)
        buf = io.StringIO()
        _orig_print(*a, file=buf, **kw)
        log_file.write(buf.getvalue()); log_file.flush()
    builtins.print = _tee_print

    writer = SummaryWriter(output_dir / "tb")

    ds_train = StageEDataset(data_dir / "train" / "labels.jsonl", data_dir / "train" / "images")
    ds_val   = StageEDataset(data_dir / "val"   / "labels.jsonl", data_dir / "val"   / "images")
    print(f"Train: {len(ds_train)} samples, Val: {len(ds_val)} samples")

    _dps = []
    for i in range(min(500, len(ds_train))):
        s = ds_train[i]
        _dps.append(s["p_t"] - s["p_tk"])
    dp_scale = torch.stack(_dps).std().to(device) + 1e-8
    print(f"Δp overall RMS: {dp_scale.item():.5f}")
    print(f"Task head: direction (cosine loss, no norm stats needed)")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = SymmetricDynamicsEncoder(
        backbone_name=args.backbone, pretrained=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone, mono=args.mono).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {args.backbone} ({model._vis_dim}D), trainable: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15)

    best_combined = float("inf")
    early_stop_patience = pt.get("early_stop_patience", 0)
    no_improve_count = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(
            model, dl_train, optimizer, device, epoch,
            w_dynamics=args.w_dynamics,
            w_task=args.w_task,
            w_eng=args.w_eng,
            w_recon=args.w_recon,
            warmup_inv_cons=args.warmup_inv_cons,
            dp_scale=dp_scale,
        )
        val_m = validate(model, dl_val, device, dp_scale)

        val_task_cos = val_m.get("task_cos", 0.0)
        val_combined = (args.w_dynamics * val_m["fwd"]
                        + args.w_task * (1.0 - val_task_cos)
                        + args.w_eng * (1.0 - val_m.get("eng_acc", 0.5)))
        scheduler.step(val_combined)
        elapsed = time.time() - t0

        for k, v in {**train_m, **{f"val_{k2}": v2 for k2, v2 in val_m.items()}}.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:3d} | Σ {train_m['total']:.4f} "
              f"fwd={train_m['fwd']:.4f} inv={train_m['inv_dyn']:.4f} "
              f"recon={train_m['recon']:.4f} "
              f"task={train_m['task']:.4f} eng={train_m['eng']:.4f}(acc={train_m['eng_acc']:.2f}) "
              f"val_fwd={val_m['fwd']:.4f} val_cos={val_task_cos:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.0f}s")

        no_improve_count += 1
        if val_combined < best_combined:
            best_combined = val_combined
            no_improve_count = 0
            torch.save(model.get_inference_state_dict(),
                       output_dir / "pretrained_encoder_best.pt")
            print(f"  → best (combined={val_combined:.4f} "
                  f"fwd={val_m['fwd']:.4f} cos={val_task_cos:.3f} "
                  f"eng_acc={val_m.get('eng_acc',0):.3f})")
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f"Early stop at epoch {epoch}")
            break

        if epoch % 20 == 0:
            torch.save(model.get_inference_state_dict(),
                       output_dir / f"pretrained_encoder_e{epoch:03d}.pt")

    torch.save(model.get_inference_state_dict(), output_dir / "pretrained_encoder_final.pt")
    writer.close()
    print(f"Done. Total time: {time.time() - t0:.0f}s. Output: {output_dir}")


if __name__ == "__main__":
    main()
