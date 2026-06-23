#!/usr/bin/env python3
"""
Minimal verification: DINOv2 raw features + proprio → XY direction.

Tests whether DINOv2 frozen features retain enough spatial information
for XY direction prediction, bypassing all intermediate modules (proj_mlp,
fusion_mlp, force, diff_extract, dynamics, MAE, engagement).
"""
from __future__ import annotations

import argparse, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reuse dataset from pretrain_dynamics.py
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from pretrain_dynamics import StageEDataset, _build_backbone, FT_FRAMES, FT_DIM, IMG_SIZE, K

# ---------------------------------------------------------------------------
# CNN backbone (lightweight, trainable, spatial-preserving)
# ---------------------------------------------------------------------------

def _build_cnn_backbone():
    """Small CNN: 3→32→64→128→128→64, flatten(14×14×64)→Linear→384D."""
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(32), nn.GELU(),
        nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(64), nn.GELU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.GELU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.GELU(),
        nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.GELU(),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 384),
    )
    dim = 384

    def encode_fn(x):
        return cnn(x)

    return cnn, dim, encode_fn


# ---------------------------------------------------------------------------
# Model: visual backbone + proprio → MLP → XY direction
# ---------------------------------------------------------------------------

class VerifyModel(nn.Module):
    def __init__(self, backbone_name: str = "dinov2_vits14", mono: bool = False):
        super().__init__()
        self._mono = mono
        if backbone_name == "cnn_small":
            self._backbone, self._vis_dim, self._encode_frame_fn = _build_cnn_backbone()
        else:
            self._backbone, self._vis_dim, self._encode_frame_fn = \
                _build_backbone(backbone_name, pretrained=True, freeze=True)
        D = self._vis_dim

        vis_in = D if mono else 2 * D
        self.head = nn.Sequential(
            nn.Linear(vis_in + 20, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 2),
        )

    def encode_visual(self, imgs_left: torch.Tensor,
                       imgs_right: torch.Tensor = None) -> torch.Tensor:
        xl = imgs_left[:, -1, :, :, :] if imgs_left.dim() == 5 else imgs_left
        fl = self._encode_frame_fn(xl)
        if not self._mono and imgs_right is not None:
            xr = imgs_right[:, -1, :, :, :] if imgs_right.dim() == 5 else imgs_right
            fr = self._encode_frame_fn(xr)
            return torch.cat([fl, fr], dim=-1)
        return fl

    def forward(self, img_left_t, img_right_t, p_t):
        v = self.encode_visual(img_left_t, img_right_t)
        x = torch.cat([v, p_t], dim=-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    mets = defaultdict(float)
    pbar = tqdm(dataloader, desc="Train")
    for batch in pbar:
        il = batch["img_left_t"].to(device)
        ir = batch["img_right_t"].to(device)
        p_t = batch["p_t"].to(device)
        dir_gt = batch["fingertip_dir_xy"].to(device)

        pred_raw = model(il, ir, p_t)
        pred_dir = F.normalize(pred_raw, dim=-1, eps=1e-8)
        cos_sim = torch.sum(pred_dir * dir_gt, dim=-1)
        loss = (1.0 - cos_sim).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mets["loss"] += loss.item()
        mets["cos"] += cos_sim.mean().item()
        pbar.set_postfix(L=f"{loss.item():.4f}", cos=f"{cos_sim.mean().item():.3f}")

    n = max(1, len(dataloader))
    return {k: v / n for k, v in mets.items()}


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    mets = defaultdict(float)
    for batch in dataloader:
        il = batch["img_left_t"].to(device)
        ir = batch["img_right_t"].to(device)
        p_t = batch["p_t"].to(device)
        dir_gt = batch["fingertip_dir_xy"].to(device)

        pred_raw = model(il, ir, p_t)
        pred_dir = F.normalize(pred_raw, dim=-1, eps=1e-8)
        cos_sim = torch.sum(pred_dir * dir_gt, dim=-1)
        mets["loss"] += (1.0 - cos_sim).mean().item()
        mets["cos"] += cos_sim.mean().item()

    n = max(1, len(dataloader))
    return {k: v / n for k, v in mets.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify DINOv2 raw features for XY direction")
    parser.add_argument("--data-dir", type=str, default=str(THIS_DIR.parent.parent / "data"))
    parser.add_argument("--backbone", type=str, default="dinov2_vits14_attn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else \
        THIS_DIR.parent.parent / "output" / "pretrain_verify"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
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

    # Dataset
    ds_train = StageEDataset(data_dir / "train" / "labels.jsonl", data_dir / "train" / "images")
    ds_val = StageEDataset(data_dir / "val" / "labels.jsonl", data_dir / "val" / "images")
    print(f"Train: {len(ds_train)} samples, Val: {len(ds_val)} samples")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = VerifyModel(backbone_name=args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_backbone = "cnn_small" in args.backbone
    print(f"Backbone: {args.backbone} ({model._vis_dim}D) "
          f"{'(trainable)' if trainable_backbone else '(frozen)'}, "
          f"trainable: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    best_loss = float("inf")
    early_stop_patience = 20
    no_improve_count = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, dl_train, optimizer, device)
        val_m = validate(model, dl_val, device)
        scheduler.step(val_m["loss"])
        elapsed = time.time() - t0

        for k, v in {**train_m, **{f"val_{k}": v2 for k, v2 in val_m.items()}}.items():
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:3d} | train_loss={train_m['loss']:.4f} cos={train_m['cos']:.3f} "
              f"val_loss={val_m['loss']:.4f} val_cos={val_m['cos']:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.0f}s")

        no_improve_count += 1
        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            no_improve_count = 0
            torch.save(model.state_dict(), output_dir / "pretrained_verify_best.pt")
            print(f"  → best (val_loss={best_loss:.4f} val_cos={val_m['cos']:.3f})")
        if no_improve_count >= early_stop_patience:
            print(f"Early stop at epoch {epoch}")
            break

    torch.save(model.state_dict(), output_dir / "pretrained_verify_final.pt")
    writer.close()
    print(f"Done. Total time: {time.time() - t0:.0f}s. Output: {output_dir}")


if __name__ == "__main__":
    main()
