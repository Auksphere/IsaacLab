# 视-力动力学预训练架构 V4

## 1. 设计原则

1. **双通路独立**：视觉通路预测 XY 方向，力觉通路预测接触状态。两路不融合，各自产出 128D
2. **Spatial Attention**：DINOv2 patch tokens 通过 learnable attention 保留空间信息，替代 CLS token
3. **方向 + 接触解耦**：head_task 只预测 XY 方向（cosine loss），head_engagement 用视觉+力觉共同预测接触
4. **MAE 跨模态重建**：拼接 [z_vis, z_force, p_t] = 276D，随机掩码 50%，MLP 重建全部
5. **本体感知全程参与**：p_t 同时喂两个特征头、动力学头、MAE 编码器
6. **去冗余**：无 VICReg、无 bottleneck、无门控、无不确定性门

---

## 2. 数据流

```
img_L/R(t) → DINOv2(frozen) + SpatialAttn → ProjMLP → FusionMLP → v_t(384D)
F(t-9..t)  → ForceMLP(60→128→64) → f_t(64D)
p_t(20D)

时序差分 (K=3):
  Δv = v_t − v_{t-K} (384D)    Δf = f_t − f_{t-K} (64D)

潜动作提取:
  Δv → diff_extract_v: 384→128→64→32 → α_v(32D)
  Δf → diff_extract_f: 64→64→32→32   → α_f(32D)
```

---

## 3. 双特征通路

```
┌────────────────────────────────────────────────────────┐
│ 通路 1: 视觉-几何定位                                   │
│                                                        │
│ 输入: v_t(384) + α_v(32) + p_t(20) = 436D              │
│                                                        │
│ head_vis:                                              │
│   fc1: Linear(436→256) → GELU                          │
│   fc2: Linear(256→128) → GELU → z_vis(128D)             │
│                                                        │
│ head_task: z_vis(128) + a(6) → 128 → GELU → 2 → dir_xŷ │
│                                                        │
│ 监督: L_task = 1 - cos(dir_xŷ, dir_xy_gt)              │
│ 职责: "孔在指尖的哪个 XY 方向"                           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 通路 2: 力觉+视觉-接触感知                               │
│                                                        │
│ 输入: f_t(64) + α_f(32) + p_t(20) = 116D              │
│                                                        │
│ head_force:                                            │
│   fc1: Linear(116→64) → GELU                           │
│   fc2: Linear(64→128) → GELU → z_force(128D)           │
│                                                        │
│ head_engagement: z_vis(128)+z_force(128) → 32 → 1 → enĝ │
│                                                        │
│ 监督: L_eng = BCE(enĝ, eng_gt)                          │
│ 职责: "碰没碰" (视觉 + 力觉联合判断)                      │
└────────────────────────────────────────────────────────┘
```

---

## 4. 统一动力学预测

```
正向:
  α_v(32) + α_f(32) + p_t(20) → head_dyn: 84→128→GELU→20 → p̂_{t+K}

逆向 (共享权重):
  α_v_inv = diff_extract_v(−Δv)
  α_f_inv = diff_extract_f(−Δf)
  α_v_inv + α_f_inv + p_fwd_gt → head_dyn (同权重) → p̂_t

监督:
  L_fwd = Huber(p̂_{t+K}, p_{t+K})
  L_inv = Huber(p̂_t, p_t)       [warmup 5 epoch 后启用]
```

---

## 5. MAE 掩码重建

```
拼接: [z_vis(128), z_force(128), p_t(20)] = 276D
随机掩码 50% 维度 (置 0)
masked_encoder: 276 → 256 → GELU → 276
L_recon = MSE(feats_recon, feats_gt)

被掩码维度必须从未掩码的幸存维度推断:
  - 力觉维度被掩 → 视觉必须补充
  - 视觉维度被掩 → 力觉+本体必须补充
```

---

## 6. 损失函数

```
L_fwd   = Huber(p̂_{t+K}, p_{t+K})
L_inv   = Huber(p̂_t, p_t)                        [warmup_epoch=5]
L_task  = 1 - cos(dir_xŷ, dir_xy_gt)             [mask: kd>1e-4 & eng<0.5]
L_eng   = BCE(enĝ, eng_gt)
L_recon = MSE(feats_recon, feats_gt)

L_dyn   = L_fwd + 0.5*L_inv
L_total = w_dyn*L_dyn + w_task*L_task + w_eng*L_eng + w_recon*L_recon
```

---

## 7. 训练超参

```yaml
w_dyn: 1.0          # 动力学
w_task: 15.0        # 方向预测（主导 loss）
w_eng:  10.0        # 接触检测
w_recon: 1.0        # MAE 重建
warmup_inv_cons: 5  # 逆向一致性启用 epoch
early_stop_patience: 15
```

---

## 8. 模型结构

```python
class SymmetricDynamicsEncoder(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14_attn", ...):
        # Backbone (frozen DINOv2 + trainable spatial attention)
        self._backbone, D, self._encode_frame_fn = _build_backbone(...)

        # Per-frame projection
        self.proj_mlp = Sequential(Linear(D→D)→GELU→Linear(D→D))
        # Stereo fusion (mono时None)
        self.fusion_mlp = Sequential(Linear(2D→D)→GELU→Linear(D→D))

        # Force pathway (10 frames × 6D = 60D)
        self.force_mlp = Sequential(LayerNorm(60)→Linear(60→128)→GELU→Linear(128→64))

        # DiffExtract
        self.diff_extract_v = Sequential(Linear(D→128)→GELU→Linear(128→64)→GELU→Linear(64→32))
        self.diff_extract_f = Sequential(Linear(64→64)→GELU→Linear(64→32)→GELU→Linear(32→32))

        # Pathway 1: Visual-Geometry
        self.head_vis_fc1 = Linear(D+32+20, 256)
        self.head_vis_fc2 = Linear(256, 128)
        self.head_task = Sequential(Linear(128+6→128)→GELU→Linear(128→2))  # XY only

        # Pathway 2: Force-Contact
        self.head_force = Sequential(Linear(116→64)→GELU→Linear(64→128))
        self.head_engagement = Sequential(Linear(256→32)→GELU→Linear(32→1)→Sigmoid)

        # Unified dynamics
        self.head_dyn = Sequential(Linear(84→128)→GELU→Linear(128→20))

        # MAE masked autoencoder
        self.masked_encoder = Sequential(Linear(276→256)→GELU→Linear(256→276))
```

---

## 9. forward_pretrain

```python
def forward_pretrain(self, img_left_t, img_right_t, img_left_tk, img_right_tk,
                     ft_t, ft_tk, a, p_t, p_tk, p_fwd):
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
    h_vis = F.gelu(self.head_vis_fc1(cat([v_t, α_v, p_t])))
    z_vis = F.gelu(self.head_vis_fc2(h_vis))                         # (B,128)
    task_pred = self.head_task(cat([z_vis, a]))                       # (B,2) XY dir

    # 4. Pathway 2: Force-Contact (visual + force)
    z_force = F.gelu(self.head_force(cat([f_t, α_f, p_t])))          # (B,128)
    eng_pred = self.head_engagement(cat([z_vis, z_force]))           # (B,1)

    # 5. MAE masked reconstruction
    feats = cat([z_vis, z_force, p_t])                                # (B,276)
    mask = torch.rand(B, 276) < 0.5
    feats_recon = self.masked_encoder(feats * (~mask))               # (B,276)

    # 6. Unified dynamics (forward + inverse)
    p_fwd = self.head_dyn(cat([α_v, α_f, p_t]))
    α_v_inv, α_f_inv = self.diff_extract_v(-Δv), self.diff_extract_f(-Δf)
    p_inv = self.head_dyn(cat([α_v_inv, α_f_inv, p_fwd]))           # p_fwd = ground truth

    return {z_vis, z_force, task_pred, eng_pred, p_fwd, p_inv, feats_recon, ...}
```

---

## 10. DINOv2 Backbone 变体

| 变体 | 编码 | 参数 | 说明 |
|------|------|------|------|
| `dinov2_vits14` | CLS token → 384D | 0 (frozen) | 语义摘要，空间信息少 |
| `dinov2_vits14_patches` | Patch tokens mean pool → 384D | 0 (frozen) | 等权平均，保留空间但稀释 |
| `dinov2_vits14_attn` | Patch tokens + spatial attn → 384D | ~50K | **当前使用**，learnable focus |

```python
# Spatial Attention (50K params, trainable)
class PatchAttnEncoder(nn.Module):
    def __init__(self, dino_model, dim):
        self.dino = dino_model  # frozen
        self.attn = Sequential(Linear(dim→128)→GELU→Linear(128→1))
    def forward(self, x):
        with torch.no_grad():
            patches = self.dino.get_intermediate_layers(x, n=1)[0]  # (B,256,384)
        w = F.softmax(self.attn(patches).squeeze(-1), dim=-1)      # (B,256)
        return (patches * w.unsqueeze(-1)).sum(dim=1)               # (B,384)
```

---

## 11. 视觉增强

训练时在 `sim2real_visual_augment` 中施加：

- 亮度: ±25%, 对比度: ±25%, 饱和度: ±40%
- 50% 概率高斯模糊 (k=3)
- 50% 概率高斯噪声 (σ=0.03)
- **随机平移 ±5px**（模拟相机 target noise）

---

## 12. RL 推理

```python
def forward_features(img_l, img_r, ft, ft_k, prev_action, proprio):
    v_t  = encode_visual(img_l, img_r)
    f_t  = encode_force(ft)
    f_tk = encode_force(ft_k)
    Δv = v_t - self._prev_v_t;  Δf = f_t - f_tk
    self._prev_v_t = v_t.detach()
    α_v = diff_extract_v(Δv);  α_f = diff_extract_f(Δf)

    z_vis   = F.gelu(head_vis_fc2(F.gelu(head_vis_fc1([v_t, α_v, proprio]))))  # 128D
    z_force = F.gelu(head_force([f_t, α_f, proprio]))                            # 128D
    task_norm = head_task([z_vis, prev_action])                                   # 2D XY
    eng_pred  = head_engagement([z_vis, z_force])                                 # 1D
    return z_vis, z_force, task_norm, eng_pred

# Policy obs = [z_vis(128), z_force(128), proprio(20), prev_a(6)] = 282D
```

---

## 13. get_inference_state_dict

```python
exclude = ("head_dyn.",)
# 保留: backbone, proj_mlp, fusion_mlp, force_mlp,
#       diff_extract_v, diff_extract_f,
#       head_vis_fc1, head_vis_fc2, head_force,
#       head_task, head_engagement, masked_encoder
```

---

## 14. 参数统计

| 模块 | 参数 |
|------|------|
| DINOv2 ViT-S (frozen) | 22M |
| Spatial Attention | ~50K |
| ProjMLP + FusionMLP | ~890K |
| ForceMLP | ~25K |
| diff_extract_v/f | ~57K |
| head_vis (436→256→128) | ~146K |
| head_force (116→64→128) | ~24K |
| head_dyn (84→128→20) | ~14K |
| head_task (134→128→2) | ~17K |
| head_engagement (256→32→1) | ~8K |
| masked_encoder (276→256→276) | ~143K |
| **总计 (可训练)** | **~1.21M** |

---

## 15. 实现文件清单

| 文件 | 作用 |
|------|------|
| `scripts/tools/pretrain_dynamics.py` | 预训练模型 + 训练循环 |
| `scripts/tools/pretrain_config.yaml` | 训练配置 |
| `scripts/tools/pretrain_verify.py` | 快速验证脚本 (raw features → XY) |
| `source/.../factory/encoder.py` | RL 推理编码器 |
| `source/.../factory/factory_env.py` | RL 环境 (encoder 集成) |
| `source/.../factory/factory_env_cfg.py` | 环境配置 |
| `source/.../factory/factory_tasks_cfg.py` | 任务配置 (peg/hole scale) |
| `scripts/tools/eval_encoder_features.py` | 编码器评估 |
| `scripts/tools/record_eval_video.py` | 评估视频录制 |
| `scripts/tools/collect_unified_dataset.py` | 数据采集 |
