#!/usr/bin/env python3
"""
Record evaluation video with dual camera views + encoder task_pred overlay.

Loads a trained RL policy checkpoint, runs deterministic rollouts in the
encoder environment, captures left/right TiledCamera frames, and compiles
side-by-side MP4 videos with task prediction, engagement, and keypoint
distance overlays.

Usage (inside container):
  # With encoder:
  ./isaaclab.sh -p scripts/tools/record_eval_video.py \
    --ckpt /workspace/isaaclab/logs/rl_games/Factory/.../nn/last_Factory_*.pth \
    --encoder-ckpt /workspace/isaaclab/output/pretrain/pretrained_encoder_best.pt \
    --episodes 5 --output-dir /workspace/isaaclab/output/videos --no-rand

  # Sanity check (no encoder, privileged-state policy):
  ./isaaclab.sh -p scripts/tools/record_eval_video.py \
    --ckpt /workspace/isaaclab/logs/rl_games/Factory/.../nn/last_Factory_*.pth \
    --episodes 5 --output-dir /workspace/isaaclab/output/videos
"""
import argparse, math, os, sys, time
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record eval videos with dual cameras.")
parser.add_argument("--ckpt", type=str, required=True,
                    help="RL Games policy checkpoint (.pth).")
parser.add_argument("--encoder-ckpt", type=str, default="",
                    help="Encoder checkpoint (.pt). Omit for debug_state_policy mode.")
parser.add_argument("--backbone", type=str, default=None,
                    help="Encoder backbone (default: use config class default).")
parser.add_argument("--episodes", type=int, default=5,
                    help="Number of episodes to record (default: 5).")
parser.add_argument("--max-steps", type=int, default=200,
                    help="Max steps per episode (default: 200).")
parser.add_argument("--output-dir", type=str,
                    default="/workspace/isaaclab/output/videos",
                    help="Output directory for videos (default: /workspace/isaaclab/output/videos).")
parser.add_argument("--fps", type=int, default=15,
                    help="Video frame rate (default: 15).")
parser.add_argument("--no-video", action="store_true",
                    help="Save PNG frames only, skip MP4 compilation.")
parser.add_argument("--no-rand", action="store_true",
                    help="Disable domain randomization for deterministic eval.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This script always needs cameras — force the flag
args_cli.enable_cameras = True
# Headless is required for server/Docker — no GUI available
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All Isaac Lab imports AFTER AppLauncher ──
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertEncoderCfg
from isaaclab_tasks.direct.factory.encoder import VisualForceEncoder, preprocess_rgb

# ── Output setup ──
output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ── Create env ──
N = 1
device = "cuda:0"
has_encoder = args_cli.encoder_ckpt and args_cli.encoder_ckpt != ""

print(f"[INFO] Creating env (N={N}, encoder={'YES' if has_encoder else 'debug_state_policy'})...")
cfg = parse_env_cfg("Isaac-Factory-PegInsert-Encoder-Direct-v0", device=device, num_envs=N)

if has_encoder:
    cfg.encoder_checkpoint = args_cli.encoder_ckpt
    if args_cli.backbone is not None:
        cfg.encoder_backbone = args_cli.backbone
    cfg.encoder_debug_state_policy = False
else:
    # Privileged-state policy + cameras kept on for recording
    cfg.encoder_debug_state_policy = True
    cfg.encoder_debug_keep_cameras = True

cfg.seed = 42
cfg.episode_length_s = 15.0  # eval only: more time for 16mm insertion
if args_cli.no_rand:
    # Disable all domain randomization
    dr = cfg.domain_rand
    dr.material_full_replace = True
    dr.dome_light_intensity = [1.0, 1.0]
    dr.key_light_intensity = [1.0, 1.0]
    dr.camera_pos_noise = [0.0, 0.0, 0.0]
    dr.camera_target_noise = [0.0, 0.0, 0.0]
    dr.held_mass_scale = [1.0, 1.0]
    dr.fixed_mass_scale = [1.0, 1.0]
    dr.held_friction = [1.0, 1.0]
    dr.fixed_friction = [1.0, 1.0]
    dr.joint_damping_scale = [1.0, 1.0]
    dr.joint_stiffness_scale = [1.0, 1.0]
    dr.peg_scale = [1.0, 1.0]
    dr.hole_scale = [1.0, 1.0]
    print("[INFO] Domain randomization DISABLED")
env = gym.make("Isaac-Factory-PegInsert-Encoder-Direct-v0", cfg=cfg)
env_ = env.unwrapped  # direct access to FactoryEnv internals

# ── Load encoder (if provided) ──
encoder = None
if has_encoder:
    print(f"[INFO] Loading encoder: {args_cli.encoder_ckpt}")
    backbone_enc = args_cli.backbone or cfg.encoder_backbone
    encoder = VisualForceEncoder.from_checkpoint(args_cli.encoder_ckpt, backbone_enc, device)
    encoder.eval()
    print(f"  feature_dim={encoder.feature_dim}")

# ── Load RL policy checkpoint ──
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

THIS_DIR = Path(__file__).resolve().parent
print(f"[INFO] Loading RL policy: {args_cli.ckpt}")

agent_yaml = (
    THIS_DIR.parent.parent / "source" / "isaaclab_tasks"
    / "isaaclab_tasks" / "direct" / "factory" / "agents" / "rl_games_ppo_encoder_cfg.yaml"
)
with open(agent_yaml, encoding="utf-8") as f:
    agent_cfg = yaml.safe_load(f)

agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = os.path.abspath(args_cli.ckpt)
agent_cfg["params"]["config"]["num_actors"] = N
agent_cfg["params"]["seed"] = 42

clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

venv = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)
vecenv.register(
    "IsaacRlgWrapper",
    lambda cn, na, **kw: type('X', (), {
        '__init__': lambda s: None,
        'get_number_of_agents': lambda s: N,
    })(),
)
env_configurations.register(
    "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: venv}
)

runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
runner.reset()
player = runner.create_player()
player.restore(agent_cfg["params"]["load_path"])
player.has_batch_dimension = True
print("[INFO] RL policy loaded.")

# ── Find a font for overlay text ──
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
font = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 14)
        break
if font is None:
    font = ImageFont.load_default()

# ── Helper: build overlayed side-by-side frame ──
def make_frame(left_rgb, right_rgb, step, dir_pred, dir_gt, cos_sim,
               eng_pred, keypoint_dist, reward):
    """Horizontally stack left/right camera frames with text overlay bar."""
    left_img = Image.fromarray(left_rgb)
    right_img = Image.fromarray(right_rgb)
    h, w = left_img.height, left_img.width

    # Side-by-side
    combined = Image.new("RGB", (w * 2, h + 60), color=(20, 20, 20))
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (w, 0))

    # Text overlay bar at the bottom
    draw = ImageDraw.Draw(combined)
    y0 = h + 4
    lines = [
        f"Step:{step:3d}  kd:{keypoint_dist:.4f}  rew:{reward:.3f}  "
        f"eng_pred:{eng_pred:.2f}  eng_gt:{env_.engagement_state[0].item():.0f}",
        f"xy pred=({dir_pred[0]:.3f},{dir_pred[1]:.3f})  "
        f"gt=({dir_gt[0]:.3f},{dir_gt[1]:.3f})  cos={cos_sim:.3f}",
    ]
    for i, line in enumerate(lines):
        draw.text((6, y0 + i * 18), line, fill=(200, 200, 200), font=font)
    return combined


# ── Evaluation loop ──
results = []
ckpt_stem = Path(args_cli.ckpt).stem[:40]

for ep in range(args_cli.episodes):
    print(f"\n{'='*50}\nEpisode {ep}/{args_cli.episodes}")
    # Per-episode log file (same name as video)
    ep_log_path = output_dir / f"ep_{ep:03d}_{ckpt_stem}.txt"
    ep_log = open(ep_log_path, "w")
    ep_log.write(f"# Episode {ep}\n")
    ep_log.write(f"# step kd rew cos eng_pred eng_gt dir_pred_x dir_pred_y dir_gt_x dir_gt_y\n")

    obs_dict, _ = env.reset()
    player.reset()
    _ = player.get_batch_size(obs_dict["policy"], N)
    if player.is_rnn:
        player.init_rnn()
    prev_action = torch.zeros(N, 6, device=device)

    frames = []
    ep_reward = 0.0
    ep_min_kd = 1.0
    success = False
    ep_dir_pred = []
    ep_dir_gt = []

    for step in range(args_cli.max_steps):
        # ── Encoder task_pred evaluation ──
        if encoder is not None:
            img_l = preprocess_rgb(env_._tiled_camera_left.data.output["rgb"])
            img_r = preprocess_rgb(env_._tiled_camera_right.data.output["rgb"])
            ft_3f   = env_.ft_ring[:, -10:, :].reshape(N, 60)
            ft_3f_k = env_.ft_ring[:, :10, :].reshape(N, 60)
            proprio_20 = torch.cat([
                env_.joint_pos[:, 0:7], env_.fingertip_midpoint_pos,
                env_.fingertip_midpoint_quat, env_.ee_linvel_fd, env_.ee_angvel_fd,
            ], dim=-1)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                dir_pred, eng_pred_val = encoder.predict_task(
                    img_l, ft_3f, img_r=img_r,
                    prev_action=prev_action, proprio=proprio_20,
                    ft_3frame_k=ft_3f_k)
            dp = dir_pred.float().cpu().numpy().squeeze()
            eng_np = float(eng_pred_val.float().cpu().item())
            # Ground-truth XY direction (world frame, skip Z)
            d_world = env_.fingertip_midpoint_pos[0, :2] - env_.fixed_pos_obs_frame[0, :2]
            dg = (d_world / (d_world.norm() + 1e-8)).cpu().numpy()
            cos_sim = float(np.dot(dp, dg))
        else:
            dp = np.zeros(3)
            dg = np.zeros(3)
            cos_sim = 0.0
            eng_np = 0.0

        kd = env_.keypoint_dist[0].item()
        ep_dir_pred.append(dp)
        ep_dir_gt.append(dg)

        # ── Capture camera frames ──
        if "camera_left_rgb" in obs_dict:
            left_rgb  = obs_dict["camera_left_rgb"][0].cpu().numpy()
            right_rgb = obs_dict["camera_right_rgb"][0].cpu().numpy()
            frame_img = make_frame(left_rgb, right_rgb, step, dp, dg, cos_sim,
                                   eng_np, kd, ep_reward)
            frames.append(frame_img)

        # ── Get action from policy ──
        obs_tensor = obs_dict["policy"]
        with torch.inference_mode():
            actions = player.get_action(obs_tensor, is_deterministic=True)

        obs_dict, rews, terms, truncs, infos = env.step(actions)
        prev_action = actions.clone()

        ep_reward += float(rews[0].item())
        ep_min_kd = min(ep_min_kd, kd)

        eng_gt_val = env_.engagement_state[0].item()
        if True:  # every step
            print(f"  step={step:3d} kd={kd:.4f} rew={rews[0].item():.3f} "
                  f"xy=({dp[0]:.2f},{dp[1]:.2f}) cos={cos_sim:.3f} "
                  f"eng={eng_np:.2f}/{eng_gt_val:.0f}")
            ep_log.write(f"{step} {kd:.4f} {rews[0].item():.3f} {cos_sim:.3f} "
                         f"{eng_np:.2f} {eng_gt_val:.0f} "
                         f"{dp[0]:.4f} {dp[1]:.4f} {dg[0]:.4f} {dg[1]:.4f}\n")

        done = terms.any() or truncs.any()
        if done or ep_min_kd < 0.002:
            success = ep_min_kd < 0.002
            if success:
                print(f"  → success at step {step}!")
            break

    # ── Episode summary ──
    if len(ep_dir_pred) > 0:
        dp_arr = np.array(ep_dir_pred)
        dg_arr = np.array(ep_dir_gt)
        cos_sims = [float(np.dot(dp_arr[j], dg_arr[j])) for j in range(len(dp_arr))]
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    else:
        mean_cos = 0.0

    results.append({
        "ep": ep, "frames": step + 1, "success": success,
        "min_kd": ep_min_kd, "reward": ep_reward,
        "cos_sim": mean_cos,
    })
    print(f"Ep {ep:2d}: succ={success} min_kd={ep_min_kd:.4f} "
          f"cum_rew={ep_reward:.1f} cos={mean_cos:.3f}")
    ep_log.write(f"# succ={success} min_kd={ep_min_kd:.4f} cum_rew={ep_reward:.1f} cos={mean_cos:.3f}\n")
    ep_log.close()

    # ── Compile video ──
    if not args_cli.no_video and len(frames) > 0:
        video_path = output_dir / f"ep_{ep:03d}_{ckpt_stem}.mp4"
        try:
            import imageio
            writer = imageio.get_writer(str(video_path), fps=args_cli.fps,
                                        codec="libx264", quality=8)
            for f_img in frames:
                writer.append_data(np.array(f_img))
            writer.close()
            print(f"  Video saved: {video_path}")
        except ImportError:
            # Fallback: save as PNGs
            png_dir = output_dir / f"ep_{ep:03d}_frames"
            png_dir.mkdir(exist_ok=True)
            for i, f_img in enumerate(frames):
                f_img.save(png_dir / f"step_{i:04d}.png")
            print(f"  Frames saved to: {png_dir} (imageio not available)")

# ── Summary ──
env.close()
summary_path = output_dir / "summary.txt"
with open(summary_path, "w") as sf:
    sf.write(f"Checkpoint: {args_cli.ckpt}\n")
    sf.write(f"Encoder:    {args_cli.encoder_ckpt or '(debug_state_policy)'}\n")
    sf.write(f"Episodes:   {len(results)}\n\n")
    sf.write(f"{'ep':>4s} {'succ':>5s} {'min_kd':>8s} {'rew':>8s} "
             f"{'frames':>7s} {'cos':>7s}\n")
    for r in results:
        sf.write(f"{r['ep']:4d} {str(r['success']):>5s} {r['min_kd']:8.4f} "
                 f"{r['reward']:8.1f} {r['frames']:7d} {r['cos_sim']:7.3f}\n")

    succ_rate = sum(r["success"] for r in results) / len(results)
    coss = [r["cos_sim"] for r in results if r["cos_sim"] != 0]
    sf.write(f"\nSuccess: {succ_rate*100:.0f}%\n")
    if coss:
        sf.write(f"Dir cos: {np.mean(coss):.3f} +- {np.std(coss):.3f}\n")

print(f"\nDone. Summary: {summary_path}")
simulation_app.close()
