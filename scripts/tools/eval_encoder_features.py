#!/usr/bin/env python3
"""Evaluate v5 encoder: task_pred vs held_pos_rel_fixed.

Based on collect_unified_dataset.py (known-working expert pipeline).
Adds encoder loading + feature quality metrics on top.
"""

import argparse, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--encoder-ckpt", type=str, default="")
parser.add_argument("--backbone", type=str, default="dinov2_vits14_attn")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

THIS_DIR = Path(__file__).resolve().parent
# NOTE: sys.path manipulation here can break Isaac Sim module resolution.
# Import encoder lazily inside main() if needed.

N = 1  # single env

# ── Create env (EXACT same flow as collect_unified_dataset.py) ──
print(f"[INFO] Creating encoder debug env ({N} envs, 43D privileged obs, cameras ON)...")
cfg = parse_env_cfg("Isaac-Factory-PegInsert-Encoder-Direct-v0", device="cuda:0", num_envs=N)
cfg.encoder_debug_state_policy = True
cfg.encoder_debug_keep_cameras = True
cfg.encoder_checkpoint = ""
cfg.seed = 42
env = gym.make("Isaac-Factory-PegInsert-Encoder-Direct-v0", cfg=cfg)

# ── Load encoder (lazy import to avoid sys.path pollution) ──
encoder = None
if args_cli.encoder_ckpt and args_cli.encoder_ckpt != "":
    sys.path.insert(0, str(THIS_DIR.parent.parent / "source" / "isaaclab_tasks"
                           / "isaaclab_tasks" / "direct" / "factory"))
    from encoder import VisualForceEncoder, preprocess_rgb
    print(f"[INFO] Loading encoder: {args_cli.encoder_ckpt}")
    encoder = VisualForceEncoder.from_checkpoint(args_cli.encoder_ckpt, args_cli.backbone, "cuda:0")
    encoder.eval()
    print(f"  feature_dim={encoder.feature_dim}")
else:
    print("[INFO] No encoder — pure expert test mode")

# Tee output to eval_log.txt in the checkpoint directory
import builtins
log_dir = Path(args_cli.encoder_ckpt).resolve().parent if args_cli.encoder_ckpt else Path(".")
log_file = open(log_dir / "eval_log.txt", "a", buffering=1, encoding="utf-8")
_orig_print = builtins.print
def _tee_print(*a, **kw):
    import io
    _orig_print(*a, **kw)
    buf = io.StringIO()
    _orig_print(*a, file=buf, **kw)
    log_file.write(buf.getvalue()); log_file.flush()
builtins.print = _tee_print

# ── Load expert policy (EXACT same as collect) ──
ckpt_path = "/workspace/isaaclab/logs/rl_games/Factory/sanity_check/nn/last_Factory_ep_200_rew_389.05432.pth"
print(f"[INFO] Loading checkpoint: {ckpt_path}")
agent_yaml = (
    THIS_DIR.parent.parent / "source" / "isaaclab_tasks"
    / "isaaclab_tasks" / "direct" / "factory" / "agents" / "rl_games_ppo_cfg.yaml"
)
import yaml
with open(agent_yaml) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = os.path.abspath(ckpt_path)
agent_cfg["params"]["config"]["num_actors"] = N
agent_cfg["params"]["seed"] = 42

clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
venv = RlGamesVecEnvWrapper(env, "cuda:0", clip_obs, clip_actions)
vecenv.register("IsaacRlgWrapper",
    lambda cn, na, **kw: type('X', (), {
        '__init__': lambda s: None, 'get_number_of_agents': lambda s: N,
    })(),
)
env_configurations.register("rlgpu",
    {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: venv})
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
runner.reset()
player = runner.create_player()
player.restore(agent_cfg["params"]["load_path"])
player.has_batch_dimension = True
print("[INFO] Checkpoint loaded.")

# ── Evaluation loop (EXACT same loop structure as collect) ──
results = []
all_task_pred = []
all_held_rel = []

# Initial reset + warmup
obs_dict, _ = env.reset()
zero_action = torch.zeros(N, env.action_space.shape[-1], device="cuda:0")
obs_dict, _, _, _, _ = env.step(zero_action)
player.reset()
_ = player.get_batch_size(obs_dict["policy"], N)
if player.is_rnn:
    player.init_rnn()
obs_tensor = obs_dict["policy"]

prev_action = zero_action.clone()

global_ep = 0
ep_frames = torch.zeros(N, dtype=torch.int32)
ep_rewards = torch.zeros(N, device="cuda:0")
ep_min_kd = 1.0   # track min kd during episode (env auto-resets on done)
ep_data = defaultdict(list)

while global_ep < args_cli.episodes:
    with torch.inference_mode():
        obs_rl = player.obs_to_torch(obs_tensor)
        actions = player.get_action(obs_rl, is_deterministic=True)

    # ── Encoder evaluation (parallel to expert) ──
    if encoder is not None:
        env_ = env.unwrapped
        img_l = preprocess_rgb(env_._tiled_camera_left.data.output["rgb"])
        img_r = preprocess_rgb(env_._tiled_camera_right.data.output["rgb"])
        ft_3f = env_.ft_ring[:, -10:, :].reshape(N, 60)     # 10-frame window
        ft_3f_k = env_.ft_ring[:, :10, :].reshape(N, 60)   # K steps ago
        proprio_20 = torch.cat([
            env_.joint_pos[:, 0:7], env_.fingertip_midpoint_pos,
            env_.fingertip_midpoint_quat, env_.ee_linvel_fd, env_.ee_angvel_fd,
        ], dim=-1)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            dir_pred, eng_pred = encoder.predict_task(img_l, ft_3f, img_r=img_r,
                                                          prev_action=prev_action, proprio=proprio_20,
                                                          ft_3frame_k=ft_3f_k)
        dir_np = dir_pred.float().cpu().numpy().squeeze()
        eng_np = eng_pred.float().cpu().item()
        # Ground-truth XY direction (world frame XY, skip Z)
        d_world = env_.fingertip_midpoint_pos[0, :2] - env_.fixed_pos_obs_frame[0, :2]
        dir_gt_np = (d_world / (d_world.norm() + 1e-8)).cpu().numpy()
        ep_data["dir_pred"].append(dir_np)
        ep_data["dir_gt"].append(dir_gt_np)
        ep_data["eng_pred"].append(eng_np)

    obs_dict, rews, terms, truncs, infos = env.step(actions)
    prev_action = actions.clone()

    for i in range(N):
        ep_frames[i] += 1
        ep_rewards[i] += rews[i]

    # Track kd during episode (env auto-resets on done, making post-hoc kd invalid)
    env_ = env.unwrapped
    _kd = env_.keypoint_dist[0].item()
    ep_min_kd = min(ep_min_kd, _kd)
    step_n = ep_frames[0].item()
    if step_n <= 25 or step_n % 25 == 0:
        tp = ep_data["dir_pred"][-1] if ep_data["dir_pred"] else [0,0]
        tg = ep_data["dir_gt"][-1] if ep_data["dir_gt"] else [0,0]
        cos = float(np.dot(tp, tg))
        eng = ep_data["eng_pred"][-1] if ep_data["eng_pred"] else 0.0
        eng_gt = env_.engagement_state[0].item()
        print(f"  step={step_n:3d} kd={_kd:.4f} min_kd={ep_min_kd:.4f} "
              f"rew={rews[0].item():.3f} cum_rew={ep_rewards[0].item():.1f} "
              f"| xy=({tp[0]:.2f},{tp[1]:.2f}) gt=({tg[0]:.2f},{tg[1]:.2f}) "
              f"cos={cos:.3f} eng={eng:.2f}/{eng_gt:.0f}")

    obs_tensor = obs_dict["policy"]

    # ── Done check ──
    done = terms | truncs
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)

    if len(done_ids) > 0:
        for i in done_ids.tolist():
            success = ep_min_kd < 0.002
            print(f"  EP DONE: succ={success} min_kd={ep_min_kd:.4f} cum_rew={ep_rewards[i].item():.1f}")

            # Compute metrics if encoder was used
            if len(ep_data["dir_pred"]) > 0:
                dir_preds = np.array(ep_data["dir_pred"])
                dir_gts = np.array(ep_data["dir_gt"])
                cos_sims = [float(np.dot(dir_preds[j], dir_gts[j]))
                            for j in range(len(dir_preds))]
                mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
                frames = len(dir_preds)
            else:
                mean_cos = 0.0
                frames = 0

            results.append({
                "ep": global_ep, "frames": frames or ep_frames[i].item(),
                "success": success, "min_kd": ep_min_kd,
                "cos_sim": mean_cos,
            })

            print(f"Ep {global_ep:2d}: succ={success} min_kd={ep_min_kd:.4f} "
                  f"cos={mean_cos:.3f} frames={frames}")

            global_ep += 1
            if global_ep >= args_cli.episodes:
                break

        if global_ep >= args_cli.episodes:
            break

        if global_ep < args_cli.episodes:
            obs_dict, _ = env.reset()
            player.reset()
            _ = player.get_batch_size(obs_dict["policy"], N)
            if player.is_rnn:
                player.init_rnn()
            ep_frames[:] = 0
            ep_rewards[:] = 0.0
            ep_min_kd = 1.0
            ep_data = defaultdict(list)
            prev_action = torch.zeros(N, 6, device="cuda:0")
            obs_tensor = obs_dict["policy"]

env.close()

# ── Summary ──
if len(results) > 0:
    succ_rate = sum(r["success"] for r in results) / len(results)
    coses = [r["cos_sim"] for r in results if r["cos_sim"] != 0]
    print(f"\n{'='*60}")
    print(f"Summary ({len(results)} episodes, {succ_rate*100:.0f}% success):")
    if coses:
        print(f"  Direction cosine:          {np.mean(coses):.3f} ± {np.std(coses):.3f}")

simulation_app.close()
