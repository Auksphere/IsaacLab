#!/usr/bin/env python3
"""
Stage D: Collect unified dataset from IsaacLab Factory PegInsert expert policy.

Configuration is read from a YAML file (see collect_dataset.yaml).
Overridable via CLI: ``--config``, ``--headless``, ``--enable_cameras``.

See HANDOFF.md §9 for the full pipeline.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect unified dataset (Stage D).")
parser.add_argument("--config", type=str,
                    default=str(Path(__file__).resolve().parent / "collect_dataset.yaml"),
                    help="YAML config file (default: collect_dataset.yaml next to this script).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── Load config ──
import yaml
with open(args_cli.config) as f:
    cfg_yaml = yaml.safe_load(f)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports (AFTER AppLauncher) ──
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from PIL import Image
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.direct.factory.factory_env_cfg import (
    FactoryTaskPegInsertCfg,
)

N = cfg_yaml["num_envs"]

# ── Create env ──
print(f"[INFO] Creating vision env ({N} envs, base obs_order for ckpt compat)...")
cfg = parse_env_cfg("Isaac-Factory-PegInsert-Vision-Direct-v0", device="cuda:0", num_envs=N)
base_cfg = FactoryTaskPegInsertCfg()
cfg.obs_order = list(base_cfg.obs_order)
cfg.state_order = list(base_cfg.state_order)
cfg.seed = cfg_yaml["splits"][0]["seed"]  # initial seed; per-split overrides later

res = cfg_yaml.get("camera_resolution", 224)
if res != 224:
    cfg.tiled_camera_left.width = res
    cfg.tiled_camera_left.height = res
    cfg.tiled_camera_right.width = res
    cfg.tiled_camera_right.height = res
    print(f"[INFO] Camera resolution: {res}×{res}")

env = gym.make("Isaac-Factory-PegInsert-Vision-Direct-v0", cfg=cfg)

# ── Load checkpoint ──
ckpt = cfg_yaml["checkpoint"]
print(f"[INFO] Loading checkpoint: {ckpt}")
agent_yaml = (
    Path(__file__).resolve().parents[2] / "source" / "isaaclab_tasks"
    / "isaaclab_tasks" / "direct" / "factory" / "agents" / "rl_games_ppo_cfg.yaml"
)
with open(agent_yaml) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = os.path.abspath(ckpt)
agent_cfg["params"]["config"]["num_actors"] = N
agent_cfg["params"]["seed"] = cfg_yaml["splits"][0]["seed"]

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

# ── Helpers ──
def _apply_dr_preset(dr_cfg, preset: dict):
    """Copy YAML preset values onto DomainRandCfg fields."""
    for key, val in preset.items():
        if hasattr(dr_cfg, key):
            setattr(dr_cfg, key, list(val) if isinstance(val, list) else val)

class StepProfiler:
    _PHASES = ("inference", "physics", "io")
    def __init__(self, interval_s=10.0):
        self._iv = interval_s
        self._last = time.time(); self._n = 0
        self._sums = dict.fromkeys(self._PHASES, 0.0)
    def record(self, inf_ms, phy_ms, io_ms):
        self._n += 1
        self._sums["inference"] += inf_ms
        self._sums["physics"] += phy_ms
        self._sums["io"] += io_ms
        now = time.time()
        if now - self._last < self._iv:
            return
        n = self._n
        if n == 0:
            return
        parts = [f"{p}={self._sums[p]/n:.1f}ms" for p in self._PHASES]
        total_ms = sum(self._sums.values()) / n
        print(f"  [profile] {n} steps, {n/(now-self._last):.1f} Hz "
              f"({total_ms:.1f}ms/step): {' | '.join(parts)}")
        self._last = now; self._n = 0
        for p in self._PHASES:
            self._sums[p] = 0.0

# ── Collection loop (per split) ──
def collect_split(sc):
    """Collect episodes for one split. Returns frame count."""
    name = sc["name"]; num_ep = sc["episodes"]; seed = sc["seed"]
    dr_level = sc["domain_rand"]
    dr_preset = cfg_yaml.get("domain_randomization", {}).get(dr_level, {})

    out_root = Path(cfg_yaml["output_dir"]) / name
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_f = open(out_root / "labels.jsonl", "w")

    _apply_dr_preset(cfg.domain_rand, dr_preset)
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}\n[{name}] episodes={num_ep} seed={seed} dr={dr_level}"
          f"\n{'='*60}")

    ep_ids = torch.zeros(N, dtype=torch.int32)
    ep_frames = torch.zeros(N, dtype=torch.int32)
    success_steps = torch.zeros(N, dtype=torch.int32, device="cuda:0")
    steps_since_reset = torch.zeros(N, dtype=torch.int32, device="cuda:0")
    ep_rewards = torch.zeros(N, device="cuda:0")

    profiler = StepProfiler()
    _first_step = True
    global_ep = 0; total_frames = 0; next_ep_id = 0
    t_start = time.time()
    max_steps = cfg_yaml["max_steps_per_ep"]
    min_ss = cfg_yaml["min_success_steps"]

    # Initial reset
    obs_dict, _ = env.reset()
    for i in range(N):
        ep_ids[i] = next_ep_id; next_ep_id += 1
    player.reset()
    _ = player.get_batch_size(obs_dict["policy"], N)
    if player.is_rnn:
        player.init_rnn()
    obs_tensor = obs_dict["policy"]

    while global_ep < num_ep:
        t0 = time.perf_counter()
        with torch.inference_mode():
            obs_rl = player.obs_to_torch(obs_tensor)
            actions = player.get_action(obs_rl, is_deterministic=True)
        t1 = time.perf_counter()

        obs_dict, rews, terms, truncs, infos = env.step(actions)
        t2 = time.perf_counter()

        env_ = env.unwrapped
        sth = env_.cfg_task.success_threshold
        tn = env_.cfg_task.name

        for i in range(N):
            ep_frames[i] += 1
            steps_since_reset[i] += 1
            if min_ss > 0 and steps_since_reset[i] >= min_ss:
                cs = env_._get_curr_successes(success_threshold=sth, check_rot=(tn == "nut_thread"))
                if cs[i]:
                    success_steps[i] += 1
                else:
                    success_steps[i] = 0
            ep_rewards[i] += rews[i]

            # Images
            fid = f"ep{ep_ids[i].item():04d}_step{ep_frames[i].item():04d}"
            if "camera_left_rgb" in obs_dict:
                lp = img_dir / f"{fid}_L.png"; rp = img_dir / f"{fid}_R.png"
                Image.fromarray(obs_dict["camera_left_rgb"][i].cpu().numpy()).save(lp)
                Image.fromarray(obs_dict["camera_right_rgb"][i].cpu().numpy()).save(rp)
                lr = str(lp.relative_to(out_root)); rr = str(rp.relative_to(out_root))
            else:
                lr = rr = ""

            labels_f.write(json.dumps({
                "episode_id": ep_ids[i].item(), "frame_idx": ep_frames[i].item(),
                "split": name,
                "camera_left": lr, "camera_right": rr,
                "joint_pos": env_.joint_pos[i, 0:7].tolist(),
                "fingertip_pos": env_.fingertip_midpoint_pos[i].tolist(),
                "fingertip_quat": env_.fingertip_midpoint_quat[i].tolist(),
                "ee_linvel": env_.ee_linvel_fd[i].tolist(),
                "ee_angvel": env_.ee_angvel_fd[i].tolist(),
                "force_torque": env_.force_torque[i].tolist(),
                "delta_f": env_.delta_f[i].tolist(),
                "ft_history_ptr": int(env_.ft_history_ptr),
                "contact_force_state": env_.contact_force_state[i].tolist(),
                "engagement_state": float(env_.engagement_state[i, 0].item()),
                "keypoint_dist": float(env_.keypoint_dist[i].item()),
                "action": actions[i].tolist(),
                "is_terminal": bool(terms[i] or truncs[i]),
                "reward": float(rews[i].item()),
            }) + "\n")
            total_frames += 1

        # ── Done check ──
        done = terms | truncs
        early = success_steps >= min_ss if min_ss > 0 else torch.zeros(N, dtype=torch.bool, device="cuda:0")
        finished = done | early
        done_ids = finished.nonzero(as_tuple=False).squeeze(-1)

        if len(done_ids) > 0:
            for i in done_ids.tolist():
                stop = "success" if early[i] or not terms[i] else "timeout"
                print(f"  ep {ep_ids[i].item():04d} frames={ep_frames[i].item()} "
                      f"reward={ep_rewards[i].item():.0f} stop={stop} "
                      f"({global_ep+1}/{num_ep})")
                global_ep += 1
                if global_ep >= num_ep:
                    break
            if global_ep >= num_ep:
                break
            if global_ep < num_ep:
                obs_dict, _ = env.reset()
                player.reset()
                _ = player.get_batch_size(obs_dict["policy"], N)
                if player.is_rnn:
                    player.init_rnn()
                for i in range(N):
                    ep_ids[i] = next_ep_id; next_ep_id += 1
                ep_frames[:] = 0; success_steps[:] = 0
                steps_since_reset[:] = 0; ep_rewards[:] = 0.0

        t3 = time.perf_counter()
        if not _first_step:
            profiler.record((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
        _first_step = False
        obs_tensor = obs_dict["policy"]

    labels_f.close()
    elapsed = time.time() - t_start
    print(f"[{name}] DONE: {global_ep} ep, {total_frames} frames "
          f"in {elapsed:.0f}s ({total_frames/elapsed:.1f} fps) → {out_root}")
    return total_frames

# ── Run ──
grand_total = 0
for sc in cfg_yaml["splits"]:
    grand_total += collect_split(sc)

env.close()
simulation_app.close()
print(f"\n[DONE] All splits: {grand_total} total frames → {cfg_yaml['output_dir']}/")
