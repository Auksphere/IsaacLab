"""Demo buffer for LfD: load expert trajectories from HDF5 for PPO mixing.

obs=(T,285D), reward derived from ||task_pred|| (mm), done=dist<2mm.
"""

import h5py
import numpy as np
import torch


class DemoBuffer:
    def __init__(self, data_path: str, device: str = "cuda:0"):
        self.device = device
        self.trajectories = []

        hf = h5py.File(data_path, "r")
        for demo_name in sorted(hf["data"].keys()):
            demo = hf["data"][demo_name]
            obs = demo["obs/flat_obs"][:]   # (T, 285)
            act = demo["actions"][:]          # (T, 6)
            T = obs.shape[0]
            if T < 8:
                continue

            # Detect obs dim: 285 = z(256)+task_pred(3)+proprio(20)+prev_a(6)
            #                   282 = z(256)+proprio(20)+prev_a(6)
            obs_dim = obs.shape[1]
            has_task_pred = obs_dim == 285

            # Extract prev_actions from obs (last 6 dims)
            prev_actions = obs[:, -6:].copy()  # (T, 6)

            # Compute reward from task_pred if available, else use zero
            if has_task_pred:
                task_pred = obs[:, 256:259]              # (T, 3) mm
            else:
                task_pred = np.zeros((T, 3))             # no task_pred in 282D
            dist_mm = np.linalg.norm(task_pred, axis=-1)  # (T,) mm
            dist_m = dist_mm / 1000.0                 # m (matches env units)
            # Use same squashing function as env: 1/(exp(-a*x) + b + exp(a*x))
            def squashing_fn(x, a, b):
                return 1.0 / (np.exp(-a * x) + b + np.exp(a * x))
            rew_kp = squashing_fn(dist_m, 50, 2)     # coarse keypoint reward
            # Action penalty (matches env)
            action_norm = np.linalg.norm(act, axis=-1)  # ||a||
            action_grad = np.linalg.norm(np.diff(act, axis=0, prepend=act[:1]), axis=-1)  # ||a - a_prev||
            rew = rew_kp - 0.0 * action_norm - 0.0 * action_grad  # match env penalty scales

            dones = np.zeros(T, dtype=bool)
            dones[-1] = True
            for t in range(T):
                if dist_mm[t] < 2.0:
                    dones[t] = True
                    rew[t] += 1.0  # success bonus
                    break

            self.trajectories.append({
                "obs": torch.from_numpy(obs.astype(np.float32)).to(device),
                "actions": torch.from_numpy(act.astype(np.float32)).to(device),
                "prev_actions": torch.from_numpy(prev_actions.astype(np.float32)).to(device),
                "rewards": torch.from_numpy(rew.astype(np.float32)).to(device),
                "dones": torch.from_numpy(dones).to(device),
            })
        hf.close()

        print(f"[DemoBuffer] Loaded {len(self.trajectories)} trajectories "
              f"({sum(t['obs'].shape[0] for t in self.trajectories)} frames)")

    def sample(self, n_traj: int):
        """Sample n_traj random trajectories. Returns list of dicts."""
        idx = torch.randperm(len(self.trajectories))[:n_traj]
        return [self.trajectories[i] for i in idx.tolist()]
