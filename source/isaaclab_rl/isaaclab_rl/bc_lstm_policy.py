"""BC LSTM policy loaded from robomimic checkpoint for residual RL.

Architecture: obs → LSTM(obs_dim→1024, 2) → MLP(1024→512→128→64) → action(6)
Matches RL Games actor structure (before_mlp=True).
"""

import torch
import torch.nn as nn


class BCLSTMPolicy(nn.Module):
    """Frozen BC LSTM policy for batched inference in residual RL."""

    def __init__(self, ckpt_path: str, device: str, num_envs: int = 128):
        super().__init__()
        self.device = device
        self.num_envs = num_envs

        # Build arch: obs → LSTM(285→1024, 2) → MLP(1024→512→128→64) → action(6)
        self.rnn = nn.LSTM(285, 1024, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.action_head = nn.Linear(64, 6)

        # Load weights from robomimic checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        rob_state = ckpt['model']
        bc_state = self._map_keys(rob_state)
        self.load_state_dict(bc_state, strict=False)
        n_loaded = sum(1 for k in bc_state if k in self.state_dict())
        print(f"[BC-LSTM] Loaded {n_loaded}/{len(self.state_dict())} weights from BC checkpoint")

        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        self.eval()

        # RNN hidden state: (num_layers, num_envs, hidden_dim)
        self.num_layers = 2
        self.hidden_dim = 1024
        self._hidden_h = torch.zeros(self.num_layers, num_envs, self.hidden_dim, device=device)
        self._hidden_c = torch.zeros(self.num_layers, num_envs, self.hidden_dim, device=device)

    def _map_keys(self, rob_state):
        """Map robomimic RNN-first architecture keys to this module."""
        key_map = {}

        # RNN keys: policy.nets.rnn.nets.* → rnn.*
        rob_prefix = 'policy.nets.rnn.nets.'
        for rob_key in rob_state:
            if rob_key.startswith(rob_prefix):
                # weight_ih_l0 → weight_ih_l0 (same name, nn.LSTM uses same convention)
                suffix = rob_key[len(rob_prefix):]
                key_map[rob_key] = f'rnn.{suffix}'

        # MLP: per_step_net.0._model.N.* → mlp.N.* (same Sequential index)
        for rob_key in rob_state:
            if 'per_step_net.0._model' in rob_key:
                new_key = rob_key.replace('policy.nets.rnn.per_step_net.0._model.', 'mlp.')
                key_map[rob_key] = new_key

        # Action head: per_step_net.1.nets.action.* → action_head.*
        for rob_key in rob_state:
            if 'per_step_net.1.nets.action' in rob_key:
                new_key = rob_key.replace('policy.nets.rnn.per_step_net.1.nets.action.', 'action_head.')
                key_map[rob_key] = new_key

        bc_state = {}
        for rob_key, new_key in key_map.items():
            if rob_key in rob_state:
                bc_state[new_key] = rob_state[rob_key]

        return bc_state

    def reset_hidden(self, env_ids=None):
        """Reset LSTM hidden state for given envs (all if None)."""
        if env_ids is None:
            self._hidden_h.zero_()
            self._hidden_c.zero_()
        else:
            self._hidden_h[:, env_ids] = 0.0
            self._hidden_c[:, env_ids] = 0.0

    @torch.no_grad()
    def forward(self, obs):
        """obs: (num_envs, 285), returns action (num_envs, 6)."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (N, 1, 285) for LSTM

        out, (h, c) = self.rnn(obs, (self._hidden_h, self._hidden_c))
        self._hidden_h, self._hidden_c = h.detach(), c.detach()
        out = out.squeeze(1)             # (N, 1024)
        out = self.mlp(out)              # (N, 64)
        out = self.action_head(out)      # (N, 6)
        out = torch.tanh(out)            # robomimic uses tanh → [-1, 1]
        return out
