"""PPO for the factorized action a = (t, c) with a single critic V(s)."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple


class PPOTrainer:

    def __init__(
        self,
        network: nn.Module,
        lr: float = 3e-4,
        entropy_coef: float = 0.03,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
    ):
        self.network = network
        self.device = device
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

    def standardize_reward(self, makespans: torch.Tensor) -> torch.Tensor:
        """Terminal reward: -C_max standardized within the batch."""
        r = -makespans.detach().float()
        if r.numel() > 1 and r.std() > 1e-8:
            r = (r - r.mean()) / (r.std() + 1e-8)
        return r


    def compute_gae(
        self,
        v: torch.Tensor,
        alive: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GAE with delta_k = r_k + gamma V(s_{k+1}) - V(s_k) and A_k = delta_k + gamma lambda A_{k+1}."""
        T, B = v.shape
        v = v.detach(); alive = alive.detach()

        next_alive = torch.cat([alive[1:], torch.zeros(1, B, device=v.device)], dim=0)
        is_terminal = alive * (1.0 - next_alive)
        r = is_terminal * reward.detach().unsqueeze(0)
        v_next = torch.cat([v[1:], torch.zeros(1, B, device=v.device)], dim=0)

        adv_raw = torch.zeros(T, B, device=v.device)
        gae = torch.zeros(B, device=v.device)
        for k in range(T - 1, -1, -1):
            delta = r[k] + self.gamma * v_next[k] * next_alive[k] - v[k]
            delta = delta * alive[k]
            gae = delta + self.gamma * self.gae_lambda * next_alive[k] * gae
            gae = gae * alive[k]
            adv_raw[k] = gae

        v_target = adv_raw + v
        adv = self._normalize(adv_raw, alive)
        return adv, v_target, adv_raw

    @staticmethod
    def _normalize(adv: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """Normalize the advantages over the active steps; inactive steps stay 0."""
        mask = alive > 0.5
        if mask.sum() > 1:
            vals = adv[mask]
            adv = (adv - vals.mean()) / (vals.std() + 1e-8)
            adv = adv * alive
        return adv.detach()

    def step_losses(
        self,
        new_logp_t, old_logp_t, ent_t,
        new_logp_p, old_logp_p, ent_p,
        adv,
        v_pred, v_target,
        active, n_dec, N,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """PPO loss of one decision step."""
        eps = self.clip_eps
        adv = adv.detach()

        new_logp = new_logp_t + new_logp_p
        old_logp = (old_logp_t + old_logp_p).detach()
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        pg = -(torch.min(surr1, surr2) * active).sum() / N

        v_err = (v_pred - v_target.detach()) ** 2
        vloss = self.value_coef * (v_err * active).sum() / N

        ent = ((ent_t + ent_p) * active).sum() / N

        loss = pg + vloss - self.entropy_coef * ent
        info = {"pg": pg.item(), "vloss": vloss.item(), "ent": ent.item()}
        return loss, info

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_and_step(self) -> float:
        """Clip the gradients accumulated by ppo_replay_epoch and update the parameters."""
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return grad_norm if isinstance(grad_norm, float) else grad_norm.item()
