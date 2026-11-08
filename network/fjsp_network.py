import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional

from .hgnn import HGNNEncoder


def normalize_feats(
    raw_opes: torch.Tensor,
    raw_mas: torch.Tensor,
    proc_time: torch.Tensor,
    ope_valid_mask: torch.Tensor,
    eps: float = 1e-5,
) -> tuple:
    """Instance-wise z-score normalization of the node features and processing times.

    Operation features are normalized over the valid operations, machine features over the machines, and processing
    times over the existing operation–machine edges (non-edges stay 0).
    """
    batch_size = raw_opes.size(0)
    num_opes_max = raw_opes.size(1)
    device = raw_opes.device

    ope_mask = ope_valid_mask.unsqueeze(2).float()

    masked_opes = raw_opes * ope_mask
    sum_opes = masked_opes.sum(dim=1, keepdim=True)
    count_opes = ope_mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean_opes = sum_opes / count_opes

    diff_opes = (raw_opes - mean_opes) * ope_mask
    var_opes = (diff_opes ** 2).sum(dim=1, keepdim=True) / count_opes
    std_opes = torch.sqrt(var_opes + eps)

    norm_opes = (raw_opes - mean_opes) / (std_opes + eps)

    mean_mas = raw_mas.mean(dim=1, keepdim=True)
    std_mas = raw_mas.std(dim=1, keepdim=True)
    norm_mas = (raw_mas - mean_mas) / (std_mas + eps)

    proc_time_norm = proc_time.clone()
    proc_mask = (proc_time > 0).float()

    masked_proc = proc_time * proc_mask
    sum_proc = masked_proc.sum(dim=(1, 2), keepdim=True)
    count_proc = proc_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
    mean_proc = sum_proc / count_proc

    diff_proc = (proc_time - mean_proc) * proc_mask
    var_proc = (diff_proc ** 2).sum(dim=(1, 2), keepdim=True) / count_proc
    std_proc = torch.sqrt(var_proc + eps)

    proc_time_norm = torch.where(
        proc_mask.bool(),
        (proc_time - mean_proc) / (std_proc + eps),
        torch.zeros_like(proc_time)
    )

    return norm_opes, norm_mas, proc_time_norm


class MLPActor(nn.Module):
    """MLP with LayerNorm and tanh activations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = torch.tanh(x)
            x = self.dropouts[i](x)
        return self.output_layer(x)


class PairSelfAttention(nn.Module):
    """Self-attention over the candidate operation–machine pairs."""

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(embed_dim),
            }))

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            attn_out, _ = layer['attn'](
                x, x, x, key_padding_mask=key_padding_mask,
            )
            x = layer['norm1'](x + attn_out)
            x = layer['norm2'](x + layer['ffn'](x))
        return x


class PairAgent(nn.Module):
    """Pairing head pi_P(c|s,t) (Section 4.4.3): pair embedding, pair self-attention and MLP scorer.

    A pair embedding concatenates the embedding of the next operation of the job, the machine embedding and the pair
    features (Eq. (17)); self-attention is restricted to the feasible pairs (Eq. (18)); the scorer also sees the
    pooled state embedding (Eq. (19)).
    """

    def __init__(
        self,
        ope_embed_dim: int,
        ma_embed_dim: int,
        pair_private_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        attn_embed_dim: int = 128,
        attn_heads: int = 4,
        attn_layers: int = 2,
    ):
        super().__init__()
        pair_init_dim = ope_embed_dim + ma_embed_dim + pair_private_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_init_dim, attn_embed_dim),
            nn.LayerNorm(attn_embed_dim),
            nn.ReLU(),
        )

        self.pair_self_attn = PairSelfAttention(
            attn_embed_dim, attn_heads, attn_layers, dropout,
        )

        actor_input_dim = (
            attn_embed_dim
            + ope_embed_dim * 2
            + ma_embed_dim * 2
        )
        self.actor = MLPActor(actor_input_dim, hidden_dim, 1, num_layers, dropout)

    def forward(
        self,
        h_ops: torch.Tensor,
        h_mas: torch.Tensor,
        pair_feats: torch.Tensor,
        h_opes_pooled: torch.Tensor,
        h_mas_pooled: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """h_ops: [B, n_jobs, d] next-operation embeddings; returns pair probabilities [B, n_jobs * n_machines]."""
        B, n_jobs, _ = h_ops.shape
        n_machines = h_mas.size(1)
        n_pairs = n_jobs * n_machines

        h_ops_exp = h_ops.unsqueeze(2).expand(-1, -1, n_machines, -1)
        h_mas_exp = h_mas.unsqueeze(1).expand(-1, n_jobs, -1, -1)
        h_ops_flat = h_ops_exp.reshape(B, n_pairs, -1)
        h_mas_flat = h_mas_exp.reshape(B, n_pairs, -1)

        pair_init = torch.cat([h_ops_flat, h_mas_flat, pair_feats], dim=-1)
        pair_embed = self.pair_proj(pair_init)

        action_mask_bool = action_mask.bool() if action_mask.dtype != torch.bool else action_mask
        key_padding_mask = ~action_mask_bool
        # Finished instances have no feasible pair; unmask them to avoid NaNs in the attention
        all_masked = ~action_mask_bool.any(dim=-1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        pair_enriched = self.pair_self_attn(pair_embed, key_padding_mask)

        pooled_ope = h_opes_pooled.unsqueeze(1).expand(-1, n_pairs, -1)
        pooled_ma = h_mas_pooled.unsqueeze(1).expand(-1, n_pairs, -1)

        actor_input = torch.cat([pair_enriched, pooled_ope, pooled_ma], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)

        scores = scores.masked_fill(~action_mask_bool, float('-inf'))

        action_probs = F.softmax(scores, dim=-1)
        action_probs = torch.where(
            all_masked.unsqueeze(-1),
            torch.ones_like(action_probs) / n_pairs,
            action_probs,
        )
        return action_probs


class StateCritic(nn.Module):
    """State-value function V(s) (Eq. (20)), evaluated before the timing decision and shared by both heads."""

    def __init__(self, ope_embed_dim: int, ma_embed_dim: int,
                 global_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        in_dim = ope_embed_dim * 2 + ma_embed_dim * 2 + global_dim
        self.net = MLPActor(in_dim, hidden_dim, 1, num_layers, dropout)

    def forward(self, h_opes_pooled: torch.Tensor, h_mas_pooled: torch.Tensor,
                global_feats: torch.Tensor) -> torch.Tensor:
        """Returns V(s) [batch]."""
        x = torch.cat([h_opes_pooled, h_mas_pooled, global_feats], dim=-1)
        return self.net(x).squeeze(-1)


class TimingHead(nn.Module):
    """Timing head pi_T(t|s) (Section 4.4.2): selects a start time from the timing action space.

    Each candidate time is represented by the mean embedding of the operations that can start at it plus its encoded
    scalar features (Eqs. (14)-(15)); the candidates attend to each other and are scored by an MLP (Eq. (16)).
    """

    def __init__(self, ope_embed_dim: int, ma_embed_dim: int,
                 scalar_feat_dim: int = 9, hidden_dim: int = 64,
                 use_candidate_self_attention: bool = True):
        super().__init__()
        self.ope_embed_dim = ope_embed_dim
        self.ma_embed_dim = ma_embed_dim
        self.hidden_dim = hidden_dim
        self.use_candidate_self_attention = use_candidate_self_attention

        self.op_proj = nn.Linear(ope_embed_dim, hidden_dim)

        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.candidate_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True,
        ) if use_candidate_self_attention else None

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_feats: torch.Tensor, mask: torch.Tensor):
        """candidate_feats: [batch, max_cand, hidden_dim] candidate representations."""
        if self.use_candidate_self_attention:
            attn_feats = self.candidate_self_attn(
                candidate_feats, src_key_padding_mask=~mask
            )
        else:
            attn_feats = candidate_feats
        scores = self.scorer(attn_feats).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        all_masked = ~mask.any(dim=-1)
        probs = F.softmax(scores, dim=-1)
        probs = torch.where(all_masked.unsqueeze(-1),
                            torch.ones_like(probs) / probs.size(-1), probs)
        return probs

    def build_candidate_features(self, h_opes: torch.Tensor, h_mas: torch.Tensor,
                                  candidate_data: dict):
        """Candidate representation: masked mean of the ready-operation embeddings plus the encoded scalar features."""
        ready_op = candidate_data['ready_op_mask']
        scalar = candidate_data['scalar_feats']
        B, C, N_ops = ready_op.shape

        h_ops_proj = self.op_proj(h_opes)

        # Masked mean over the operations that can start at each candidate time

        h_ops_exp = h_ops_proj.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, N_ops, -1)

        ready_mask_flat = ready_op.reshape(B * C, N_ops).unsqueeze(-1).float()

        matching_base_masked = h_ops_exp * ready_mask_flat
        n_ready = ready_mask_flat.sum(dim=1).clamp(min=1)
        matching_feat = matching_base_masked.sum(dim=1) / n_ready
        matching_feat = matching_feat.reshape(B, C, -1)

        scalar_feat = self.scalar_proj(scalar)
        candidate_feats = matching_feat + scalar_feat

        return candidate_feats

    def act(self, h_opes: torch.Tensor, h_mas: torch.Tensor,
            candidate_data: dict, deterministic: bool = False,
            action: torch.Tensor = None):
        """Returns: action [batch], log_prob [batch], entropy [batch]"""
        feats = self.build_candidate_features(h_opes, h_mas, candidate_data)
        mask = candidate_data['mask']
        probs = self.forward(feats, mask)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def act_with_probs(self, h_opes: torch.Tensor, h_mas: torch.Tensor,
                       candidate_data: dict, deterministic: bool = False,
                       action: torch.Tensor = None):
        """Like act(), and also returns the probabilities."""
        feats = self.build_candidate_features(h_opes, h_mas, candidate_data)
        mask = candidate_data['mask']
        probs = self.forward(feats, mask)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, probs


class FJSPNetwork(nn.Module):
    """W2S policy: shared HGNN encoder, timing head, pairing head and critic (Fig. 5 of the paper).

    Per decision step: encode() -> act_timing() -> act_pair() for the pairs that start at the chosen time.
    """

    def __init__(
        self,
        in_size_ope: int,
        in_size_ma: int,
        pair_private_dim: int,
        out_size_ope: int = 64,
        out_size_ma: int = 64,
        hidden_size: int = 128,
        num_hgnn_layers: int = 3,
        num_heads: int = 1,
        actor_hidden_dim: int = 256,
        actor_num_layers: int = 3,
        dropout: float = 0.0,
        pair_attn_layers: int = 2,
        timing_scalar_dim: int = 9,
        timing_hidden_dim: int = 64,
        use_candidate_self_attention: bool = True,
        global_feat_dim: int = 2,
    ):
        super().__init__()

        self.out_size_ope = out_size_ope
        self.out_size_ma = out_size_ma

        self.hgnn_encoder = HGNNEncoder(
            in_size_ope=in_size_ope,
            in_size_ma=in_size_ma,
            out_size_ope=out_size_ope,
            out_size_ma=out_size_ma,
            hidden_size=hidden_size,
            num_layers=num_hgnn_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.pair_agent = PairAgent(
            ope_embed_dim=out_size_ope,
            ma_embed_dim=out_size_ma,
            pair_private_dim=pair_private_dim,
            hidden_dim=actor_hidden_dim,
            num_layers=actor_num_layers,
            dropout=dropout,
            attn_embed_dim=128,
            attn_heads=4,
            attn_layers=pair_attn_layers,
        )

        self.timing_head = TimingHead(
            ope_embed_dim=out_size_ope,
            ma_embed_dim=out_size_ma,
            scalar_feat_dim=timing_scalar_dim,
            hidden_dim=timing_hidden_dim,
            use_candidate_self_attention=use_candidate_self_attention,
        )

        self.critic = StateCritic(
            ope_embed_dim=out_size_ope,
            ma_embed_dim=out_size_ma,
            global_dim=global_feat_dim,
            hidden_dim=actor_hidden_dim,
            num_layers=actor_num_layers,
            dropout=dropout,
        )

    def value_state(self, encoded: Dict[str, torch.Tensor],
                    global_feats: torch.Tensor) -> torch.Tensor:
        """V(s) [batch]."""
        return self.critic(
            encoded["h_opes_pooled"], encoded["h_mas_pooled"], global_feats,
        )

    def act_timing(
        self,
        encoded: Dict[str, torch.Tensor],
        candidate_data: dict,
        deterministic: bool = False,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a start time t ~ pi_T(t|s). Returns (action, log_prob, entropy)."""
        return self.timing_head.act(
            encoded["h_opes"], encoded["h_mas"], candidate_data, deterministic, action
        )

    def act_timing_with_probs(
        self,
        encoded: Dict[str, torch.Tensor],
        candidate_data: dict,
        deterministic: bool = False,
        action: torch.Tensor = None,
    ):
        """Like act_timing(), and also returns the probabilities."""
        return self.timing_head.act_with_probs(
            encoded["h_opes"], encoded["h_mas"], candidate_data, deterministic, action
        )


    def encode(
        self,
        ope_feats: torch.Tensor,
        ma_feats: torch.Tensor,
        proc_time: torch.Tensor,
        ope_ma_adj: torch.Tensor,
        ope_pre_adj: torch.Tensor,
        ope_sub_adj: torch.Tensor,
        ope_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Normalize the features, run the HGNN encoder and pool the embeddings.

        Returns operation / machine embeddings and the pooled state embedding [mean || max] of each (Eq. (13)).
        """
        ope_feats, ma_feats, proc_time = normalize_feats(
            ope_feats, ma_feats, proc_time, ope_valid_mask
        )

        h_opes, h_mas = self.hgnn_encoder(
            ope_feats, ma_feats, proc_time, ope_ma_adj, ope_pre_adj, ope_sub_adj
        )

        if ope_valid_mask is not None:
            mask = ope_valid_mask.bool()
            mask_exp = mask.unsqueeze(-1).float()

            masked_h = h_opes * mask_exp
            h_opes_sum = masked_h.sum(dim=1)
            count_safe = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            h_opes_pooled_mean = h_opes_sum / count_safe

            h_opes_for_max = h_opes.clone()
            mask_inv = ~mask.unsqueeze(-1).expand_as(h_opes)
            h_opes_for_max[mask_inv] = -1e9
            h_opes_pooled_max = h_opes_for_max.max(dim=1).values

            h_opes_pooled = torch.cat([h_opes_pooled_mean, h_opes_pooled_max], dim=-1)
        else:
            h_opes_pooled = torch.cat([
                h_opes.mean(dim=1),
                h_opes.max(dim=1).values
            ], dim=-1)

        h_mas_pooled = torch.cat([
            h_mas.mean(dim=1),
            h_mas.max(dim=1).values
        ], dim=-1)

        return {
            "h_opes": h_opes,
            "h_mas": h_mas,
            "h_opes_pooled": h_opes_pooled,
            "h_mas_pooled": h_mas_pooled,
            "ope_ma_adj": ope_ma_adj,
        }

    def forward_pair_agent(
        self,
        encoded: Dict[str, torch.Tensor],
        job_op_indices: torch.Tensor,
        pair_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Pair probabilities; pair features are standardized across the pairs of each instance."""
        h_opes = encoded["h_opes"]
        h_mas = encoded["h_mas"]
        h_opes_pooled = encoded["h_opes_pooled"]
        h_mas_pooled = encoded["h_mas_pooled"]

        batch_size, num_jobs = job_op_indices.shape
        indices = job_op_indices.unsqueeze(-1).expand(-1, -1, self.out_size_ope)
        h_ops = torch.gather(h_opes, dim=1, index=indices)

        mean = pair_feats.mean(dim=1, keepdim=True)
        std = pair_feats.std(dim=1, keepdim=True).clamp(min=1e-5)
        pair_feats = (pair_feats - mean) / std

        action_probs = self.pair_agent(
            h_ops, h_mas, pair_feats,
            h_opes_pooled, h_mas_pooled,
            action_mask,
        )

        return {"action_probs": action_probs}

    def act_pair(
        self,
        encoded: Dict[str, torch.Tensor],
        job_op_indices: torch.Tensor,
        pair_feats: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an O-M pair c ~ pi_P(c|s,t). Returns (action, log_prob, entropy)."""
        output = self.forward_pair_agent(encoded, job_op_indices, pair_feats, action_mask)
        action_probs = output["action_probs"]
        dist = Categorical(action_probs)

        if action is None:
            if deterministic:
                action = action_probs.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
