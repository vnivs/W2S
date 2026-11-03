"""Heterogeneous graph neural network (HGNN) encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATedge(nn.Module):
    """Edge-aware graph attention layer that updates the machine nodes."""

    def __init__(
        self,
        in_feats: tuple,
        out_feats: int,
        num_heads: int = 1,
        feat_drop: float = 0.,
        attn_drop: float = 0.,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_src_feats = in_feats[0]
        self.in_dst_feats = in_feats[1]
        self.out_feats = out_feats

        self.fc_src = nn.Linear(self.in_src_feats, out_feats * num_heads, bias=False)
        self.fc_dst = nn.Linear(self.in_dst_feats, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(1, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.attn_e = nn.Parameter(torch.zeros(1, num_heads, out_feats))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if num_heads > 1:
            self.head_proj = nn.Linear(out_feats * num_heads, out_feats, bias=False)
        else:
            self.head_proj = nn.Identity()

        if self.in_dst_feats != out_feats:
            self.residual_proj = nn.Linear(self.in_dst_feats, out_feats, bias=False)
        else:
            self.residual_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if self.num_heads > 1:
            nn.init.xavier_normal_(self.head_proj.weight, gain=gain)

    def forward(
        self,
        ope_ma_adj: torch.Tensor,
        ope_feats: torch.Tensor,
        ma_feats: torch.Tensor,
        proc_time: torch.Tensor,
    ) -> torch.Tensor:
        h_src = self.feat_drop(ope_feats)
        h_dst = self.feat_drop(ma_feats)

        feat_src = self.fc_src(h_src)
        feat_dst = self.fc_dst(h_dst)
        feat_edge = self.fc_edge(proc_time.unsqueeze(-1))

        el = (feat_src.view(*feat_src.shape[:-1], self.num_heads, self.out_feats) * self.attn_l).sum(dim=-1, keepdim=True)
        er = (feat_dst.view(*feat_dst.shape[:-1], self.num_heads, self.out_feats) * self.attn_r).sum(dim=-1, keepdim=True)
        ee = (feat_edge.view(*feat_edge.shape[:-1], self.num_heads, self.out_feats) * self.attn_e).sum(dim=-1, keepdim=True)

        el = el.unsqueeze(2)
        er = er.unsqueeze(1)

        adj_mask = ope_ma_adj.unsqueeze(-1).unsqueeze(-1)

        e = adj_mask * el + ee + adj_mask * er
        e = self.leaky_relu(e)

        e_self = self.leaky_relu(er + er)

        e_all = torch.cat([e, e_self], dim=1)

        adj_with_self = torch.cat([
            ope_ma_adj,
            torch.ones(ope_ma_adj.size(0), 1, ope_ma_adj.size(2), device=ope_ma_adj.device)
        ], dim=1)
        mask = adj_with_self.unsqueeze(-1).unsqueeze(-1).bool()

        e_all = e_all.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(e_all.squeeze(-1), dim=1)
        alpha = self.attn_drop(alpha)

        alpha_neighbors = alpha[:, :-1, :, :]
        alpha_self = alpha[:, -1:, :, :]

        feat_src_reshaped = feat_src.view(*feat_src.shape[:-1], self.num_heads, self.out_feats)
        feat_dst_reshaped = feat_dst.view(*feat_dst.shape[:-1], self.num_heads, self.out_feats)
        feat_edge_reshaped = feat_edge.view(*feat_edge.shape[:-1], self.num_heads, self.out_feats)

        Wmu = feat_src_reshaped.unsqueeze(2) + feat_edge_reshaped

        neighbor_msg = (Wmu * alpha_neighbors.unsqueeze(-1)).sum(dim=1)
        self_msg = feat_dst_reshaped * alpha_self.squeeze(1).unsqueeze(-1)

        attn_out = neighbor_msg + self_msg
        attn_out = attn_out.view(*attn_out.shape[:-2], -1)

        attn_out = self.head_proj(attn_out)

        residual = self.residual_proj(ma_feats)

        out = attn_out + residual

        out = torch.tanh(out)

        return out


class MLPsim(nn.Module):
    """MLP used in the operation update."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_feats),
        )

    def forward(
        self,
        adj: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate the source-node features of every target node through the adjacency matrix."""
        msg = adj.unsqueeze(-1) * feats.unsqueeze(1)
        agg = msg.sum(dim=2)
        out = self.project(agg)
        return out


class MLPs(nn.Module):
    """Operation-node update."""

    def __init__(
        self,
        in_sizes: list,
        hidden_size: int,
        out_size: int,
    ):
        super().__init__()
        self.out_size = out_size
        self.in_ope_size = in_sizes[1]

        self.mlps = nn.ModuleList([
            MLPsim(in_sizes[i], out_size, hidden_size) for i in range(len(in_sizes))
        ])

        self.project = nn.Sequential(
            nn.Tanh(),
            nn.Linear(out_size * len(in_sizes), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
        )

        if self.in_ope_size != out_size:
            self.residual_proj = nn.Linear(self.in_ope_size, out_size, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        ope_ma_adj: torch.Tensor,
        ope_pre_adj: torch.Tensor,
        ope_sub_adj: torch.Tensor,
        ma_feats: torch.Tensor,
        ope_feats: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_opes = ope_feats.shape[:2]
        device = ope_feats.device

        self_adj = torch.eye(num_opes, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        h = [ma_feats, ope_feats, ope_feats, ope_feats]
        adj = [ope_ma_adj, ope_pre_adj, ope_sub_adj, self_adj]

        embeddings = []
        for i in range(len(self.mlps)):
            embeddings.append(self.mlps[i](adj[i], h[i]))

        combined = torch.cat(embeddings, dim=-1)
        new_feats = self.project(combined)

        residual = self.residual_proj(ope_feats)

        out = new_feats + residual

        out = torch.tanh(out)

        return out


class HGNNEncoder(nn.Module):
    """HGNN encoder with alternating machine and operation updates."""

    def __init__(
        self,
        in_size_ope: int,
        in_size_ma: int,
        out_size_ope: int,
        out_size_ma: int,
        hidden_size: int,
        num_layers: int = 3,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.get_machines = nn.ModuleList()
        self.get_machines.append(
            GATedge((in_size_ope, in_size_ma), out_size_ma, num_heads, dropout, dropout)
        )
        for _ in range(num_layers - 1):
            self.get_machines.append(
                GATedge((out_size_ope, out_size_ma), out_size_ma, num_heads, dropout, dropout)
            )

        self.get_operations = nn.ModuleList()
        self.get_operations.append(
            MLPs([out_size_ma, in_size_ope, in_size_ope, in_size_ope], hidden_size, out_size_ope)
        )
        for _ in range(num_layers - 1):
            self.get_operations.append(
                MLPs([out_size_ma, out_size_ope, out_size_ope, out_size_ope], hidden_size, out_size_ope)
            )

    def forward(
        self,
        ope_feats: torch.Tensor,
        ma_feats: torch.Tensor,
        proc_time: torch.Tensor,
        ope_ma_adj: torch.Tensor,
        ope_pre_adj: torch.Tensor,
        ope_sub_adj: torch.Tensor,
    ) -> tuple:
        h_opes = ope_feats
        h_mas = ma_feats

        for i in range(self.num_layers):
            h_mas = self.get_machines[i](ope_ma_adj, h_opes, h_mas, proc_time)

            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, h_mas, h_opes)

        return h_opes, h_mas
