import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoEncoder(nn.Module):
    """
    psi(h, s) -> g
    Learned geometry embedding used for reachability-style distances.
    """
    def __init__(self, deter_dim: int, stoch_dim: int, geo_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        in_dim = deter_dim + stoch_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, geo_dim),
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1) if s is not None else h
        g = self.net(x)
        # bounded / stable metric space
        return F.normalize(g, dim=-1)


def temporal_reachability_loss(
    g_seq: torch.Tensor,
    pos_k: int = 3,
    neg_k: int = 12,
    margin: float = 0.25,
) -> torch.Tensor:
    """
    g_seq: [B, T, D]
    Encourage temporally-close states to be close in geometry space,
    and states >= neg_k apart to be farther away.
    """
    B, T, D = g_seq.shape
    if T <= neg_k or pos_k < 1 or neg_k <= pos_k:
        return g_seq.new_zeros(())

    # choose anchor times that allow a neg pair at i + neg_k
    i = torch.randint(0, T - neg_k, (B,), device=g_seq.device)
    pos_delta = torch.randint(1, pos_k + 1, (B,), device=g_seq.device)
    j_pos = i + pos_delta
    j_neg = i + neg_k

    b = torch.arange(B, device=g_seq.device)

    g_i = g_seq[b, i]
    g_pos = g_seq[b, j_pos]
    g_neg = g_seq[b, j_neg]

    d_pos = torch.norm(g_i - g_pos, dim=-1)
    d_neg = torch.norm(g_i - g_neg, dim=-1)

    # nearby states should have distance roughly proportional to short horizon
    target_pos = pos_delta.float() / float(pos_k)
    loss_reg = F.smooth_l1_loss(d_pos, target_pos)

    # farther states should be farther than near states
    loss_rank = F.relu(d_pos + margin - d_neg).mean()

    return loss_reg + loss_rank


def min_bank_dist(x: torch.Tensor, bank_sub: torch.Tensor | None) -> torch.Tensor:
    """
    x: [B, T, D] or [N, D]
    returns nearest-neighbor distance to bank_sub.
    """
    if bank_sub is None or bank_sub.numel() == 0:
        return x.new_zeros(x.shape[:-1])

    x_flat = x.reshape(-1, x.shape[-1])
    d = torch.cdist(x_flat, bank_sub).min(dim=1).values
    return d.view(*x.shape[:-1])


def geo_step_penalty(g_seq: torch.Tensor, step_radius: float) -> torch.Tensor:
    """
    Penalize imagined one-step jumps that are too large in geometry space.
    g_seq: [B, H+1, D]
    """
    step_d = torch.norm(g_seq[:, 1:] - g_seq[:, :-1], dim=-1)
    return F.relu(step_d - step_radius).pow(2).mean()