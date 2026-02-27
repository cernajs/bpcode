import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    """
    psi(h, s) -> g
    Learned geometry embedding for reachability-style distances.
    No normalization here — we normalize only where needed so the network has full freedom to spread embeddings.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        geo_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = deter_dim + stoch_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, geo_dim),
        )
        # Small output init — don't start too spread out
        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1) if s is not None else h
        # Normalize onto unit sphere for stable distances
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Core loss
# ---------------------------------------------------------------------------


def temporal_reachability_loss(
    g_seq: torch.Tensor,
    pos_k: int = 3,
    neg_k: int = 12,
    margin: float = 0.6,
    uniformity_weight: float = 0.1,
    uniformity_t: float = 2.0,
) -> torch.Tensor:
    """
    g_seq: [B, T, D]  (unit-normalized embeddings)

    Three terms:
      1. ranking:     d(anchor, close) + margin < d(anchor, far)   [no absolute target]
      2. uniformity:  pushes embeddings apart on the sphere globally
      3. cross-seq:   states from *different* sequences are treated as far

    Notable removals vs old version:
      - NO loss_reg (d_pos ~ pos_delta/pos_k) — that caused collapse 
      (Wang & Isola 2020 Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere!!)
    """
    B, T, D = g_seq.shape
    device = g_seq.device

    if T <= neg_k or pos_k < 1 or neg_k <= pos_k:
        return g_seq.new_zeros(())

    # ---- 1. Ranking loss (multiple negatives per anchor) ----
    # Anchor at time i, positive within [i+1, i+pos_k], negative at i+neg_k
    # Use B anchors, but also add B more with a second set of negatives
    # (i + neg_k + 1) when available — doubles the gradient signal cheaply.

    max_i = T - neg_k - 1  # leave room for two neg offsets
    if max_i < 1:
        max_i = T - neg_k

    i = torch.randint(0, max_i, (B,), device=device)
    pos_delta = torch.randint(1, pos_k + 1, (B,), device=device)
    j_pos = i + pos_delta
    j_neg1 = i + neg_k
    j_neg2 = (i + neg_k + 1).clamp(max=T - 1)

    b = torch.arange(B, device=device)
    g_i = g_seq[b, i]           # [B, D]
    g_pos = g_seq[b, j_pos]     # [B, D]
    g_neg1 = g_seq[b, j_neg1]   # [B, D]
    g_neg2 = g_seq[b, j_neg2]   # [B, D]

    d_pos = torch.norm(g_i - g_pos, dim=-1)    # [B]
    d_neg1 = torch.norm(g_i - g_neg1, dim=-1)  # [B]
    d_neg2 = torch.norm(g_i - g_neg2, dim=-1)  # [B]

    loss_rank = (
        F.relu(d_pos + margin - d_neg1).mean()
        + F.relu(d_pos + margin - d_neg2).mean()
    ) * 0.5

    # ---- 2. Uniformity loss (prevent all-cluster collapse) ----
    # Subsample for efficiency: use [B, D] anchors only
    g_flat = g_seq.reshape(B * T, D)
    # Random subsample to keep O(N^2) manageable
    max_uni = min(B * T, 256)
    if B * T > max_uni:
        idx = torch.randperm(B * T, device=device)[:max_uni]
        g_uni = g_flat[idx]
    else:
        g_uni = g_flat

    # logsumexp(-t * ||g_i - g_j||^2) over all pairs — Wang & Isola 2020
    sq = torch.cdist(g_uni, g_uni).pow(2)
    # Mask diagonal
    mask = torch.eye(g_uni.size(0), device=device, dtype=torch.bool)
    sq = sq.masked_fill(mask, 1e9)
    loss_uni = torch.logsumexp(-uniformity_t * sq, dim=1).mean()

    # ---- 3. Cross-sequence negatives ----
    # States from different sequences in the batch are almost certainly
    # in different locations — treat them as negatives for each anchor.
    # Pick one cross-seq negative per anchor by rolling the batch dim.
    g_cross = g_seq.roll(1, dims=0)[b, i]  # [B, D]
    d_cross = torch.norm(g_i - g_cross, dim=-1)
    loss_cross = F.relu(d_pos + margin - d_cross).mean()

    return loss_rank + uniformity_weight * loss_uni + 0.5 * loss_cross


# ---------------------------------------------------------------------------
# Bank / frontier utils  (unchanged)
# ---------------------------------------------------------------------------


def min_bank_dist(x: torch.Tensor, bank_sub: torch.Tensor | None) -> torch.Tensor:
    """
    x: [B, T, D] or [N, D]
    returns nearest-neighbour distance to bank_sub.
    """
    if bank_sub is None or bank_sub.numel() == 0:
        return x.new_zeros(x.shape[:-1])
    x_flat = x.reshape(-1, x.shape[-1])
    d = torch.cdist(x_flat, bank_sub).min(dim=1).values
    return d.view(*x.shape[:-1])


# ---------------------------------------------------------------------------
# Actor planning constraint
# ---------------------------------------------------------------------------


def geo_step_penalty(g_seq: torch.Tensor, step_radius: float) -> torch.Tensor:
    """
    Penalise imagined one-step jumps that are too large in geometry space.
    g_seq: [B, H+1, D]
    """
    step_d = torch.norm(g_seq[:, 1:] - g_seq[:, :-1], dim=-1)
    return F.relu(step_d - step_radius).pow(2).mean()


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def geo_embedding_stats(g: torch.Tensor) -> dict:
    """
    g: [N, D] or [B, T, D]
    Returns a dict of scalars useful for Tensorboard logging.
    Tells you whether the embeddings are spreading or collapsing.
    """
    g_flat = g.detach().reshape(-1, g.shape[-1])
    # Pairwise distances on a random subsample
    n = min(g_flat.size(0), 256)
    idx = torch.randperm(g_flat.size(0), device=g_flat.device)[:n]
    g_sub = g_flat[idx]
    dists = torch.cdist(g_sub, g_sub)
    mask = ~torch.eye(n, device=g_flat.device, dtype=torch.bool)
    dists_off = dists[mask]
    return {
        "geo/mean_pairwise_dist": dists_off.mean().item(),
        "geo/min_pairwise_dist": dists_off.min().item(),
        "geo/std_pairwise_dist": dists_off.std().item(),
        # std of the embedding coordinates — should be >0 if not collapsed
        "geo/embed_std": g_flat.std(dim=0).mean().item(),
    }