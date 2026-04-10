#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling scripts importable when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dreamer_vics import PointMazeLargeDreamerWrapper
from dreamer_kstep_replay import compute_geo_head_correlation
from geom_head import GeoEncoder
from models import ConvEncoder, RSSM
from utils import get_device, set_seed
from test_best_dreamer_heads import (
    _compute_teacher_quality,
    _embed_with_head,
    _evaluate_embedding,
    _fit_scale,
    _make_oracle_topology,
    _make_summary_plots,
    _sample_pairs_by_bins,
    build_replay_graph,
    collect_teacher_data_with_quotas,
)


def infer_geo_dims(geo_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer (hidden_dim, input_dim, geo_dim) from GeoEncoder state dict."""
    w0 = geo_state["net.0.weight"]
    w_last = geo_state["net.4.weight"]
    hidden_dim = int(w0.shape[0])
    input_dim = int(w0.shape[1])
    geo_dim = int(w_last.shape[0])
    return hidden_dim, input_dim, geo_dim



def load_best_bundle(
    ckpt_path: str,
    device: torch.device,
    obs_channels: int,
    act_dim: int,
    embed_dim: int,
    stoch_dim: int,
    deter_dim: int,
    hidden_dim: int,
) -> tuple[ConvEncoder, RSSM, GeoEncoder, dict[str, Any]]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if "geo_head" not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain 'geo_head'.")

    encoder = ConvEncoder(embedding_size=embed_dim, in_channels=obs_channels).to(device)
    rssm = RSSM(stoch_dim, deter_dim, act_dim, embed_dim, hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])
    encoder.eval()
    rssm.eval()

    geo_state = ckpt["geo_head"]
    head_hidden, input_dim, geo_dim = infer_geo_dims(geo_state)
    expected_input = int(deter_dim + stoch_dim)
    if input_dim != expected_input:
        raise RuntimeError(
            f"Stored geo_head expects input dim {input_dim}, but deter_dim+stoch_dim={expected_input}."
        )
    geo_head = GeoEncoder(deter_dim, stoch_dim, geo_dim=geo_dim, hidden_dim=head_hidden).to(device)
    geo_head.load_state_dict(geo_state)
    geo_head.eval()

    meta = {
        "episode": int(ckpt.get("episode", -1)),
        "total_steps": int(ckpt.get("total_steps", -1)),
        "best_geo_score": float(ckpt.get("best_geo_score", 0.0)),
        "geo_dim": geo_dim,
        "geo_hidden": head_hidden,
    }
    return encoder, rssm, geo_head, meta



def evaluate_best_bundle(args: argparse.Namespace, ckpt_path: str) -> dict[str, Any]:
    set_seed(args.seed)
    device = get_device()

    env = PointMazeLargeDreamerWrapper(img_size=args.img_size, reset_mode=args.reset_mode)
    geodesic = env.geodesic
    H, W, C = env.observation_space.shape
    act_dim = int(env.action_space.shape[0])

    encoder, rssm, geo_head, bundle_meta = load_best_bundle(
        ckpt_path=ckpt_path,
        device=device,
        obs_channels=C,
        act_dim=act_dim,
        embed_dim=args.embed_dim,
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
        hidden_dim=args.hidden_dim,
    )

    teacher_data, mode_hist = collect_teacher_data_with_quotas(
        env=env,
        encoder=encoder,
        rssm=rssm,
        actor=None,
        device=device,
        bit_depth=args.bit_depth,
        teacher_collect_episodes=args.teacher_collect_episodes,
        state_stride=args.state_stride,
        max_nodes=args.teacher_max_nodes,
        deterministic_policy=args.policy_deterministic,
        expl_noise=args.expl_noise,
        teacher_high_noise=args.teacher_high_noise,
        teacher_random_fraction=args.teacher_random_fraction,
        teacher_subset_fraction=args.teacher_subset_fraction,
        teacher_start_subset_size=args.teacher_start_subset_size,
    )

    replay_graph = build_replay_graph(
        data=teacher_data,
        basis_name=args.graph_knn_basis,
        graph_knn_k=args.graph_knn_k,
        graph_knn_weight=args.graph_knn_weight,
        graph_temporal_weight=args.graph_temporal_weight,
        graph_knn_max_percentile=args.graph_knn_max_percentile,
        graph_same_ep_gap=args.graph_same_ep_gap,
        max_graph_nodes=args.max_graph_nodes,
        graph_views=[x.strip() for x in args.graph_views.split(",") if x.strip()],
        graph_min_view_votes=args.graph_min_view_votes,
    )

    teacher_quality = _compute_teacher_quality(teacher_data, replay_graph, geodesic)
    teacher_quality["thresholds"] = {
        "min_coverage": float(args.teacher_min_coverage),
        "min_giant_component_fraction": float(args.teacher_min_giant_fraction),
        "min_geo_replay_spearman": float(args.teacher_min_geo_replay_spearman),
    }
    teacher_quality["passes"] = {
        "coverage": bool(teacher_quality["coverage"] >= args.teacher_min_coverage),
        "giant_component_fraction": bool(teacher_quality["giant_component_fraction"] >= args.teacher_min_giant_fraction),
        "geo_vs_replay": bool(teacher_quality["geo_vs_replay"]["spearman"] >= args.teacher_min_geo_replay_spearman),
    }
    teacher_quality["topology_ok"] = bool(all(teacher_quality["passes"].values()))

    oracle_topo = _make_oracle_topology(geodesic)
    gc = replay_graph.giant_nodes_local
    if len(gc) < 32:
        raise RuntimeError("Replay graph giant component too small for meaningful evaluation.")

    pair_train_i, pair_train_j, replay_train_d = _sample_pairs_by_bins(
        replay_graph.dist_mat, gc, args.train_pairs, seed=args.seed + 17
    )
    pair_eval_i, pair_eval_j, replay_eval_d = _sample_pairs_by_bins(
        replay_graph.dist_mat, gc, args.eval_pairs, seed=args.seed + 29
    )

    global_nodes = replay_graph.node_indices
    raw_emb = teacher_data.z[global_nodes]
    best_emb = _embed_with_head(
        "best_geo_head",
        teacher_data.h[global_nodes],
        teacher_data.s[global_nodes],
        teacher_data.z[global_nodes],
        geo_head,
        device,
    )

    raw_scale = _fit_scale(np.linalg.norm(raw_emb[pair_train_i] - raw_emb[pair_train_j], axis=-1), replay_train_d)
    best_scale = _fit_scale(np.linalg.norm(best_emb[pair_train_i] - best_emb[pair_train_j], axis=-1), replay_train_d)

    raw_eval = _evaluate_embedding(
        name="raw_hs",
        emb_nodes=raw_emb,
        replay_graph=replay_graph,
        data=teacher_data,
        geodesic=geodesic,
        oracle_topo=oracle_topo,
        scale_for_distance=raw_scale,
        pair_eval_i_local=pair_eval_i,
        pair_eval_j_local=pair_eval_j,
        replay_eval_d=replay_eval_d,
        device=device,
        env=env,
        encoder=encoder,
        rssm=rssm,
        bit_depth=args.bit_depth,
        head=None,
    )
    best_eval = _evaluate_embedding(
        name="best_geo_head",
        emb_nodes=best_emb,
        replay_graph=replay_graph,
        data=teacher_data,
        geodesic=geodesic,
        oracle_topo=oracle_topo,
        scale_for_distance=best_scale,
        pair_eval_i_local=pair_eval_i,
        pair_eval_j_local=pair_eval_j,
        replay_eval_d=replay_eval_d,
        device=device,
        env=env,
        encoder=encoder,
        rssm=rssm,
        bit_depth=args.bit_depth,
        head=geo_head,
    )

    # Also compute the exact same direct correlation metric used during training.
    obs_arr = teacher_data.obs
    pos_arr = teacher_data.pos
    direct_replay = compute_geo_head_correlation(
        encoder, rssm, geo_head, obs_arr, pos_arr, replay_graph.dist, geodesic,
        prefix="best_geo_head_replay", device=device, bit_depth=args.bit_depth, n_pairs=args.geo_eval_pairs,
    )
    direct_oracle = compute_geo_head_correlation(
        encoder, rssm, geo_head, obs_arr, pos_arr, geodesic.dist_matrix, geodesic,
        prefix="best_geo_head_oracle", device=device, bit_depth=args.bit_depth, n_pairs=args.geo_eval_pairs,
    )

    visited_cells = int(len(np.unique(teacher_data.cell_idx[replay_graph.node_indices])))
    summary = {
        "file": os.path.basename(ckpt_path),
        "bundle_meta": bundle_meta,
        "sanity": {
            "collection_mode_hist": {k: int(v) for k, v in mode_hist.items()},
            "teacher_quality": teacher_quality,
            "n_rollout_nodes": int(teacher_data.z.shape[0]),
            "n_graph_nodes": int(len(replay_graph.node_indices)),
            "n_free_cells_total": int(geodesic.n_free),
            "n_cells_visited": int(visited_cells),
            "coverage": float(visited_cells / max(int(geodesic.n_free), 1)),
            "graph": replay_graph.graph_stats,
            "mean_reward": float(np.mean(teacher_data.reward)),
            "success_rate": float(np.mean(teacher_data.success)),
        },
        "raw_hs": raw_eval,
        "best_geo_head": best_eval,
        "best_geo_head_direct": {
            **direct_replay,
            **direct_oracle,
        },
    }
    env.close()
    return summary



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct evaluation of stored best_geo_bundle.pt heads")
    p.add_argument("--run_dir", type=str, default="./dreamer_topo/")
    p.add_argument("--checkpoints", nargs="*", type=str, default=["best_geo_bundle.pt"])
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--reset_mode", type=str, default="fixed_start")

    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)

    p.add_argument("--state_stride", type=int, default=2)
    p.add_argument("--teacher_collect_episodes", type=int, default=80)
    p.add_argument("--teacher_max_nodes", type=int, default=3500)
    p.add_argument("--teacher_high_noise", type=float, default=0.45)
    p.add_argument("--teacher_random_fraction", type=float, default=0.4)
    p.add_argument("--teacher_subset_fraction", type=float, default=0.35)
    p.add_argument("--teacher_start_subset_size", type=int, default=6)
    p.add_argument("--teacher_min_coverage", type=float, default=0.45)
    p.add_argument("--teacher_min_giant_fraction", type=float, default=0.3)
    p.add_argument("--teacher_min_geo_replay_spearman", type=float, default=0.45)
    p.add_argument("--policy_deterministic", action="store_true")
    p.add_argument("--expl_noise", type=float, default=0.15)

    p.add_argument("--max_graph_nodes", type=int, default=1800)
    p.add_argument("--graph_knn_basis", type=str, default="encoder", choices=["encoder", "z", "h", "s"])
    p.add_argument("--graph_views", type=str, default="encoder,z")
    p.add_argument("--graph_min_view_votes", type=int, default=1)
    p.add_argument("--graph_knn_k", type=int, default=6)
    p.add_argument("--graph_knn_weight", type=float, default=1.0)
    p.add_argument("--graph_temporal_weight", type=float, default=1.0)
    p.add_argument("--graph_knn_max_percentile", type=float, default=80.0)
    p.add_argument("--graph_same_ep_gap", type=int, default=2)

    p.add_argument("--train_pairs", type=int, default=1500)
    p.add_argument("--eval_pairs", type=int, default=1500)
    p.add_argument("--geo_eval_pairs", type=int, default=2000)
    return p



def make_summary_plot(results: list[dict[str, Any]], out_dir: str) -> None:
    if not results:
        return
    names = [r["file"] for r in results]
    x = np.arange(len(names))
    raw_geo = [float(r["raw_hs"]["oracle_geodesic"]["spearman"]) for r in results]
    best_geo = [float(r["best_geo_head"]["oracle_geodesic"]["spearman"]) for r in results]
    direct_geo = [float(r["best_geo_head_direct"].get("best_geo_head_oracle/spearman", 0.0)) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    axes[0].bar(x - 0.2, raw_geo, width=0.2, label="raw_hs")
    axes[0].bar(x, best_geo, width=0.2, label="best_head (cell-agg)")
    axes[0].bar(x + 0.2, direct_geo, width=0.2, label="best_head (direct)")
    axes[0].set_title("Oracle geodesic Spearman")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].legend(fontsize=8)

    raw_rep = [float(r["raw_hs"]["replay_distance"]["spearman"]) for r in results]
    best_rep = [float(r["best_geo_head"]["replay_distance"]["spearman"]) for r in results]
    direct_rep = [float(r["best_geo_head_direct"].get("best_geo_head_replay/spearman", 0.0)) for r in results]
    axes[1].bar(x - 0.2, raw_rep, width=0.2, label="raw_hs")
    axes[1].bar(x, best_rep, width=0.2, label="best_head (cell-agg)")
    axes[1].bar(x + 0.2, direct_rep, width=0.2, label="best_head (direct)")
    axes[1].set_title("Replay distance Spearman")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_best_head.png"))
    plt.close(fig)



def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or os.path.join(args.run_dir, "best_head_eval")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_paths = []
    for name in args.checkpoints:
        path = name if os.path.isabs(name) else os.path.join(args.run_dir, name)
        if os.path.isfile(path):
            ckpt_paths.append(path)
        else:
            print(f"[skip] missing checkpoint: {path}")
    if not ckpt_paths:
        raise SystemExit("No valid checkpoints found.")

    all_results = {"meta": vars(args), "checkpoints": []}
    for ckpt in ckpt_paths:
        print(f"\n=== Evaluating {os.path.basename(ckpt)} ===")
        res = evaluate_best_bundle(args, ckpt)
        all_results["checkpoints"].append(res)
        print(json.dumps({
            "file": res["file"],
            "saved_best_geo_score": res["bundle_meta"]["best_geo_score"],
            "coverage": res["sanity"]["coverage"],
            "teacher_geo_vs_replay": res["sanity"]["teacher_quality"]["geo_vs_replay"]["spearman"],
            "teacher_topology_ok": res["sanity"]["teacher_quality"]["topology_ok"],
            "raw_geo_spearman": res["raw_hs"]["oracle_geodesic"]["spearman"],
            "best_geo_spearman": res["best_geo_head"]["oracle_geodesic"]["spearman"],
            "best_geo_replay_spearman": res["best_geo_head"]["replay_distance"]["spearman"],
            "best_direct_oracle_spearman": res["best_geo_head_direct"].get("best_geo_head_oracle/spearman", 0.0),
            "best_direct_replay_spearman": res["best_geo_head_direct"].get("best_geo_head_replay/spearman", 0.0),
        }, indent=2))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    make_summary_plot(all_results["checkpoints"], out_dir)

    print(f"\nWrote metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()
