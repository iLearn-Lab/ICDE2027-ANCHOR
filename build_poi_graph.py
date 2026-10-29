import argparse
import json
import os
from collections import Counter

import pandas as pd
import torch


def build_graph_artifacts(train_csv: str, skip_self_loops: bool = True):
    df = pd.read_csv(train_csv, usecols=["trajectory_id", "UTC_time", "POI_id"])
    df["UTC_time"] = pd.to_datetime(df["UTC_time"], errors="coerce")
    df = df.sort_values(["trajectory_id", "UTC_time"])

    poi_ids = sorted(df["POI_id"].astype(str).unique().tolist())
    poi2idx = {poi: idx for idx, poi in enumerate(poi_ids)}

    edge_counter = Counter()
    pair_list = []
    num_pairs = 0
    num_self_pairs = 0

    for _, g in df.groupby("trajectory_id", sort=False):
        pois = g["POI_id"].astype(str).tolist()
        for prev_poi, next_poi in zip(pois[:-1], pois[1:]):
            num_pairs += 1
            if prev_poi == next_poi:
                num_self_pairs += 1
                if skip_self_loops:
                    continue

            prev_idx = poi2idx[prev_poi]
            next_idx = poi2idx[next_poi]
            pair_list.append((prev_idx, next_idx))

            if prev_idx == next_idx and skip_self_loops:
                continue

            u, v = sorted((prev_idx, next_idx))
            edge_counter[(u, v)] += 1

    pair_tensor = torch.tensor(pair_list, dtype=torch.long)

    undirected_edges = list(edge_counter.keys())
    undirected_weights = [edge_counter[e] for e in undirected_edges]

    if undirected_edges:
        edge_index_undirected = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
        edge_weight_undirected = torch.tensor(undirected_weights, dtype=torch.float32)
    else:
        edge_index_undirected = torch.empty((2, 0), dtype=torch.long)
        edge_weight_undirected = torch.empty((0,), dtype=torch.float32)

    bidir_src = []
    bidir_dst = []
    bidir_w = []
    for (u, v), w in zip(undirected_edges, undirected_weights):
        bidir_src.extend([u, v])
        bidir_dst.extend([v, u])
        bidir_w.extend([w, w])

    if bidir_src:
        edge_index = torch.tensor([bidir_src, bidir_dst], dtype=torch.long)
        edge_weight = torch.tensor(bidir_w, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    metadata = {
        "train_csv": train_csv,
        "num_rows": int(len(df)),
        "num_trajectories": int(df["trajectory_id"].nunique()),
        "num_pois": int(len(poi_ids)),
        "num_adjacent_pairs_before_filter": int(num_pairs),
        "num_self_pairs": int(num_self_pairs),
        "num_training_pairs_after_filter": int(pair_tensor.shape[0]),
        "num_undirected_edges": int(edge_index_undirected.shape[1]),
        "skip_self_loops": bool(skip_self_loops),
    }

    return {
        "num_nodes": len(poi_ids),
        "poi_ids": poi_ids,
        "poi2idx": poi2idx,
        "train_pairs": pair_tensor,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_index_undirected": edge_index_undirected,
        "edge_weight_undirected": edge_weight_undirected,
        "metadata": metadata,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_self_loops", action="store_true",
                    help="Keep transitions where prev_poi == next_poi.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    artifacts = build_graph_artifacts(
        train_csv=args.train_csv,
        skip_self_loops=not args.keep_self_loops,
    )

    graph_path = os.path.join(args.out_dir, "poi_graph_data.pt")
    poi_index_path = os.path.join(args.out_dir, "poi_index.json")
    meta_path = os.path.join(args.out_dir, "build_meta.json")

    torch.save(
        {
            "num_nodes": artifacts["num_nodes"],
            "train_pairs": artifacts["train_pairs"],
            "edge_index": artifacts["edge_index"],
            "edge_weight": artifacts["edge_weight"],
            "edge_index_undirected": artifacts["edge_index_undirected"],
            "edge_weight_undirected": artifacts["edge_weight_undirected"],
            "metadata": artifacts["metadata"],
        },
        graph_path,
    )

    json.dump(
        {
            "poi2idx": artifacts["poi2idx"],
            "idx2poi": artifacts["poi_ids"],
        },
        open(poi_index_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    json.dump(
        artifacts["metadata"],
        open(meta_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print(f"[Saved] graph data : {graph_path}")
    print(f"[Saved] poi index  : {poi_index_path}")
    print(f"[Saved] build meta : {meta_path}")
    for k, v in artifacts["metadata"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
