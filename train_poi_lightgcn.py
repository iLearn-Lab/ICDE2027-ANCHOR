import argparse
import json
import math
import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_normalized_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, device):
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.index_add_(0, row, edge_weight)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        edge_index,
        norm_weight,
        (num_nodes, num_nodes),
        device=device,
    ).coalesce()


def load_eval_pairs(eval_csv: str, poi2idx: dict):
    df = pd.read_csv(eval_csv, usecols=["trajectory_id", "UTC_time", "POI_id"])
    df["UTC_time"] = pd.to_datetime(df["UTC_time"], errors="coerce")
    df = df.sort_values(["trajectory_id", "UTC_time"])

    pairs = []
    dropped = 0
    for _, g in df.groupby("trajectory_id", sort=False):
        pois = g["POI_id"].astype(str).tolist()
        for prev_poi, next_poi in zip(pois[:-1], pois[1:]):
            prev_idx = poi2idx.get(prev_poi)
            next_idx = poi2idx.get(next_poi)
            if prev_idx is None or next_idx is None:
                dropped += 1
                continue
            pairs.append((prev_idx, next_idx))

    if pairs:
        pair_tensor = torch.tensor(pairs, dtype=torch.long)
    else:
        pair_tensor = torch.empty((0, 2), dtype=torch.long)
    return pair_tensor, dropped


class LightGCN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def compute_embeddings(self, norm_adj: torch.Tensor):
        x0 = self.embedding.weight
        xs = [x0]
        x = x0
        for _ in range(self.num_layers):
            x = torch.sparse.mm(norm_adj, x)
            xs.append(x)
        return torch.stack(xs, dim=0).mean(dim=0)

    def bpr_loss(self, final_emb: torch.Tensor, prev_idx, pos_idx, neg_idx, reg_weight: float):
        prev_emb = final_emb[prev_idx]
        pos_emb = final_emb[pos_idx]
        neg_emb = final_emb[neg_idx]

        pos_score = (prev_emb * pos_emb).sum(dim=1)
        neg_score = (prev_emb * neg_emb).sum(dim=1)
        ranking_loss = -F.logsigmoid(pos_score - neg_score).mean()

        reg = (
            self.embedding(prev_idx).pow(2).sum(dim=1) +
            self.embedding(pos_idx).pow(2).sum(dim=1) +
            self.embedding(neg_idx).pow(2).sum(dim=1)
        ).mean()
        return ranking_loss + reg_weight * reg


@torch.no_grad()
def recall_at_k(final_emb: torch.Tensor, eval_pairs: torch.Tensor, ks):
    if eval_pairs.numel() == 0:
        return {k: float("nan") for k in ks}

    item_emb_t = final_emb.t()
    results = {k: 0 for k in ks}
    num_items = int(final_emb.shape[0])
    max_k = min(max(ks), num_items)
    if max_k <= 0:
        return {k: float("nan") for k in ks}

    for prev_idx, true_idx in eval_pairs.tolist():
        scores = final_emb[prev_idx] @ item_emb_t
        topk = torch.topk(scores, k=max_k).indices
        for k in ks:
            kk = min(int(k), max_k)
            results[k] += int((topk[:kk] == true_idx).any().item())

    total = eval_pairs.shape[0]
    return {k: results[k] / total for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_data", required=True)
    ap.add_argument("--poi_index", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg_weight", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--eval_csv", default="")
    ap.add_argument("--eval_ks", type=str, default="10,50,100")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(args.graph_data, map_location="cpu")
    index_bundle = json.load(open(args.poi_index, "r", encoding="utf-8"))
    poi2idx = {str(k): int(v) for k, v in index_bundle["poi2idx"].items()}
    idx2poi = list(index_bundle["idx2poi"])

    train_pairs = bundle["train_pairs"].long()
    num_nodes = int(bundle["num_nodes"])
    edge_index = bundle["edge_index"].long()
    edge_weight = bundle["edge_weight"].float()

    norm_adj = build_normalized_adj(edge_index, edge_weight, num_nodes, device)

    model = LightGCN(num_nodes=num_nodes, emb_dim=args.emb_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_pairs = None
    eval_ks = sorted({int(x) for x in args.eval_ks.split(",") if x.strip()})
    dropped_eval = 0
    if args.eval_csv:
        eval_pairs, dropped_eval = load_eval_pairs(args.eval_csv, poi2idx)

    num_pairs = train_pairs.shape[0]
    num_batches = math.ceil(num_pairs / args.batch_size)
    best_metric = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(num_pairs)
        epoch_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            batch = train_pairs[idx]
            prev_idx = batch[:, 0].to(device)
            pos_idx = batch[:, 1].to(device)

            neg_idx = torch.randint(0, num_nodes, (batch.shape[0],), dtype=torch.long, device=device)
            same_mask = neg_idx == pos_idx
            while same_mask.any():
                neg_idx[same_mask] = torch.randint(0, num_nodes, (int(same_mask.sum().item()),), dtype=torch.long, device=device)
                same_mask = neg_idx == pos_idx

            optimizer.zero_grad()
            final_emb = model.compute_embeddings(norm_adj)
            loss = model.bpr_loss(final_emb, prev_idx, pos_idx, neg_idx, reg_weight=args.reg_weight)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(num_batches, 1)
        log = {"epoch": epoch, "train_loss": avg_loss}

        model.eval()
        with torch.no_grad():
            final_emb = model.compute_embeddings(norm_adj).detach().cpu()
            if eval_pairs is not None:
                metrics = recall_at_k(final_emb, eval_pairs, eval_ks)
                log.update({f"recall@{k}": metrics[k] for k in eval_ks})
                monitor = metrics[max(eval_ks)]
                if monitor > best_metric:
                    best_metric = monitor
                    best_state = {
                        "epoch": epoch,
                        "embeddings": final_emb.clone(),
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    }
            history.append(log)

        metric_msg = " ".join(
            [f"Recall@{k}={log[f'recall@{k}']:.6f}" for k in eval_ks if f"recall@{k}" in log]
        )
        print(f"Epoch {epoch:03d} | loss={avg_loss:.6f}" + (f" | {metric_msg}" if metric_msg else ""))

    final_embeddings = model.compute_embeddings(norm_adj).detach().cpu()

    model_path = os.path.join(args.out_dir, "lightgcn_model.pt")
    emb_path = os.path.join(args.out_dir, "poi_embeddings.pt")
    meta_path = os.path.join(args.out_dir, "train_meta.json")
    history_path = os.path.join(args.out_dir, "train_history.json")

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": {
                "num_nodes": num_nodes,
                "emb_dim": args.emb_dim,
                "num_layers": args.num_layers,
            },
        },
        model_path,
    )

    torch.save(
        {
            "embeddings": final_embeddings,
            "idx2poi": idx2poi,
            "poi2idx": poi2idx,
        },
        emb_path,
    )

    meta = {
        "graph_data": args.graph_data,
        "poi_index": args.poi_index,
        "out_dir": args.out_dir,
        "num_nodes": num_nodes,
        "num_train_pairs": int(num_pairs),
        "device": str(device),
        "emb_dim": args.emb_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "reg_weight": args.reg_weight,
        "eval_csv": args.eval_csv,
        "eval_ks": eval_ks,
        "eval_pairs": int(eval_pairs.shape[0]) if eval_pairs is not None else 0,
        "eval_pairs_dropped": int(dropped_eval),
        "best_recall_at_max_k": best_metric if best_metric >= 0 else None,
    }
    if best_state is not None:
        best_emb_path = os.path.join(args.out_dir, "poi_embeddings_best.pt")
        torch.save(
            {
                "embeddings": best_state["embeddings"],
                "idx2poi": idx2poi,
                "poi2idx": poi2idx,
                "best_epoch": best_state["epoch"],
            },
            best_emb_path,
        )
        meta["best_epoch"] = int(best_state["epoch"])

    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(history, open(history_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[Saved] model      : {model_path}")
    print(f"[Saved] embeddings : {emb_path}")
    print(f"[Saved] meta       : {meta_path}")
    print(f"[Saved] history    : {history_path}")


if __name__ == "__main__":
    main()
