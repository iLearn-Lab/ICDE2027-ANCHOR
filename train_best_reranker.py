import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from laststep_data_utils import build_laststep_samples, concat_history_frames, load_checkin_frame
from reranker_laststep_core import (
    HISTORY_CANDIDATE_FEATURE_NAMES,
    Metrics,
    RerankDatasetLastStep,
    RerankTransformerLastStep,
    build_catalog,
    build_laststep_context_map,
    build_region_catalog,
    build_user_features,
    collate_fn_laststep,
    compute_loss,
    count_candidate_miss,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_candidate_dir", required=True)
    ap.add_argument("--val_candidate_dir", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--catalog_csv", required=True)
    ap.add_argument("--poi_embeddings", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_model_ckpt", default="")
    ap.add_argument("--history_limit", type=int, default=50)
    ap.add_argument("--history_summary_len", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_gcn_init", action="store_true")
    ap.add_argument("--random_init_poi_emb", action="store_true")
    ap.add_argument("--disable_candidate_gcn_rank", action="store_true", default=True)
    ap.add_argument("--enable_candidate_gcn_rank", dest="disable_candidate_gcn_rank", action="store_false")
    ap.add_argument("--disable_candidate_score", action="store_true", default=True)
    ap.add_argument("--enable_candidate_score", dest="disable_candidate_score", action="store_false")
    ap.add_argument("--disable_candidate_cat_rank", action="store_true", default=True)
    ap.add_argument("--enable_candidate_cat_rank", dest="disable_candidate_cat_rank", action="store_false")
    ap.add_argument("--disable_same_cat_prev", action="store_true", default=True)
    ap.add_argument("--enable_same_cat_prev", dest="disable_same_cat_prev", action="store_false")
    ap.add_argument("--disable_distance_features", action="store_true")
    ap.add_argument("--disable_user_features", action="store_true")
    ap.add_argument("--disable_time_features", action="store_true")
    ap.add_argument("--disable_history", action="store_true")
    ap.add_argument("--enable_candidate_history_attention", action="store_true")
    ap.add_argument("--enable_history_attention_recency_bias", action="store_true")
    ap.add_argument("--enable_history_attention_category_bias", action="store_true")
    ap.add_argument("--enable_history_attention_time_bias", action="store_true")
    ap.add_argument("--enable_history_attention_dow_bias", action="store_true")
    ap.add_argument("--history_attention_bias_init", type=float, default=0.1)
    ap.add_argument("--disable_candidate_self_attention", action="store_true")
    ap.add_argument("--single_query_token", action="store_true")
    ap.add_argument("--recall_feature_dim", type=int, default=-1)
    ap.add_argument("--disable_recall_features", default="")
    ap.add_argument("--enable_history_candidate_features", action="store_true")
    ap.add_argument("--summary_recency_mode", choices=["legacy", "inverse"], default="legacy")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--rank_loss_lambda", type=float, default=1.0)
    ap.add_argument("--coarse_loss_lambda", type=float, default=0.3)
    ap.add_argument("--region_loss_lambda", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--region_grid_km", type=float, default=2.5)
    ap.add_argument("--best_metric", default="val_hit1_all")
    ap.add_argument("--early_stop_patience", type=int, default=0)
    return ap.parse_args()


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_rows = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = compute_loss(outputs, batch, args)
        loss.backward()
        optimizer.step()
        rows = batch["label_index"].size(0)
        total_rows += rows
        total_loss += float(loss.item()) * rows
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            rank=f"{parts['rank']:.4f}",
            coarse=f"{parts['coarse']:.4f}",
        )
    return total_loss / max(1, total_rows)


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total_rows = total_loss = total_rank_loss = total_coarse_loss = total_region_loss = 0.0
    valid_rows_total = 0
    hit1_all = hit5_all = hit10_all = 0
    mrr_all = 0.0
    hit1_hit = hit5_hit = hit10_hit = 0
    mrr_hit = 0.0
    region_correct = 0
    candidate_miss_count = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch)
        loss, parts = compute_loss(outputs, batch, args)
        logits = outputs["rank_logits"]
        labels = batch["label_index"]
        valid_rows = labels >= 0
        rows = logits.size(0)
        total_rows += rows
        total_loss += float(loss.item()) * rows
        total_region_loss += float(parts["region"]) * rows
        region_pred = torch.argmax(outputs["region_logits"], dim=1)
        region_correct += int((region_pred == batch["target_region_id"]).sum().item())

        if bool(valid_rows.any().item()):
            valid_count = int(valid_rows.sum().item())
            valid_rows_total += valid_count
            total_rank_loss += float(parts["rank"]) * valid_count
            total_coarse_loss += float(parts["coarse"]) * valid_count
            valid_logits = logits[valid_rows]
            valid_labels = labels[valid_rows]
            order = torch.argsort(valid_logits, dim=1, descending=True)
            ranks = (order == valid_labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
            hit1_hit += int((ranks < 1).sum().item())
            hit5_hit += int((ranks < 5).sum().item())
            hit10_hit += int((ranks < 10).sum().item())
            mrr_hit += float((1.0 / (ranks.float() + 1.0)).sum().item())
            hit1_all += int((ranks < 1).sum().item())
            hit5_all += int((ranks < 5).sum().item())
            hit10_all += int((ranks < 10).sum().item())
            mrr_all += float((1.0 / (ranks.float() + 1.0)).sum().item())

        candidate_miss_count += int((~valid_rows).sum().item())

    return Metrics(
        loss=total_loss / max(1, total_rows),
        rank_loss=total_rank_loss / max(1, valid_rows_total),
        coarse_loss=total_coarse_loss / max(1, valid_rows_total),
        region_loss=total_region_loss / max(1, total_rows),
        hit1_all=hit1_all / max(1, total_rows),
        hit5_all=hit5_all / max(1, total_rows),
        hit10_all=hit10_all / max(1, total_rows),
        mrr_all=mrr_all / max(1, total_rows),
        hit1_hitonly=hit1_hit / max(1, valid_rows_total),
        hit5_hitonly=hit5_hit / max(1, valid_rows_total),
        hit10_hitonly=hit10_hit / max(1, valid_rows_total),
        mrr_hitonly=mrr_hit / max(1, valid_rows_total),
        region_acc=region_correct / max(1, total_rows),
        candidate_miss_count=candidate_miss_count,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_bundle = torch.load(args.poi_embeddings, map_location="cpu")
    gcn_embeddings = emb_bundle["embeddings"].float()
    idx2poi = list(emb_bundle["idx2poi"])
    poi2idx = {str(k): int(v) for k, v in emb_bundle["poi2idx"].items()}

    catalog = build_catalog(args.catalog_csv, idx2poi)
    region_catalog = build_region_catalog(catalog["poi_lat"], catalog["poi_lon"], args.region_grid_km)
    user2idx, user_center = build_user_features(args.train_csv)

    train_df = load_checkin_frame(args.train_csv, "train")
    val_df = load_checkin_frame(args.val_csv, "val")
    train_samples, train_stats = build_laststep_samples(train_df, train_df, history_limit=args.history_limit, require_history=True)
    val_history_df = concat_history_frames([train_df, val_df])
    val_samples, val_stats = build_laststep_samples(val_df, val_history_df, history_limit=args.history_limit, require_history=True)

    train_context = build_laststep_context_map(train_samples, poi2idx, catalog["cat2id"], user2idx, region_catalog["poi_region_input_ids"])
    val_context = build_laststep_context_map(val_samples, poi2idx, catalog["cat2id"], user2idx, region_catalog["poi_region_input_ids"])

    train_ds = RerankDatasetLastStep(
        args.train_candidate_dir,
        train_context,
        catalog["poi_cat_ids"],
        catalog["poi_lat"],
        catalog["poi_lon"],
        poi2idx,
        user_center,
        region_catalog["poi_region_input_ids"],
        keep_only_hits=True,
        max_samples=args.max_train_samples,
        disabled_recall_features=args.disable_recall_features,
        enable_history_candidate_features=args.enable_history_candidate_features,
    )
    val_ds = RerankDatasetLastStep(
        args.val_candidate_dir,
        val_context,
        catalog["poi_cat_ids"],
        catalog["poi_lat"],
        catalog["poi_lon"],
        poi2idx,
        user_center,
        region_catalog["poi_region_input_ids"],
        keep_only_hits=False,
        max_samples=args.max_val_samples,
        disabled_recall_features=args.disable_recall_features,
        enable_history_candidate_features=args.enable_history_candidate_features,
    )
    if args.recall_feature_dim < 0:
        args.recall_feature_dim = len(train_ds.recall_feature_names)
    history_candidate_feature_dim = len(HISTORY_CANDIDATE_FEATURE_NAMES) if args.enable_history_candidate_features else 0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_laststep)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_laststep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RerankTransformerLastStep(
        num_pois=len(idx2poi),
        num_cats=len(catalog["cat2id"]),
        num_users=len(user2idx),
        num_regions=region_catalog["num_regions"],
        gcn_embeddings=gcn_embeddings,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_gcn_init=args.freeze_gcn_init,
        random_init_poi_emb=args.random_init_poi_emb,
        disable_candidate_gcn_rank=args.disable_candidate_gcn_rank,
        disable_candidate_score=args.disable_candidate_score,
        disable_candidate_cat_rank=args.disable_candidate_cat_rank,
        disable_same_cat_prev=args.disable_same_cat_prev,
        disable_distance_features=args.disable_distance_features,
        disable_user_features=args.disable_user_features,
        disable_time_features=args.disable_time_features,
        disable_history=args.disable_history,
        enable_candidate_history_attention=args.enable_candidate_history_attention,
        enable_history_attention_recency_bias=args.enable_history_attention_recency_bias,
        enable_history_attention_category_bias=args.enable_history_attention_category_bias,
        enable_history_attention_time_bias=args.enable_history_attention_time_bias,
        enable_history_attention_dow_bias=args.enable_history_attention_dow_bias,
        history_attention_bias_init=args.history_attention_bias_init,
        disable_candidate_self_attention=args.disable_candidate_self_attention,
        single_query_token=args.single_query_token,
        history_summary_len=args.history_summary_len,
        recall_feature_dim=args.recall_feature_dim,
        history_candidate_feature_dim=history_candidate_feature_dim,
        summary_recency_mode=args.summary_recency_mode,
    ).to(device)
    if args.init_model_ckpt:
        init_ckpt = torch.load(args.init_model_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(init_ckpt["model_state"], strict=False)
        print(
            json.dumps(
                {
                    "init_model_ckpt": args.init_model_ckpt,
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                ensure_ascii=False,
            )
        )
    optimizer_params = [param for param in model.parameters() if param.requires_grad]
    if not optimizer_params:
        raise ValueError("No trainable parameters. Check freeze settings.")
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_metric_score = -1.0
    metric_checkpoint_specs = {
        "hit1": ("val_hit1_all", "best_hit1.pt"),
        "hit5": ("val_hit5_all", "best_hit5.pt"),
        "hit10": ("val_hit10_all", "best_hit10.pt"),
        "mrr": ("val_mrr_all", "best_mrr.pt"),
    }
    metric_best = {
        name: {"metric": metric_key, "score": -1.0, "epoch": 0, "checkpoint": filename}
        for name, (metric_key, filename) in metric_checkpoint_specs.items()
    }
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_rank_loss": val_metrics.rank_loss,
            "val_coarse_loss": val_metrics.coarse_loss,
            "val_region_loss": val_metrics.region_loss,
            "val_hit1_all": val_metrics.hit1_all,
            "val_hit5_all": val_metrics.hit5_all,
            "val_hit10_all": val_metrics.hit10_all,
            "val_mrr_all": val_metrics.mrr_all,
            "val_hit1_hitonly": val_metrics.hit1_hitonly,
            "val_hit5_hitonly": val_metrics.hit5_hitonly,
            "val_hit10_hitonly": val_metrics.hit10_hitonly,
            "val_mrr_hitonly": val_metrics.mrr_hitonly,
            "val_region_acc": val_metrics.region_acc,
            "val_candidate_miss_count": val_metrics.candidate_miss_count,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if args.best_metric not in row:
            raise ValueError(f"unknown best_metric={args.best_metric}; available metrics: {sorted(row.keys())}")
        current_metric = float(row[args.best_metric])
        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": row,
            "user2idx": user2idx,
            "num_regions": region_catalog["num_regions"],
            "region_grid_km": region_catalog["grid_km"],
            "poi_region_input_ids": region_catalog["poi_region_input_ids"],
        }
        torch.save(ckpt, out_dir / "last.pt")
        improved_primary = current_metric > best_metric_score
        if improved_primary:
            best_metric_score = current_metric
            bad_epochs = 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            bad_epochs += 1

        for name, (metric_key, filename) in metric_checkpoint_specs.items():
            metric_value = float(row[metric_key])
            if metric_value > float(metric_best[name]["score"]):
                metric_best[name] = {
                    "metric": metric_key,
                    "score": metric_value,
                    "epoch": epoch,
                    "checkpoint": filename,
                }
                torch.save(ckpt, out_dir / filename)

        with (out_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metric_best, f, indent=2, ensure_ascii=False)

        if not improved_primary:
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                print(
                    json.dumps(
                        {
                            "early_stop": True,
                            "epoch": epoch,
                            "best_metric": args.best_metric,
                            "best_metric_score": best_metric_score,
                            "patience": args.early_stop_patience,
                        },
                        ensure_ascii=False,
                    )
                )
                break

    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with (out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "num_users": len(user2idx),
                "num_regions": region_catalog["num_regions"],
                "num_train_samples": len(train_ds),
                "num_val_samples": len(val_ds),
                "num_val_candidate_miss_samples": int(count_candidate_miss(val_ds)),
                "best_metric": args.best_metric,
                "best_metric_score": best_metric_score,
                "multi_metric_best": metric_best,
                "candidate_recall_feature_names": train_ds.recall_feature_names,
                "history_candidate_feature_names": HISTORY_CANDIDATE_FEATURE_NAMES
                if args.enable_history_candidate_features
                else [],
                "train_sample_stats": train_stats,
                "val_sample_stats": val_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
