import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from laststep_data_utils import build_laststep_samples, concat_history_frames, load_checkin_frame
from reranker_laststep_core import (
    HISTORY_CANDIDATE_FEATURE_NAMES,
    RerankDatasetLastStep,
    RerankTransformerLastStep,
    build_catalog,
    build_laststep_context_map,
    build_user_features,
    collate_fn_laststep,
    compute_loss,
    count_candidate_miss,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", default="")
    ap.add_argument("--test_candidate_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--catalog_csv", required=True)
    ap.add_argument("--poi_embeddings", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--disable_recall_features", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_jsonl", default="")
    ap.add_argument("--out_csv", default="")
    return ap.parse_args()


@torch.no_grad()
def evaluate_with_cases(model, loader, device, args, dataset):
    model.eval()
    total_rows = total_loss = total_rank_loss = total_coarse_loss = total_region_loss = 0.0
    valid_rows_total = 0
    hit1_all = hit5_all = hit10_all = 0
    mrr_all = 0.0
    hit1_hit = hit5_hit = hit10_hit = 0
    mrr_hit = 0.0
    region_correct = 0
    cases = []

    for batch in loader:
        batch_cpu = batch
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        loss, parts = compute_loss(outputs, batch, args)
        logits = outputs["rank_logits"]
        labels = batch["label_index"]
        valid_rows = labels >= 0
        rows = logits.size(0)
        total_rows += rows
        total_loss += float(loss.item()) * rows
        total_region_loss += float(parts["region"]) * rows

        ranks_map = {}
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
            valid_positions = torch.nonzero(valid_rows, as_tuple=False).squeeze(1).tolist()
            for pos, rank in zip(valid_positions, ranks.tolist()):
                ranks_map[int(pos)] = int(rank) + 1

        region_pred = torch.argmax(outputs["region_logits"], dim=1)
        region_correct += int((region_pred == batch["target_region_id"]).sum().item())

        for local_idx in range(rows):
            row_idx = int(batch_cpu["sample_index"][local_idx].item())
            sample = dataset.samples[row_idx]
            label_index = int(batch_cpu["label_index"][local_idx].item())
            rank = ranks_map.get(local_idx)
            cases.append(
                {
                    "trajectory_id": str(sample["trajectory_id"]),
                    "target_idx_in_traj": int(sample["target_idx_in_traj"]),
                    "user_id": str(sample["user_id"]),
                    "target_poi_id": str(sample["target_poi_id"]),
                    "label_in_candidate_pool": bool(label_index >= 0),
                    "label_index": label_index,
                    "pred_rank": rank,
                    "is_hit1": bool(rank is not None and rank <= 1),
                    "is_hit5": bool(rank is not None and rank <= 5),
                    "is_hit10": bool(rank is not None and rank <= 10),
                }
            )

    return {
        "loss": total_loss / max(1, total_rows),
        "rank_loss": total_rank_loss / max(1, valid_rows_total),
        "coarse_loss": total_coarse_loss / max(1, valid_rows_total),
        "region_loss": total_region_loss / max(1, total_rows),
        "hit1_all": hit1_all / max(1, total_rows),
        "hit5_all": hit5_all / max(1, total_rows),
        "hit10_all": hit10_all / max(1, total_rows),
        "mrr_all": mrr_all / max(1, total_rows),
        "hit1_hitonly": hit1_hit / max(1, valid_rows_total),
        "hit5_hitonly": hit5_hit / max(1, valid_rows_total),
        "hit10_hitonly": hit10_hit / max(1, valid_rows_total),
        "mrr_hitonly": mrr_hit / max(1, valid_rows_total),
        "region_acc": region_correct / max(1, total_rows),
        "cases": cases,
    }


def build_seen_user_poi_metrics(cases, train_csv):
    train = pd.read_csv(train_csv, usecols=["user_id", "POI_id"], dtype=str)
    seen_users = set(train["user_id"].astype(str))
    seen_pois = set(train["POI_id"].astype(str))
    seen_cases = [
        case
        for case in cases
        if str(case["user_id"]) in seen_users and str(case["target_poi_id"]) in seen_pois
    ]

    def mean_bool(rows, key):
        return float(sum(1 for row in rows if bool(row[key])) / max(1, len(rows)))

    def mrr(rows):
        total = 0.0
        for row in rows:
            rank = row["pred_rank"]
            if rank:
                total += 1.0 / float(rank)
        return float(total / max(1, len(rows)))

    return {
        "n": int(len(seen_cases)),
        "filtered_out": int(len(cases) - len(seen_cases)),
        "candidate_hit_rate": mean_bool(seen_cases, "label_in_candidate_pool"),
        "hit1": mean_bool(seen_cases, "is_hit1"),
        "hit5": mean_bool(seen_cases, "is_hit5"),
        "hit10": mean_bool(seen_cases, "is_hit10"),
        "mrr": mrr(seen_cases),
    }


def main():
    args = parse_args()
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    train_args = ckpt["args"]

    emb_bundle = torch.load(args.poi_embeddings, map_location="cpu")
    gcn_embeddings = emb_bundle["embeddings"].float()
    idx2poi = list(emb_bundle["idx2poi"])
    poi2idx = {str(k): int(v) for k, v in emb_bundle["poi2idx"].items()}

    catalog = build_catalog(args.catalog_csv, idx2poi)
    user2idx, user_center = build_user_features(args.train_csv)
    poi_region_input_ids = ckpt["poi_region_input_ids"].long()
    num_regions = int(ckpt["num_regions"])

    train_df = load_checkin_frame(args.train_csv, "train")
    history_frames = [train_df]
    if args.val_csv:
        history_frames.append(load_checkin_frame(args.val_csv, "val"))
    test_df = load_checkin_frame(args.test_csv, "test")
    history_frames.append(test_df)
    history_df = concat_history_frames(history_frames)

    test_samples, sample_stats = build_laststep_samples(
        test_df,
        history_df,
        history_limit=int(train_args["history_limit"]),
        require_history=True,
    )
    test_context = build_laststep_context_map(test_samples, poi2idx, catalog["cat2id"], user2idx, poi_region_input_ids)
    test_ds = RerankDatasetLastStep(
        args.test_candidate_dir,
        test_context,
        catalog["poi_cat_ids"],
        catalog["poi_lat"],
        catalog["poi_lon"],
        poi2idx,
        user_center,
        poi_region_input_ids,
        keep_only_hits=False,
        max_samples=args.max_test_samples,
        disabled_recall_features=args.disable_recall_features,
        enable_history_candidate_features=bool(train_args.get("enable_history_candidate_features", False)),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_laststep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_model_kwargs = {
        "num_pois": len(idx2poi),
        "num_cats": len(catalog["cat2id"]),
        "num_users": len(user2idx),
        "num_regions": num_regions,
        "gcn_embeddings": gcn_embeddings,
        "hidden_dim": int(train_args["hidden_dim"]),
        "num_layers": int(train_args["num_layers"]),
        "num_heads": int(train_args["num_heads"]),
        "dropout": float(train_args["dropout"]),
        "freeze_gcn_init": bool(train_args.get("freeze_gcn_init", False)),
        "random_init_poi_emb": bool(train_args.get("random_init_poi_emb", False)),
        "disable_candidate_gcn_rank": bool(train_args.get("disable_candidate_gcn_rank", False)),
        "disable_candidate_score": bool(train_args.get("disable_candidate_score", False)),
        "disable_candidate_cat_rank": bool(train_args.get("disable_candidate_cat_rank", False)),
        "disable_same_cat_prev": bool(train_args.get("disable_same_cat_prev", False)),
        "disable_distance_features": bool(train_args.get("disable_distance_features", False)),
        "disable_user_features": bool(train_args.get("disable_user_features", False)),
        "disable_time_features": bool(train_args.get("disable_time_features", False)),
        "disable_history": bool(train_args.get("disable_history", False)),
        "enable_candidate_history_attention": bool(train_args.get("enable_candidate_history_attention", False)),
        "enable_history_attention_recency_bias": bool(train_args.get("enable_history_attention_recency_bias", False)),
        "enable_history_attention_category_bias": bool(train_args.get("enable_history_attention_category_bias", False)),
        "enable_history_attention_time_bias": bool(train_args.get("enable_history_attention_time_bias", False)),
        "enable_history_attention_dow_bias": bool(train_args.get("enable_history_attention_dow_bias", False)),
        "history_attention_bias_init": float(train_args.get("history_attention_bias_init", 0.1)),
        "single_query_token": bool(train_args.get("single_query_token", False)),
        "history_summary_len": int(train_args.get("history_summary_len", train_args.get("history_limit", 50))),
        "recall_feature_dim": int(train_args.get("recall_feature_dim", 0)),
        "history_candidate_feature_dim": len(HISTORY_CANDIDATE_FEATURE_NAMES)
        if bool(train_args.get("enable_history_candidate_features", False))
        else 0,
        "summary_recency_mode": str(train_args.get("summary_recency_mode", "legacy")),
    }
    model = RerankTransformerLastStep(
        **common_model_kwargs,
        disable_candidate_self_attention=bool(train_args.get("disable_candidate_self_attention", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)

    metric_args = argparse.Namespace(**train_args)
    metrics = evaluate_with_cases(model, test_loader, device, metric_args, test_ds)
    cases = metrics.pop("cases")
    seen_metrics = build_seen_user_poi_metrics(cases, args.train_csv)
    result = {
        "model_ckpt": args.model_ckpt,
        "test_candidate_dir": args.test_candidate_dir,
        "test_csv": args.test_csv,
        "num_test_samples": len(test_ds),
        "num_candidate_miss_samples": int(count_candidate_miss(test_ds)),
        "sample_stats": sample_stats,
        "test_loss": metrics["loss"],
        "test_rank_loss": metrics["rank_loss"],
        "test_coarse_loss": metrics["coarse_loss"],
        "test_region_loss": metrics["region_loss"],
        "test_hit1_all": metrics["hit1_all"],
        "test_hit5_all": metrics["hit5_all"],
        "test_hit10_all": metrics["hit10_all"],
        "test_mrr_all": metrics["mrr_all"],
        "test_hit1_hitonly": metrics["hit1_hitonly"],
        "test_hit5_hitonly": metrics["hit5_hitonly"],
        "test_hit10_hitonly": metrics["hit10_hitonly"],
        "test_mrr_hitonly": metrics["mrr_hitonly"],
        "test_region_acc": metrics["region_acc"],
        "seen_user_poi": seen_metrics,
        "test_seen_n": seen_metrics["n"],
        "test_seen_filtered_out": seen_metrics["filtered_out"],
        "test_seen_candidate_hit_rate": seen_metrics["candidate_hit_rate"],
        "test_seen_hit1": seen_metrics["hit1"],
        "test_seen_hit5": seen_metrics["hit5"],
        "test_seen_hit10": seen_metrics["hit10"],
        "test_seen_mrr": seen_metrics["mrr"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cases).to_csv(out_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
