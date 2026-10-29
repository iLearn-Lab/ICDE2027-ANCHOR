import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from candidate_pool_utils import (  # noqa: E402
    build_catalog,
    build_user_pref_context,
    cat_rank_score,
    load_llm_topk,
    load_samples,
)
from laststep_data_utils import save_json  # noqa: E402


FEATURE_NAMES = [
    "v3_total_score",
    "llm_cat_score",
    "user_cat_pref_score",
    "recent_cat_pref_score",
    "time_cat_pref_score",
    "history_repeat_score",
    "history_recency_score",
    "category_gcn_score",
    "no_cat_gcn_score",
    "temporal_pop_score",
    "spatial_score",
    "global_pop_score",
    "is_llm_category",
    "is_user_pref_category",
    "is_history_exact",
    "is_category_gcn",
    "is_no_cat_gcn",
    "is_temporal_fallback",
    "is_spatial_fallback",
    "user_prev_cat_transition_score",
    "user_prev_poi_transition_score",
    "user_recent_cat_transition_score",
    "user_recent_poi_transition_score",
    "global_prev_cat_transition_score",
    "global_prev_poi_transition_score",
    "is_user_transition_candidate",
    "is_global_transition_candidate",
]

FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


SCORE_FEATURES = [
    "llm_cat_score",
    "user_cat_pref_score",
    "recent_cat_pref_score",
    "time_cat_pref_score",
    "history_repeat_score",
    "history_recency_score",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Build candidates directly from llm/history/user_pref sources only.")
    ap.add_argument("--poi_embeddings", required=True)
    ap.add_argument("--catalog_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--history_csvs", required=True)
    ap.add_argument("--topk_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--history_limit", type=int, default=50)
    ap.add_argument("--split_name", default="")
    ap.add_argument("--llm_topk", type=int, default=10)
    ap.add_argument("--candidate_size", type=int, default=2000)
    ap.add_argument("--llm_category_quota", type=int, default=800)
    ap.add_argument("--history_quota", type=int, default=100)
    ap.add_argument("--user_pref_category_quota", type=int, default=200)
    ap.add_argument("--user_pref_cat_topm", type=int, default=20)
    ap.add_argument("--enabled_sources", default="llm,history,user_pref")
    return ap.parse_args()


class FeatureAccumulator:
    def __init__(self):
        self.rows = {}

    def _row(self, poi_idx):
        poi_idx = int(poi_idx)
        if poi_idx not in self.rows:
            self.rows[poi_idx] = [0.0] * len(FEATURE_NAMES)
        return self.rows[poi_idx]

    def add(self, poi_idx, feature_name, value):
        if value == 0.0:
            return
        self._row(poi_idx)[FEATURE_INDEX[feature_name]] += float(value)

    def flag(self, poi_idx, feature_name):
        self._row(poi_idx)[FEATURE_INDEX[feature_name]] = 1.0

    def ranked_all(self):
        rows = []
        total_idx = FEATURE_INDEX["v3_total_score"]
        for poi_idx, feats in self.rows.items():
            total = sum(feats[FEATURE_INDEX[name]] for name in SCORE_FEATURES)
            feats[total_idx] = total
            rows.append((poi_idx, total, feats, 0))
        rows.sort(key=lambda x: (-x[1], x[0]))
        return rows


def has(feats, name):
    return feats[FEATURE_INDEX[name]] > 0.0


def build_quota_candidates(ranked_all, candidate_size, args, enabled_sources):
    selected = []
    selected_ids = set()
    quota_counts = {}

    def add_bucket(name, quota, predicate):
        added = 0
        for row in ranked_all:
            if len(selected) >= candidate_size or added >= quota:
                break
            poi_idx, _, feats, _ = row
            if poi_idx in selected_ids or not predicate(feats):
                continue
            selected.append(row)
            selected_ids.add(poi_idx)
            added += 1
        quota_counts[name] = added

    if "history" in enabled_sources:
        add_bucket("history", args.history_quota, lambda feats: has(feats, "is_history_exact"))
    if "llm" in enabled_sources:
        add_bucket("llm_category", args.llm_category_quota, lambda feats: has(feats, "is_llm_category"))
    if "user_pref" in enabled_sources:
        add_bucket(
            "user_pref_category",
            args.user_pref_category_quota,
            lambda feats: has(feats, "is_user_pref_category") and not has(feats, "is_llm_category"),
        )

    fill_added = 0
    for row in ranked_all:
        if len(selected) >= candidate_size:
            break
        poi_idx = row[0]
        if poi_idx in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(poi_idx)
        fill_added += 1
    quota_counts["three_source_fill"] = fill_added
    return selected[:candidate_size], quota_counts


def build_candidates_for_sample(sample, llm, args, catalog, poi2idx, poi_catnames, enabled_sources):
    allowed_cats = [] if llm is None else list(llm["topk_catnames"][: args.llm_topk])
    allowed_set = set(allowed_cats)
    cat_rank_map = {cat: rank for rank, cat in enumerate(allowed_cats, start=1)}
    pref_ctx = build_user_pref_context(sample, poi2idx, poi_catnames)
    user_pref_cats = []
    if "user_pref" in enabled_sources:
        user_pref_cats = [
            cat
            for cat, _ in sorted(pref_ctx["combined_cat_pref"].items(), key=lambda x: (-x[1], x[0]))[
                : args.user_pref_cat_topm
            ]
        ]
    expanded_cats = list(dict.fromkeys((allowed_cats if "llm" in enabled_sources else []) + user_pref_cats))

    acc = FeatureAccumulator()
    for cat in expanded_cats:
        user_only_discount = 0.85 if cat not in allowed_set else 1.0
        llm_score = 4.0 * cat_rank_score(cat_rank_map[cat]) if ("llm" in enabled_sources and cat in allowed_set) else 0.0
        user_cat_score = (
            2.6 * pref_ctx["cat_pref"].get(cat, 0.0) * user_only_discount if "user_pref" in enabled_sources else 0.0
        )
        recent_cat_score = (
            2.0 * pref_ctx["recent_cat_pref"].get(cat, 0.0) * user_only_discount
            if "user_pref" in enabled_sources
            else 0.0
        )
        time_cat_score = (
            1.7 * pref_ctx["time_cat_pref"].get(cat, 0.0) * user_only_discount
            if "user_pref" in enabled_sources
            else 0.0
        )

        for poi_idx in catalog["cat_to_poi_indices"].get(cat, []):
            acc.add(poi_idx, "llm_cat_score", llm_score)
            acc.add(poi_idx, "user_cat_pref_score", user_cat_score)
            acc.add(poi_idx, "recent_cat_pref_score", recent_cat_score)
            acc.add(poi_idx, "time_cat_pref_score", time_cat_score)
            if "llm" in enabled_sources and cat in allowed_set:
                acc.flag(poi_idx, "is_llm_category")
            if "user_pref" in enabled_sources and cat in user_pref_cats:
                acc.flag(poi_idx, "is_user_pref_category")

    if "history" in enabled_sources:
        for poi_idx, count in pref_ctx["hist_counts"].items():
            acc.add(poi_idx, "history_repeat_score", 5.0 * math.log1p(count))
            acc.flag(poi_idx, "is_history_exact")

        for recency, row in enumerate(reversed(sample.history_rows), start=1):
            poi_idx = poi2idx.get(str(row["POI_id"]))
            if poi_idx is None:
                continue
            acc.add(poi_idx, "history_recency_score", 3.5 / math.log2(recency + 1.0))

    ranked_all = acc.ranked_all()
    candidates, quota_counts = build_quota_candidates(ranked_all, args.candidate_size, args, enabled_sources)
    return candidates, allowed_cats, quota_counts


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled_sources = [x.strip() for x in args.enabled_sources.split(",") if x.strip()]
    allowed_enabled = {"llm", "history", "user_pref"}
    unknown = sorted(set(enabled_sources) - allowed_enabled)
    if unknown:
        raise ValueError(f"Unknown enabled_sources: {unknown}")

    samples, sample_stats = load_samples(args.target_csv, args.history_csvs, args.history_limit)
    emb_bundle = torch.load(args.poi_embeddings, map_location="cpu")
    poi2idx = {str(k): int(v) for k, v in emb_bundle["poi2idx"].items()}
    idx2poi = list(emb_bundle["idx2poi"])
    catalog = build_catalog(args.catalog_csv, poi2idx, idx2poi)
    poi_catnames = catalog["poi_catnames"]
    llm_records = load_llm_topk(args.topk_jsonl)

    n = len(samples)
    candidate_ids = torch.full((n, args.candidate_size), -1, dtype=torch.int32)
    candidate_cat_ranks = torch.zeros((n, args.candidate_size), dtype=torch.int16)
    candidate_gcn_ranks = torch.zeros((n, args.candidate_size), dtype=torch.int16)
    candidate_scores = torch.full((n, args.candidate_size), float("-inf"), dtype=torch.float32)
    candidate_recall_features = torch.zeros((n, args.candidate_size, len(FEATURE_NAMES)), dtype=torch.float32)
    label_indices = torch.full((n,), -1, dtype=torch.int32)
    candidate_counts = torch.zeros((n,), dtype=torch.int32)

    num_missing_llm = 0
    num_missing_target_poi = 0
    num_true_cat_not_in_llm_topk = 0
    num_true_poi_in_candidates = 0
    quota_totals = Counter()
    hit_at = {1: 0, 3: 0, 5: 0, 10: 0, 100: 0, 500: 0, 1000: 0, 2000: 0}

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(samples, desc=f"three_source_{args.split_name}")):
            key = (sample.trajectory_id, sample.target_idx_in_traj)
            llm = llm_records.get(key)
            if llm is None:
                num_missing_llm += 1

            target_idx = poi2idx.get(sample.target_poi_id)
            if target_idx is None:
                num_missing_target_poi += 1

            candidates, allowed_cats, quota_counts = build_candidates_for_sample(
                sample, llm, args, catalog, poi2idx, poi_catnames, enabled_sources
            )
            quota_totals.update(quota_counts)
            true_cat_hit = str(sample.target_catname) in set(allowed_cats)
            if not true_cat_hit:
                num_true_cat_not_in_llm_topk += 1

            filled = len(candidates)
            label_index = -1
            if filled:
                candidate_ids[i, :filled] = torch.tensor([x[0] for x in candidates], dtype=torch.int32)
                candidate_scores[i, :filled] = torch.tensor([x[1] for x in candidates], dtype=torch.float32)
                candidate_recall_features[i, :filled] = torch.tensor([x[2] for x in candidates], dtype=torch.float32)
                ranks = []
                for poi_idx, _, _, _ in candidates:
                    cat = poi_catnames[poi_idx]
                    ranks.append(allowed_cats.index(cat) + 1 if ("llm" in enabled_sources and cat in allowed_cats) else 0)
                candidate_cat_ranks[i, :filled] = torch.tensor(ranks, dtype=torch.int16)

            if target_idx is not None:
                for pos, (poi_idx, _, _, _) in enumerate(candidates):
                    if poi_idx == target_idx:
                        label_index = pos
                        num_true_poi_in_candidates += 1
                        for k in hit_at:
                            if pos < k:
                                hit_at[k] += 1
                        break

            label_indices[i] = label_index
            candidate_counts[i] = filled
            prev_poi = str(sample.history_rows[-1]["POI_id"]) if sample.history_rows else ""
            f.write(
                json.dumps(
                    {
                        "trajectory_id": sample.trajectory_id,
                        "target_idx_in_traj": sample.target_idx_in_traj,
                        "target_utc_time": sample.target_utc_time,
                        "user_id": sample.user_id,
                        "prev_poi_id": prev_poi,
                        "target_poi_id": sample.target_poi_id,
                        "true_catname": sample.target_catname,
                        "history_len": len(sample.history_rows),
                        "llm_missing": llm is None,
                        "llm_topk_used": args.llm_topk,
                        "candidate_size": args.candidate_size,
                        "num_candidates": filled,
                        "label_index": int(label_index),
                        "topk_catnames": allowed_cats,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    torch.save(
        {
            "candidate_ids": candidate_ids,
            "candidate_cat_ranks": candidate_cat_ranks,
            "candidate_gcn_ranks": candidate_gcn_ranks,
            "candidate_scores": candidate_scores,
            "candidate_recall_features": candidate_recall_features,
            "candidate_recall_feature_names": FEATURE_NAMES,
            "label_indices": label_indices,
            "candidate_counts": candidate_counts,
        },
        out_dir / "candidate_tensors.pt",
    )

    known_target_den = n - num_missing_target_poi
    save_json(out_dir / "poi_vocab.json", {"split_name": args.split_name, "num_pois": len(idx2poi), "idx2poi": idx2poi})
    save_json(
        out_dir / "build_stats.json",
        {
            "strategy": "direct_three_source_llm_history_user_pref",
            "enabled_sources": enabled_sources,
            "split_name": args.split_name,
            "history_limit": args.history_limit,
            "llm_topk": args.llm_topk,
            "candidate_size": args.candidate_size,
            "llm_category_quota": args.llm_category_quota,
            "history_quota": args.history_quota,
            "user_pref_category_quota": args.user_pref_category_quota,
            "user_pref_cat_topm": args.user_pref_cat_topm,
            "sample_stats": sample_stats,
            "candidate_recall_feature_names": FEATURE_NAMES,
            "kept_score_features": SCORE_FEATURES,
            "num_samples": n,
            "num_missing_llm_records": num_missing_llm,
            "num_missing_target_poi": num_missing_target_poi,
            "num_true_cat_not_in_llm_topk": num_true_cat_not_in_llm_topk,
            "num_true_poi_in_candidates": num_true_poi_in_candidates,
            "quota_selected_totals": dict(quota_totals),
            "avg_candidate_count": float(candidate_counts.float().mean().item()) if n else 0.0,
            "candidate_hit_rate_all": (num_true_poi_in_candidates / n) if n else 0.0,
            "candidate_hit_rate_known_target": (num_true_poi_in_candidates / known_target_den)
            if known_target_den > 0
            else 0.0,
            "rule_acc_at_1_all": (hit_at[1] / n) if n else 0.0,
            "rule_acc_at_3_all": (hit_at[3] / n) if n else 0.0,
            "rule_acc_at_5_all": (hit_at[5] / n) if n else 0.0,
            "rule_acc_at_10_all": (hit_at[10] / n) if n else 0.0,
            "rule_acc_at_100_all": (hit_at[100] / n) if n else 0.0,
            "rule_acc_at_500_all": (hit_at[500] / n) if n else 0.0,
            "rule_acc_at_1000_all": (hit_at[1000] / n) if n else 0.0,
            "rule_acc_at_2000_all": (hit_at[2000] / n) if n else 0.0,
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "num_samples": n, "hit_rate": num_true_poi_in_candidates / max(1, n)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
