import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from laststep_data_utils import build_laststep_samples


HISTORY_CANDIDATE_FEATURE_NAMES = [
    "hist_poi_seen",
    "hist_poi_log_count",
    "hist_poi_recency_score",
    "hist_poi_same_dow_log_count",
    "hist_poi_same_tbin_log_count",
    "hist_cat_seen",
    "hist_cat_log_count",
    "hist_cat_recency_score",
    "hist_cat_recent5_log_count",
    "hist_cat_recent10_log_count",
    "hist_cat_recent20_log_count",
    "hist_cat_same_dow_log_count",
    "hist_cat_same_tbin_log_count",
    "hist_cat_recent10_ratio",
]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def build_catalog(catalog_csv: str, idx2poi):
    df = pd.read_csv(catalog_csv, usecols=["POI_id", "POI_catname", "latitude", "longitude"])
    df["POI_id"] = df["POI_id"].astype(str)
    df["POI_catname"] = df["POI_catname"].astype(str)

    catnames = sorted(df["POI_catname"].unique().tolist())
    cat2id = {c: i + 1 for i, c in enumerate(catnames)}

    poi_meta = {}
    for row in df.itertuples(index=False):
        poi_id = str(row.POI_id)
        if poi_id not in poi_meta:
            poi_meta[poi_id] = {
                "cat_id": cat2id[str(row.POI_catname)],
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }

    num_pois = len(idx2poi)
    poi_cat_ids = torch.zeros(num_pois, dtype=torch.long)
    poi_lat = torch.zeros(num_pois, dtype=torch.float32)
    poi_lon = torch.zeros(num_pois, dtype=torch.float32)
    for idx, poi_id in enumerate(idx2poi):
        meta = poi_meta[str(poi_id)]
        poi_cat_ids[idx] = meta["cat_id"]
        poi_lat[idx] = meta["latitude"]
        poi_lon[idx] = meta["longitude"]

    return {
        "cat2id": cat2id,
        "poi_meta": poi_meta,
        "poi_cat_ids": poi_cat_ids,
        "poi_lat": poi_lat,
        "poi_lon": poi_lon,
    }


def build_user_features(train_csv: str):
    df = pd.read_csv(train_csv, usecols=["user_id", "latitude", "longitude"])
    df["user_id"] = df["user_id"].astype(str)

    user_ids = sorted(df["user_id"].unique().tolist())
    user2idx = {uid: i + 1 for i, uid in enumerate(user_ids)}
    user_center = {}
    for uid, g in df.groupby("user_id", sort=False):
        user_center[str(uid)] = (float(g["latitude"].mean()), float(g["longitude"].mean()))
    return user2idx, user_center


def build_region_catalog(poi_lat: torch.Tensor, poi_lon: torch.Tensor, grid_km: float):
    lat = poi_lat.float()
    lon = poi_lon.float()
    min_lat = float(lat.min().item())
    min_lon = float(lon.min().item())
    mean_lat = float(lat.mean().item())
    km_per_lat = 111.32
    km_per_lon = max(1e-3, 111.32 * math.cos(math.radians(mean_lat)))

    region_input_ids = torch.zeros(lat.size(0) + 1, dtype=torch.long)
    region2id = {}
    next_region_id = 1

    for poi_idx in range(lat.size(0)):
        y_bin = int(math.floor(((float(lat[poi_idx]) - min_lat) * km_per_lat) / grid_km))
        x_bin = int(math.floor(((float(lon[poi_idx]) - min_lon) * km_per_lon) / grid_km))
        key = (y_bin, x_bin)
        if key not in region2id:
            region2id[key] = next_region_id
            next_region_id += 1
        region_input_ids[poi_idx + 1] = region2id[key]

    return {
        "poi_region_input_ids": region_input_ids,
        "num_regions": next_region_id - 1,
        "grid_km": grid_km,
    }


def build_laststep_context_map(samples, poi2idx: dict, cat2id: dict, user2idx: dict, poi_region_input_ids: torch.Tensor):
    context_map = {}
    for sample in samples:
        hist_rows = sample.history_rows
        if not hist_rows:
            continue

        hist_poi = []
        hist_cat = []
        hist_dow = []
        hist_tbin = []
        hist_recency = []
        hist_lat = []
        hist_lon = []
        for pos, r in enumerate(hist_rows):
            poi_idx = poi2idx.get(str(r["POI_id"]))
            hist_poi.append((poi_idx + 1) if poi_idx is not None else 0)
            hist_cat.append(cat2id.get(str(r["POI_catname"]), 0))
            hist_dow.append(int(r["day_of_week"]) + 1)
            hist_tbin.append(min(23, max(0, int(float(r["norm_in_day_time"]) * 24))) + 1)
            hist_recency.append(len(hist_rows) - pos)
            hist_lat.append(float(r["latitude"]))
            hist_lon.append(float(r["longitude"]))

        prev = hist_rows[-1]
        prev_ts = pd.to_datetime(prev["local_time"], errors="coerce")
        cur_ts = pd.to_datetime(sample.target_local_time, errors="coerce")
        delta_min = 0.0 if pd.isna(prev_ts) or pd.isna(cur_ts) else max(0.0, (cur_ts - prev_ts).total_seconds() / 60.0)
        if delta_min <= 10:
            delta_bucket = 1
        elif delta_min <= 30:
            delta_bucket = 2
        elif delta_min <= 60:
            delta_bucket = 3
        elif delta_min <= 360:
            delta_bucket = 4
        else:
            delta_bucket = 5

        prev_poi_idx = poi2idx.get(str(prev["POI_id"]), -1)
        target_poi_idx = poi2idx.get(str(sample.target_poi_id), -1)
        target_region_id = 0 if target_poi_idx < 0 else int(poi_region_input_ids[target_poi_idx + 1])
        context_map[(sample.trajectory_id, sample.target_idx_in_traj)] = {
            "user_idx": user2idx.get(sample.user_id, 0),
            "user_id": sample.user_id,
            "history_poi_ids": hist_poi,
            "history_cat_ids": hist_cat,
            "history_dow_ids": hist_dow,
            "history_tbin_ids": hist_tbin,
            "history_recency_ids": hist_recency,
            "history_latitudes": hist_lat,
            "history_longitudes": hist_lon,
            "target_dow_id": int(sample.target_day_of_week) + 1,
            "target_tbin_id": min(23, max(0, int(float(sample.target_norm_in_day_time) * 24))) + 1,
            "delta_bucket": delta_bucket,
            "delta_minutes": math.log1p(delta_min),
            "prev_poi_idx": prev_poi_idx,
            "prev_cat_id": cat2id.get(str(prev["POI_catname"]), 0),
            "target_region_id": target_region_id,
        }
    return context_map


class RerankDatasetLastStep(Dataset):
    def __init__(
        self,
        candidate_dir: str,
        context_map: dict,
        poi_cat_ids: torch.Tensor,
        poi_lat: torch.Tensor,
        poi_lon: torch.Tensor,
        poi2idx: dict,
        user_center: dict,
        poi_region_input_ids: torch.Tensor,
        keep_only_hits: bool = False,
        max_samples: int = 0,
        disabled_recall_features: str = "",
        enable_history_candidate_features: bool = False,
    ):
        self.samples = []
        with open(Path(candidate_dir) / "samples.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        bundle = torch.load(Path(candidate_dir) / "candidate_tensors.pt", map_location="cpu")
        self.candidate_ids = bundle["candidate_ids"].long()
        self.candidate_cat_ranks = bundle["candidate_cat_ranks"].long()
        self.candidate_gcn_ranks = bundle["candidate_gcn_ranks"].long()
        self.candidate_scores = bundle["candidate_scores"].float()
        self.label_indices = bundle["label_indices"].long()
        self.candidate_counts = bundle["candidate_counts"].long()
        self.recall_feature_names = list(bundle.get("candidate_recall_feature_names", []))
        if "candidate_recall_features" in bundle:
            self.candidate_recall_features = bundle["candidate_recall_features"].float()
        else:
            num_rows, candidate_size = self.candidate_ids.shape
            self.candidate_recall_features = torch.zeros((num_rows, candidate_size, 0), dtype=torch.float32)
        self.disabled_recall_feature_names = self._expand_disabled_recall_features(disabled_recall_features)
        if self.disabled_recall_feature_names and self.candidate_recall_features.size(-1) > 0:
            disabled_indices = [
                self.recall_feature_names.index(name)
                for name in self.disabled_recall_feature_names
                if name in self.recall_feature_names
            ]
            if disabled_indices:
                self.candidate_recall_features[:, :, disabled_indices] = 0.0

        self.context_map = context_map
        self.poi_cat_ids = poi_cat_ids
        self.poi_lat = poi_lat
        self.poi_lon = poi_lon
        self.poi2idx = poi2idx
        self.user_center = user_center
        self.poi_region_input_ids = poi_region_input_ids
        self.enable_history_candidate_features = bool(enable_history_candidate_features)
        self.history_candidate_feature_names = list(HISTORY_CANDIDATE_FEATURE_NAMES)

        self.indices = []
        for i, sample in enumerate(self.samples):
            key = (str(sample["trajectory_id"]), int(sample["target_idx_in_traj"]))
            if key not in self.context_map:
                continue
            if keep_only_hits and int(self.label_indices[i]) < 0:
                continue
            self.indices.append(i)
            if max_samples > 0 and len(self.indices) >= max_samples:
                break

    @staticmethod
    def _build_history_candidate_features(ctx: dict, cand_raw_ids: torch.Tensor, cand_cat_ids: torch.Tensor):
        target_dow_id = int(ctx["target_dow_id"])
        target_tbin_id = int(ctx["target_tbin_id"])
        poi_counts = {}
        poi_last_recency = {}
        poi_same_dow = {}
        poi_same_tbin = {}
        cat_counts = {}
        cat_last_recency = {}
        cat_recent5 = {}
        cat_recent10 = {}
        cat_recent20 = {}
        cat_same_dow = {}
        cat_same_tbin = {}

        hist_pois = ctx["history_poi_ids"]
        hist_cats = ctx["history_cat_ids"]
        hist_dows = ctx["history_dow_ids"]
        hist_tbins = ctx["history_tbin_ids"]
        hist_recencies = ctx["history_recency_ids"]
        for poi_input_id, cat_id, dow_id, tbin_id, recency in zip(
            hist_pois,
            hist_cats,
            hist_dows,
            hist_tbins,
            hist_recencies,
        ):
            poi_raw_id = int(poi_input_id) - 1
            cat_id = int(cat_id)
            recency = int(recency)
            if poi_raw_id >= 0:
                poi_counts[poi_raw_id] = poi_counts.get(poi_raw_id, 0) + 1
                poi_last_recency[poi_raw_id] = min(recency, poi_last_recency.get(poi_raw_id, recency))
                if int(dow_id) == target_dow_id:
                    poi_same_dow[poi_raw_id] = poi_same_dow.get(poi_raw_id, 0) + 1
                if int(tbin_id) == target_tbin_id:
                    poi_same_tbin[poi_raw_id] = poi_same_tbin.get(poi_raw_id, 0) + 1
            if cat_id > 0:
                cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
                cat_last_recency[cat_id] = min(recency, cat_last_recency.get(cat_id, recency))
                if recency <= 5:
                    cat_recent5[cat_id] = cat_recent5.get(cat_id, 0) + 1
                if recency <= 10:
                    cat_recent10[cat_id] = cat_recent10.get(cat_id, 0) + 1
                if recency <= 20:
                    cat_recent20[cat_id] = cat_recent20.get(cat_id, 0) + 1
                if int(dow_id) == target_dow_id:
                    cat_same_dow[cat_id] = cat_same_dow.get(cat_id, 0) + 1
                if int(tbin_id) == target_tbin_id:
                    cat_same_tbin[cat_id] = cat_same_tbin.get(cat_id, 0) + 1

        rows = []
        for poi_raw_id, cat_id in zip(cand_raw_ids.tolist(), cand_cat_ids.tolist()):
            poi_raw_id = int(poi_raw_id)
            cat_id = int(cat_id)
            poi_count = poi_counts.get(poi_raw_id, 0)
            poi_recency = poi_last_recency.get(poi_raw_id, 0)
            cat_count = cat_counts.get(cat_id, 0)
            cat_recency = cat_last_recency.get(cat_id, 0)
            cat_recent10_count = cat_recent10.get(cat_id, 0)
            rows.append(
                [
                    1.0 if poi_count > 0 else 0.0,
                    math.log1p(poi_count),
                    (1.0 / math.log2(poi_recency + 1.0)) if poi_recency > 0 else 0.0,
                    math.log1p(poi_same_dow.get(poi_raw_id, 0)),
                    math.log1p(poi_same_tbin.get(poi_raw_id, 0)),
                    1.0 if cat_count > 0 else 0.0,
                    math.log1p(cat_count),
                    (1.0 / math.log2(cat_recency + 1.0)) if cat_recency > 0 else 0.0,
                    math.log1p(cat_recent5.get(cat_id, 0)),
                    math.log1p(cat_recent10_count),
                    math.log1p(cat_recent20.get(cat_id, 0)),
                    math.log1p(cat_same_dow.get(cat_id, 0)),
                    math.log1p(cat_same_tbin.get(cat_id, 0)),
                    float(cat_recent10_count) / max(1.0, float(cat_count)),
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _expand_disabled_recall_features(value: str):
        if not value:
            return []
        groups = {
            "all": [
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
            ],
            "llm": ["llm_cat_score", "is_llm_category"],
            "user_pref": ["user_cat_pref_score", "recent_cat_pref_score", "time_cat_pref_score", "is_user_pref_category"],
            "history": ["history_repeat_score", "history_recency_score", "is_history_exact"],
            "gcn": ["category_gcn_score", "no_cat_gcn_score", "is_category_gcn", "is_no_cat_gcn"],
            "fallback": ["temporal_pop_score", "spatial_score", "global_pop_score", "is_temporal_fallback", "is_spatial_fallback"],
            "spatial": ["spatial_score", "is_spatial_fallback"],
            "temporal": ["temporal_pop_score", "is_temporal_fallback"],
            "global_pop": ["global_pop_score"],
            "transition": [
                "user_prev_cat_transition_score",
                "user_prev_poi_transition_score",
                "user_recent_cat_transition_score",
                "user_recent_poi_transition_score",
                "global_prev_cat_transition_score",
                "global_prev_poi_transition_score",
                "is_user_transition_candidate",
                "is_global_transition_candidate",
            ],
            "flags": [
                "is_llm_category",
                "is_user_pref_category",
                "is_history_exact",
                "is_category_gcn",
                "is_no_cat_gcn",
                "is_temporal_fallback",
                "is_spatial_fallback",
                "is_user_transition_candidate",
                "is_global_transition_candidate",
            ],
            "scores": [
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
                "user_prev_cat_transition_score",
                "user_prev_poi_transition_score",
                "user_recent_cat_transition_score",
                "user_recent_poi_transition_score",
                "global_prev_cat_transition_score",
                "global_prev_poi_transition_score",
            ],
        }
        disabled = []
        for token in [x.strip() for x in value.split(",") if x.strip()]:
            disabled.extend(groups.get(token, [token]))
        return list(dict.fromkeys(disabled))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        meta = self.samples[row_idx]
        key = (str(meta["trajectory_id"]), int(meta["target_idx_in_traj"]))
        ctx = self.context_map[key]

        cand_ids = self.candidate_ids[row_idx]
        cand_mask = cand_ids >= 0
        safe_ids = cand_ids.clamp(min=0)
        model_cand_ids = torch.where(cand_mask, safe_ids + 1, torch.zeros_like(safe_ids))
        cand_cat_ids = self.poi_cat_ids[safe_ids]
        cand_region_ids = self.poi_region_input_ids[model_cand_ids]

        prev_idx = int(ctx["prev_poi_idx"])
        prev_lat = float(self.poi_lat[prev_idx]) if prev_idx >= 0 else 0.0
        prev_lon = float(self.poi_lon[prev_idx]) if prev_idx >= 0 else 0.0
        cand_lat = self.poi_lat[safe_ids]
        cand_lon = self.poi_lon[safe_ids]

        user_lat, user_lon = self.user_center.get(ctx["user_id"], (prev_lat, prev_lon))
        hist_center_lat = float(sum(ctx["history_latitudes"]) / len(ctx["history_latitudes"]))
        hist_center_lon = float(sum(ctx["history_longitudes"]) / len(ctx["history_longitudes"]))

        dist_prev = []
        dist_user = []
        dist_hist = []
        for lat, lon in zip(cand_lat.tolist(), cand_lon.tolist()):
            lat = float(lat)
            lon = float(lon)
            dist_prev.append(haversine_km(prev_lat, prev_lon, lat, lon) if prev_idx >= 0 else 0.0)
            dist_user.append(haversine_km(user_lat, user_lon, lat, lon))
            dist_hist.append(haversine_km(hist_center_lat, hist_center_lon, lat, lon))

        prev_cat_id = int(ctx["prev_cat_id"])
        same_cat_prev = (cand_cat_ids == prev_cat_id).float()

        label_index = int(self.label_indices[row_idx])
        true_target_idx = self.poi2idx.get(str(meta["target_poi_id"]), -1)
        if true_target_idx >= 0 and prev_idx >= 0:
            true_target_dist = haversine_km(prev_lat, prev_lon, float(self.poi_lat[true_target_idx]), float(self.poi_lon[true_target_idx]))
        else:
            true_target_dist = 0.0

        return {
            "sample_index": torch.tensor(row_idx, dtype=torch.long),
            "user_idx": torch.tensor(ctx["user_idx"], dtype=torch.long),
            "history_poi_ids": torch.tensor(ctx["history_poi_ids"], dtype=torch.long),
            "history_cat_ids": torch.tensor(ctx["history_cat_ids"], dtype=torch.long),
            "history_dow_ids": torch.tensor(ctx["history_dow_ids"], dtype=torch.long),
            "history_tbin_ids": torch.tensor(ctx["history_tbin_ids"], dtype=torch.long),
            "history_recency_ids": torch.tensor(ctx["history_recency_ids"], dtype=torch.long),
            "target_dow_id": torch.tensor(ctx["target_dow_id"], dtype=torch.long),
            "target_tbin_id": torch.tensor(ctx["target_tbin_id"], dtype=torch.long),
            "delta_bucket": torch.tensor(ctx["delta_bucket"], dtype=torch.long),
            "delta_minutes": torch.tensor(ctx["delta_minutes"], dtype=torch.float32),
            "prev_poi_input_id": torch.tensor((prev_idx + 1) if prev_idx >= 0 else 0, dtype=torch.long),
            "prev_cat_id": torch.tensor(prev_cat_id, dtype=torch.long),
            "candidate_ids": model_cand_ids,
            "candidate_cat_ids": cand_cat_ids,
            "candidate_region_ids": cand_region_ids,
            "candidate_cat_ranks": self.candidate_cat_ranks[row_idx],
            "candidate_gcn_ranks": self.candidate_gcn_ranks[row_idx],
            "candidate_scores": self.candidate_scores[row_idx],
            "candidate_recall_features": self.candidate_recall_features[row_idx],
            "candidate_history_features": self._build_history_candidate_features(ctx, safe_ids, cand_cat_ids)
            if self.enable_history_candidate_features
            else torch.zeros((cand_ids.numel(), len(self.history_candidate_feature_names)), dtype=torch.float32),
            "candidate_dist_prev": torch.tensor(dist_prev, dtype=torch.float32),
            "candidate_dist_user": torch.tensor(dist_user, dtype=torch.float32),
            "candidate_dist_hist": torch.tensor(dist_hist, dtype=torch.float32),
            "candidate_same_cat_prev": same_cat_prev,
            "candidate_mask": cand_mask,
            "label_index": self.label_indices[row_idx],
            "target_region_id": torch.tensor(int(ctx["target_region_id"]), dtype=torch.long),
            "true_target_dist": torch.tensor(true_target_dist, dtype=torch.float32),
        }


def collate_fn_laststep(batch):
    max_hist = max(max(x["history_poi_ids"].numel(), 1) for x in batch)

    def pad_1d(t, pad_value=0):
        if t.numel() == 0:
            return torch.full((max_hist,), pad_value, dtype=t.dtype)
        if t.numel() == max_hist:
            return t
        out = torch.full((max_hist,), pad_value, dtype=t.dtype)
        out[-t.numel():] = t
        return out

    return {
        "sample_index": torch.stack([x["sample_index"] for x in batch]),
        "user_idx": torch.stack([x["user_idx"] for x in batch]),
        "history_poi_ids": torch.stack([pad_1d(x["history_poi_ids"]) for x in batch]),
        "history_cat_ids": torch.stack([pad_1d(x["history_cat_ids"]) for x in batch]),
        "history_dow_ids": torch.stack([pad_1d(x["history_dow_ids"]) for x in batch]),
        "history_tbin_ids": torch.stack([pad_1d(x["history_tbin_ids"]) for x in batch]),
        "history_recency_ids": torch.stack([pad_1d(x["history_recency_ids"]) for x in batch]),
        "history_mask": torch.stack([pad_1d(torch.ones_like(x["history_poi_ids"]), 0).bool() for x in batch]),
        "target_dow_id": torch.stack([x["target_dow_id"] for x in batch]),
        "target_tbin_id": torch.stack([x["target_tbin_id"] for x in batch]),
        "delta_bucket": torch.stack([x["delta_bucket"] for x in batch]),
        "delta_minutes": torch.stack([x["delta_minutes"] for x in batch]),
        "prev_poi_input_id": torch.stack([x["prev_poi_input_id"] for x in batch]),
        "prev_cat_id": torch.stack([x["prev_cat_id"] for x in batch]),
        "candidate_ids": torch.stack([x["candidate_ids"] for x in batch]),
        "candidate_cat_ids": torch.stack([x["candidate_cat_ids"] for x in batch]),
        "candidate_region_ids": torch.stack([x["candidate_region_ids"] for x in batch]),
        "candidate_cat_ranks": torch.stack([x["candidate_cat_ranks"] for x in batch]),
        "candidate_gcn_ranks": torch.stack([x["candidate_gcn_ranks"] for x in batch]),
        "candidate_scores": torch.stack([x["candidate_scores"] for x in batch]),
        "candidate_recall_features": torch.stack([x["candidate_recall_features"] for x in batch]),
        "candidate_history_features": torch.stack([x["candidate_history_features"] for x in batch]),
        "candidate_dist_prev": torch.stack([x["candidate_dist_prev"] for x in batch]),
        "candidate_dist_user": torch.stack([x["candidate_dist_user"] for x in batch]),
        "candidate_dist_hist": torch.stack([x["candidate_dist_hist"] for x in batch]),
        "candidate_same_cat_prev": torch.stack([x["candidate_same_cat_prev"] for x in batch]),
        "candidate_mask": torch.stack([x["candidate_mask"] for x in batch]),
        "label_index": torch.stack([x["label_index"] for x in batch]),
        "target_region_id": torch.stack([x["target_region_id"] for x in batch]),
        "true_target_dist": torch.stack([x["true_target_dist"] for x in batch]),
    }


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, key_padding_mask=None):
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query_vec, tokens, token_mask):
        scores = (self.query_proj(query_vec).unsqueeze(1) * tokens).sum(dim=-1)
        scores = scores.masked_fill(~token_mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        return (attn.unsqueeze(-1) * tokens).sum(dim=1)


class RerankTransformerLastStep(nn.Module):
    def __init__(
        self,
        num_pois: int,
        num_cats: int,
        num_users: int,
        num_regions: int,
        gcn_embeddings: torch.Tensor,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_gcn_init: bool = False,
        random_init_poi_emb: bool = False,
        disable_candidate_gcn_rank: bool = True,
        disable_candidate_score: bool = True,
        disable_candidate_cat_rank: bool = True,
        disable_same_cat_prev: bool = True,
        disable_distance_features: bool = False,
        disable_user_features: bool = False,
        disable_time_features: bool = False,
        disable_history: bool = False,
        enable_candidate_history_attention: bool = False,
        enable_history_attention_recency_bias: bool = False,
        enable_history_attention_category_bias: bool = False,
        enable_history_attention_time_bias: bool = False,
        enable_history_attention_dow_bias: bool = False,
        history_attention_bias_init: float = 0.1,
        disable_candidate_self_attention: bool = False,
        single_query_token: bool = False,
        history_summary_len: int = 50,
        recall_feature_dim: int = 0,
        history_candidate_feature_dim: int = 0,
        summary_recency_mode: str = "legacy",
    ):
        super().__init__()
        gcn_dim = gcn_embeddings.shape[1]
        self.num_regions = num_regions
        self.disable_candidate_gcn_rank = bool(disable_candidate_gcn_rank)
        self.disable_candidate_score = bool(disable_candidate_score)
        self.disable_candidate_cat_rank = bool(disable_candidate_cat_rank)
        self.disable_same_cat_prev = bool(disable_same_cat_prev)
        self.disable_distance_features = bool(disable_distance_features)
        self.disable_user_features = bool(disable_user_features)
        self.disable_time_features = bool(disable_time_features)
        self.disable_history = bool(disable_history)
        self.enable_candidate_history_attention = bool(enable_candidate_history_attention)
        self.enable_history_attention_recency_bias = bool(enable_history_attention_recency_bias)
        self.enable_history_attention_category_bias = bool(enable_history_attention_category_bias)
        self.enable_history_attention_time_bias = bool(enable_history_attention_time_bias)
        self.enable_history_attention_dow_bias = bool(enable_history_attention_dow_bias)
        self.disable_candidate_self_attention = bool(disable_candidate_self_attention)
        self.single_query_token = bool(single_query_token)
        self.history_summary_len = int(history_summary_len)
        self.recall_feature_dim = int(recall_feature_dim)
        self.history_candidate_feature_dim = int(history_candidate_feature_dim)
        self.summary_recency_mode = str(summary_recency_mode)

        self.poi_emb = nn.Embedding(num_pois + 1, gcn_dim, padding_idx=0)
        if not random_init_poi_emb:
            with torch.no_grad():
                self.poi_emb.weight[1 : num_pois + 1].copy_(gcn_embeddings)
        if freeze_gcn_init:
            self.poi_emb.weight.requires_grad = False

        self.cat_emb = nn.Embedding(num_cats + 1, 32, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, 64, padding_idx=0)
        self.region_emb = nn.Embedding(num_regions + 1, 24, padding_idx=0)
        self.dow_emb = nn.Embedding(8, 8, padding_idx=0)
        self.tbin_emb = nn.Embedding(25, 16, padding_idx=0)
        self.delta_emb = nn.Embedding(6, 8, padding_idx=0)
        self.recency_emb = nn.Embedding(64, 16, padding_idx=0)
        self.cat_rank_emb = nn.Embedding(64, 16, padding_idx=0)
        self.gcn_rank_emb = nn.Embedding(5000, 16, padding_idx=0)
        self.query_role_emb = nn.Embedding(4, hidden_dim)

        self.delta_num_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        self.candidate_num_mlp = nn.Sequential(
            nn.Linear(5 + self.recall_feature_dim + self.history_candidate_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        bias_init = torch.tensor(float(history_attention_bias_init), dtype=torch.float32).clamp(min=1e-6)
        bias_raw_init = torch.log(torch.expm1(bias_init))
        self.history_attn_recency_bias = nn.Parameter(bias_raw_init.clone())
        self.history_attn_category_bias = nn.Parameter(bias_raw_init.clone())
        self.history_attn_time_bias = nn.Parameter(bias_raw_init.clone())
        self.history_attn_dow_bias = nn.Parameter(bias_raw_init.clone())

        self.history_item_proj = nn.Linear(gcn_dim + 32 + 16 + 16, hidden_dim)
        self.user_query_proj = nn.Linear(64 + gcn_dim + 32, hidden_dim)
        self.temporal_query_proj = nn.Linear(8 + 16 + 8 + 16, hidden_dim)
        self.transition_query_proj = nn.Linear(gcn_dim + 32 + 8 + 16, hidden_dim)
        self.summary_query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.candidate_proj = nn.Linear(gcn_dim + 32 + 24 + 16 + 16 + 64, hidden_dim)
        self.history_match_norm = nn.LayerNorm(hidden_dim)
        self.history_match_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.query_encoder = nn.ModuleList([SelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(max(1, num_layers - 1))])
        self.query_pool = AttentionPooling(hidden_dim)
        self.candidate_self_blocks = nn.ModuleList([SelfAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.candidate_cross_blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])

        self.coarse_gate = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.region_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_regions + 1))

    def encode_history(self, batch):
        if self.disable_history:
            bsz = batch["history_poi_ids"].size(0)
            seq_len = batch["history_poi_ids"].size(1)
            hidden_dim = self.history_item_proj.out_features
            return torch.zeros((bsz, seq_len, hidden_dim), dtype=self.poi_emb.weight.dtype, device=batch["history_poi_ids"].device)

        hist_poi = batch["history_poi_ids"]
        hist_cat = batch["history_cat_ids"]
        hist_tbin = batch["history_tbin_ids"]
        hist_mask = batch["history_mask"]
        hist_recency = batch["history_recency_ids"].clamp(min=0, max=63)

        if self.history_summary_len > 0:
            hist_poi = hist_poi[:, -self.history_summary_len :]
            hist_cat = hist_cat[:, -self.history_summary_len :]
            hist_tbin = hist_tbin[:, -self.history_summary_len :]
            hist_mask = hist_mask[:, -self.history_summary_len :]
            hist_recency = hist_recency[:, -self.history_summary_len :]

        hist = torch.cat(
            [
                self.poi_emb(hist_poi),
                self.cat_emb(hist_cat),
                self.tbin_emb(hist_tbin),
                self.recency_emb(hist_recency),
            ],
            dim=-1,
        )
        hist = self.history_item_proj(hist)
        hist = hist.masked_fill(~hist_mask.unsqueeze(-1), 0.0)
        return hist

    def build_query_tokens(self, batch, history_tokens, history_mask):
        hist_seed = history_tokens[:, -1]
        if self.disable_history:
            hist_summary = torch.zeros_like(hist_seed)
        else:
            hist_mask = history_mask[:, -history_tokens.size(1):]
            recency = batch["history_recency_ids"][:, -history_tokens.size(1):].float()
            if self.summary_recency_mode == "inverse":
                weights = 1.0 / torch.log2(recency.clamp(min=1.0) + 1.0)
            else:
                weights = recency
            weights = weights.unsqueeze(-1)
            weights = weights * hist_mask.unsqueeze(-1).to(history_tokens.dtype)
            denom = weights.sum(dim=1).clamp(min=1e-6)
            hist_summary = (history_tokens * weights).sum(dim=1) / denom

        user_emb = self.user_emb(batch["user_idx"])
        if self.disable_user_features:
            user_emb = torch.zeros_like(user_emb)

        target_dow_emb = self.dow_emb(batch["target_dow_id"])
        target_tbin_emb = self.tbin_emb(batch["target_tbin_id"])
        delta_emb = self.delta_emb(batch["delta_bucket"])
        delta_num = self.delta_num_proj(batch["delta_minutes"].unsqueeze(-1))
        if self.disable_time_features:
            target_dow_emb = torch.zeros_like(target_dow_emb)
            target_tbin_emb = torch.zeros_like(target_tbin_emb)
            delta_emb = torch.zeros_like(delta_emb)
            delta_num = torch.zeros_like(delta_num)

        user_token = self.user_query_proj(torch.cat([user_emb, self.poi_emb(batch["prev_poi_input_id"]), self.cat_emb(batch["prev_cat_id"])], dim=-1))
        temporal_token = self.temporal_query_proj(torch.cat([target_dow_emb, target_tbin_emb, delta_emb, delta_num], dim=-1))
        transition_token = self.transition_query_proj(torch.cat([self.poi_emb(batch["prev_poi_input_id"]), self.cat_emb(batch["prev_cat_id"]), delta_emb, delta_num], dim=-1))
        summary_token = self.summary_query_proj(torch.cat([hist_seed, hist_summary], dim=-1))

        if self.single_query_token:
            query_tokens = summary_token.unsqueeze(1)
            query_mask = torch.ones(query_tokens.size(0), 1, dtype=torch.bool, device=query_tokens.device)
            return query_tokens, query_mask, summary_token

        query_tokens = torch.stack([user_token, temporal_token, transition_token, summary_token], dim=1)
        role_ids = torch.arange(4, device=query_tokens.device).unsqueeze(0)
        query_tokens = query_tokens + self.query_role_emb(role_ids)
        query_mask = torch.ones(query_tokens.size(0), query_tokens.size(1), dtype=torch.bool, device=query_tokens.device)
        for block in self.query_encoder:
            query_tokens = block(query_tokens)
        pooled = self.query_pool(summary_token, query_tokens, query_mask)
        return query_tokens, query_mask, pooled

    def encode_candidates(self, batch):
        candidate_scores = batch["candidate_scores"]
        if self.disable_candidate_score:
            candidate_scores = torch.zeros_like(candidate_scores)

        candidate_gcn_ranks = batch["candidate_gcn_ranks"]
        if self.disable_candidate_gcn_rank:
            candidate_gcn_ranks = torch.zeros_like(candidate_gcn_ranks)

        candidate_cat_ranks = batch["candidate_cat_ranks"]
        if self.disable_candidate_cat_rank:
            candidate_cat_ranks = torch.zeros_like(candidate_cat_ranks)

        candidate_dist_prev = batch["candidate_dist_prev"]
        candidate_dist_user = batch["candidate_dist_user"]
        candidate_dist_hist = batch["candidate_dist_hist"]
        if self.disable_distance_features:
            candidate_dist_prev = torch.zeros_like(candidate_dist_prev)
            candidate_dist_user = torch.zeros_like(candidate_dist_user)
            candidate_dist_hist = torch.zeros_like(candidate_dist_hist)

        candidate_same_cat_prev = batch["candidate_same_cat_prev"]
        if self.disable_same_cat_prev:
            candidate_same_cat_prev = torch.zeros_like(candidate_same_cat_prev)

        numeric_parts = [
            candidate_scores.clamp(min=-20, max=20),
            torch.log1p(candidate_dist_prev.clamp(min=0, max=2000)),
            torch.log1p(candidate_dist_user.clamp(min=0, max=2000)),
            torch.log1p(candidate_dist_hist.clamp(min=0, max=2000)),
            candidate_same_cat_prev,
        ]
        cand_num_input = torch.stack(numeric_parts, dim=-1)
        recall_features = batch.get("candidate_recall_features")
        if recall_features is not None and self.recall_feature_dim > 0:
            cand_num_input = torch.cat(
                [
                    cand_num_input,
                    recall_features[:, :, : self.recall_feature_dim].clamp(min=-20, max=20).to(cand_num_input.dtype),
                ],
                dim=-1,
            )
        history_candidate_features = batch.get("candidate_history_features")
        if history_candidate_features is not None and self.history_candidate_feature_dim > 0:
            cand_num_input = torch.cat(
                [
                    cand_num_input,
                    history_candidate_features[:, :, : self.history_candidate_feature_dim]
                    .clamp(min=-20, max=20)
                    .to(cand_num_input.dtype),
                ],
                dim=-1,
            )
        cand_num = self.candidate_num_mlp(
            cand_num_input
        )
        return self.candidate_proj(
            torch.cat(
                [
                    self.poi_emb(batch["candidate_ids"]),
                    self.cat_emb(batch["candidate_cat_ids"]),
                    self.region_emb(batch["candidate_region_ids"]),
                    self.cat_rank_emb(candidate_cat_ranks.clamp(min=0, max=63)),
                    self.gcn_rank_emb(candidate_gcn_ranks.clamp(min=0, max=4999)),
                    cand_num,
                ],
                dim=-1,
            )
        )

    def apply_candidate_history_attention(self, cand_tokens, history_tokens, batch):
        if (not self.enable_candidate_history_attention) or self.disable_history:
            return cand_tokens
        hist_mask = batch["history_mask"][:, -history_tokens.size(1):]
        if not bool(hist_mask.any().item()):
            return cand_tokens
        scale = math.sqrt(max(1.0, float(cand_tokens.size(-1))))
        scores = torch.bmm(cand_tokens, history_tokens.transpose(1, 2)) / scale

        if self.enable_history_attention_recency_bias:
            recency = batch["history_recency_ids"][:, -history_tokens.size(1):].float().clamp(min=1.0)
            recency_bias = 1.0 / torch.log2(recency + 1.0)
            scores = scores + F.softplus(self.history_attn_recency_bias).to(scores.dtype) * recency_bias.unsqueeze(1)

        if self.enable_history_attention_category_bias:
            hist_cat = batch["history_cat_ids"][:, -history_tokens.size(1):]
            cand_cat = batch["candidate_cat_ids"]
            cat_match = (cand_cat.unsqueeze(2) == hist_cat.unsqueeze(1)) & (cand_cat.unsqueeze(2) > 0)
            scores = scores + F.softplus(self.history_attn_category_bias).to(scores.dtype) * cat_match.to(scores.dtype)

        if self.enable_history_attention_time_bias:
            hist_tbin = batch["history_tbin_ids"][:, -history_tokens.size(1):]
            target_tbin = batch["target_tbin_id"].unsqueeze(1)
            time_match = (hist_tbin == target_tbin) & (hist_tbin > 0)
            scores = scores + F.softplus(self.history_attn_time_bias).to(scores.dtype) * time_match.unsqueeze(1).to(scores.dtype)

        if self.enable_history_attention_dow_bias:
            hist_dow = batch["history_dow_ids"][:, -history_tokens.size(1):]
            target_dow = batch["target_dow_id"].unsqueeze(1)
            dow_match = (hist_dow == target_dow) & (hist_dow > 0)
            scores = scores + F.softplus(self.history_attn_dow_bias).to(scores.dtype) * dow_match.unsqueeze(1).to(scores.dtype)

        scores = scores.masked_fill(~hist_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, history_tokens)
        fusion = self.history_match_fusion(torch.cat([cand_tokens, context, cand_tokens * context], dim=-1))
        out = self.history_match_norm(cand_tokens + fusion)
        return out.masked_fill(~batch["candidate_mask"].unsqueeze(-1), 0.0)

    def coarse_score(self, query_tokens, pooled_query, cand_tokens, batch):
        query_expand = pooled_query.unsqueeze(1).expand_as(cand_tokens)
        if query_tokens.size(1) >= 3:
            temporal_expand = query_tokens[:, 1:2, :].expand_as(cand_tokens)
            transition_expand = query_tokens[:, 2:3, :].expand_as(cand_tokens)
        else:
            temporal_expand = query_expand
            transition_expand = query_expand
        feats = torch.cat([cand_tokens * query_expand, cand_tokens * temporal_expand, cand_tokens * transition_expand, cand_tokens], dim=-1)
        logits = self.coarse_gate(feats).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"], -1e9)

    def forward(self, batch):
        history_tokens = self.encode_history(batch)
        query_tokens, query_mask, pooled_query = self.build_query_tokens(batch, history_tokens, batch["history_mask"])
        cand_tokens = self.encode_candidates(batch)
        cand_tokens = self.apply_candidate_history_attention(
            cand_tokens,
            history_tokens,
            batch,
        )
        coarse_logits = self.coarse_score(query_tokens, pooled_query, cand_tokens, batch)
        region_logits = self.region_head(pooled_query)
        return {
            "coarse_logits": coarse_logits,
            "rank_logits": coarse_logits,
            "region_logits": region_logits,
        }


def compute_loss(outputs, batch, args):
    labels = batch["label_index"]
    valid_rows = labels >= 0
    if bool(valid_rows.any().item()):
        rank_logits = outputs["rank_logits"][valid_rows].float()
        coarse_logits = outputs["coarse_logits"][valid_rows].float()
        valid_labels = labels[valid_rows]
        label_smoothing = float(getattr(args, "label_smoothing", 0.0))
        rank_loss_per_row = F.cross_entropy(
            rank_logits,
            valid_labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        coarse_loss_per_row = F.cross_entropy(
            coarse_logits,
            valid_labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        rank_loss = rank_loss_per_row.mean()
        coarse_loss = coarse_loss_per_row.mean()
    else:
        rank_loss = outputs["rank_logits"].sum() * 0.0
        coarse_loss = outputs["coarse_logits"].sum() * 0.0

    region_loss = F.cross_entropy(outputs["region_logits"].float(), batch["target_region_id"])
    if float(args.region_loss_lambda) == 0.0:
        region_loss = region_loss * 0.0
    total = (
        args.rank_loss_lambda * rank_loss
        + args.coarse_loss_lambda * coarse_loss
        + args.region_loss_lambda * region_loss
    )
    return total, {
        "rank": float(rank_loss.item()),
        "coarse": float(coarse_loss.item()),
        "region": float(region_loss.item()),
    }


@dataclass
class Metrics:
    loss: float
    rank_loss: float
    coarse_loss: float
    region_loss: float
    hit1_all: float
    hit5_all: float
    hit10_all: float
    mrr_all: float
    hit1_hitonly: float
    hit5_hitonly: float
    hit10_hitonly: float
    mrr_hitonly: float
    region_acc: float
    candidate_miss_count: int


def count_candidate_miss(dataset):
    misses = 0
    for row_idx in dataset.indices:
        if int(dataset.label_indices[row_idx].item()) < 0:
            misses += 1
    return misses
