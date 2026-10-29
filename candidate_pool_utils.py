import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from laststep_data_utils import build_laststep_samples, concat_history_frames, load_checkin_frame


def load_llm_topk(jsonl_path: str):
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = (str(obj["trajectory_id"]), int(obj["target_idx_in_traj"]))
            records[key] = obj
    return records


def load_samples(target_csv: str, history_csvs: str, history_limit: int):
    target_source = Path(target_csv).stem
    target_df = load_checkin_frame(target_csv, target_source)
    history_frames = []
    target_in_history = False
    for idx, path in enumerate([x for x in history_csvs.split(",") if x.strip()]):
        norm = str(Path(path.strip()).resolve())
        if norm == str(Path(target_csv).resolve()):
            history_frames.append(target_df)
            target_in_history = True
        else:
            history_frames.append(load_checkin_frame(norm, f"hist{idx}"))
    if not target_in_history:
        history_frames.append(target_df)
    history_df = concat_history_frames(history_frames)
    return build_laststep_samples(
        target_df,
        history_df,
        history_limit=history_limit,
        require_history=True,
    )


def time_bin(norm_in_day_time):
    return min(23, max(0, int(float(norm_in_day_time) * 24)))


def build_catalog(catalog_csv: str, poi2idx: dict, idx2poi: list):
    cols = ["POI_id", "POI_catname", "day_of_week", "norm_in_day_time", "latitude", "longitude"]
    df = pd.read_csv(catalog_csv, usecols=cols, low_memory=False)
    df["POI_id"] = df["POI_id"].astype(str)
    df["POI_catname"] = df["POI_catname"].astype(str)

    num_pois = len(idx2poi)
    poi_catnames = [""] * num_pois
    poi_lat = [0.0] * num_pois
    poi_lon = [0.0] * num_pois
    cat_to_poi_indices = defaultdict(list)
    global_counts = Counter()
    temporal_counts = defaultdict(Counter)

    seen_meta = set()
    for row in df.itertuples(index=False):
        poi_id = str(row.POI_id)
        poi_idx = poi2idx.get(poi_id)
        if poi_idx is None:
            continue

        cat = str(row.POI_catname)
        global_counts[poi_idx] += 1
        temporal_counts[(int(row.day_of_week), time_bin(row.norm_in_day_time))][poi_idx] += 1

        if poi_idx in seen_meta:
            continue
        seen_meta.add(poi_idx)
        poi_catnames[poi_idx] = cat
        poi_lat[poi_idx] = float(row.latitude)
        poi_lon[poi_idx] = float(row.longitude)
        cat_to_poi_indices[cat].append(poi_idx)

    max_global = max(global_counts.values()) if global_counts else 1
    global_pop = {
        poi_idx: math.log1p(count) / math.log1p(max_global)
        for poi_idx, count in global_counts.items()
    }
    temporal_pop = {}
    for key, counts in temporal_counts.items():
        max_count = max(counts.values()) if counts else 1
        temporal_pop[key] = {
            poi_idx: math.log1p(count) / math.log1p(max_count)
            for poi_idx, count in counts.items()
        }

    return {
        "poi_catnames": poi_catnames,
        "poi_lat": poi_lat,
        "poi_lon": poi_lon,
        "cat_to_poi_indices": dict(cat_to_poi_indices),
        "global_pop": global_pop,
        "temporal_pop": temporal_pop,
    }


def build_transition_index(history_csvs: str, poi2idx: dict):
    cols = ["trajectory_id", "local_time", "POI_id", "POI_catname", "user_id"]
    frames = []
    for path in [x for x in history_csvs.split(",") if x.strip()]:
        df = pd.read_csv(path.strip(), usecols=cols, low_memory=False)
        df["trajectory_id"] = df["trajectory_id"].astype(str)
        df["POI_id"] = df["POI_id"].astype(str)
        df["POI_catname"] = df["POI_catname"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["user_id", "trajectory_id", "local_time"], kind="mergesort")

    user_cat_trans = defaultdict(Counter)
    user_poi_trans = defaultdict(Counter)
    global_cat_trans = defaultdict(Counter)
    global_poi_trans = defaultdict(Counter)

    for _, g in df.groupby(["user_id", "trajectory_id"], sort=False):
        rows = g.to_dict("records")
        for prev, nxt in zip(rows[:-1], rows[1:]):
            uid = str(prev["user_id"])
            prev_cat = str(prev["POI_catname"])
            next_cat = str(nxt["POI_catname"])
            prev_poi = str(prev["POI_id"])
            next_poi = str(nxt["POI_id"])
            next_idx = poi2idx.get(next_poi)

            user_cat_trans[(uid, prev_cat)][next_cat] += 1
            global_cat_trans[prev_cat][next_cat] += 1
            if next_idx is None:
                continue
            user_poi_trans[(uid, prev_poi)][next_idx] += 1
            global_poi_trans[prev_poi][next_idx] += 1

    return {
        "user_cat_trans": user_cat_trans,
        "user_poi_trans": user_poi_trans,
        "global_cat_trans": global_cat_trans,
        "global_poi_trans": global_poi_trans,
    }


def build_visible_user_transition_counters(sample, poi2idx):
    user_cat_trans = defaultdict(Counter)
    user_poi_trans = defaultdict(Counter)
    rows = sorted(
        sample.history_rows,
        key=lambda r: (str(r["trajectory_id"]), pd.to_datetime(r["local_time"], errors="coerce")),
    )

    by_traj = defaultdict(list)
    for r in rows:
        by_traj[str(r["trajectory_id"])].append(r)

    for traj_rows in by_traj.values():
        for prev, nxt in zip(traj_rows[:-1], traj_rows[1:]):
            prev_cat = str(prev["POI_catname"])
            next_cat = str(nxt["POI_catname"])
            prev_poi = str(prev["POI_id"])
            next_poi = str(nxt["POI_id"])
            next_idx = poi2idx.get(next_poi)
            user_cat_trans[prev_cat][next_cat] += 1
            if next_idx is not None:
                user_poi_trans[prev_poi][next_idx] += 1

    return user_cat_trans, user_poi_trans


def counter_norm_items(counter: Counter, topm: int):
    if not counter or topm <= 0:
        return []
    max_count = max(counter.values())
    rows = sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))[:topm]
    return [(key, math.log1p(count) / math.log1p(max_count)) for key, count in rows]


def cat_rank_score(rank: int):
    return 1.0 / math.log2(rank + 1.0)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2
    a += math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def proximity_score(distance_km: float, scale_km: float = 3.0):
    return 1.0 / (1.0 + max(0.0, distance_km) / scale_km)


def add_score(score_map: dict, poi_idx: int, value: float):
    if value == 0.0:
        return
    score_map[poi_idx] = score_map.get(poi_idx, 0.0) + float(value)


class GcnTopkCache:
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        self.emb_t = embeddings.t()
        self.cache = {}

    def get(self, poi_idx: int, k: int):
        key = (int(poi_idx), int(k))
        if key not in self.cache:
            kk = min(k, int(self.embeddings.shape[0]))
            scores = self.embeddings[poi_idx] @ self.emb_t
            top_scores, top_indices = torch.topk(scores, k=kk)
            self.cache[key] = list(zip(top_indices.tolist(), top_scores.tolist()))
        return self.cache[key]



def build_user_pref_context(sample, poi2idx, poi_catnames):
    hist_counts = Counter()
    cat_counts = Counter()
    recent_cat_scores = Counter()
    time_cat_counts = Counter()
    cat_lat_lon = defaultdict(list)
    all_lat_lon = []
    recent_lat_lon = []
    target_tbin = time_bin(sample.target_norm_in_day_time)

    for r in sample.history_rows:
        poi_idx = poi2idx.get(str(r["POI_id"]))
        if poi_idx is None:
            continue
        cat = poi_catnames[poi_idx]
        hist_counts[poi_idx] += 1
        cat_counts[cat] += 1
        lat_lon = (float(r["latitude"]), float(r["longitude"]))
        cat_lat_lon[cat].append(lat_lon)
        all_lat_lon.append(lat_lon)
        if int(r["day_of_week"]) == int(sample.target_day_of_week) and time_bin(r["norm_in_day_time"]) == target_tbin:
            time_cat_counts[cat] += 1

    for recency, r in enumerate(reversed(sample.history_rows), start=1):
        poi_idx = poi2idx.get(str(r["POI_id"]))
        if poi_idx is None:
            continue
        cat = poi_catnames[poi_idx]
        recent_cat_scores[cat] += 1.0 / math.log2(recency + 1.0)
        if len(recent_lat_lon) < 5:
            recent_lat_lon.append((float(r["latitude"]), float(r["longitude"])))

    def normalize_counter(counter):
        if not counter:
            return {}
        max_value = max(float(v) for v in counter.values())
        if max_value <= 0:
            return {}
        return {k: float(v) / max_value for k, v in counter.items()}

    def center(points):
        if not points:
            return None
        return (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )

    cat_centers = {cat: center(points) for cat, points in cat_lat_lon.items()}
    cat_pref = normalize_counter(cat_counts)
    recent_cat_pref = normalize_counter(recent_cat_scores)
    time_cat_pref = normalize_counter(time_cat_counts)
    combined_cat_pref = {}
    for cat in set(cat_pref) | set(recent_cat_pref) | set(time_cat_pref):
        combined_cat_pref[cat] = (
            1.5 * cat_pref.get(cat, 0.0)
            + 1.2 * recent_cat_pref.get(cat, 0.0)
            + 1.0 * time_cat_pref.get(cat, 0.0)
        )

    return {
        "hist_counts": hist_counts,
        "cat_pref": cat_pref,
        "recent_cat_pref": recent_cat_pref,
        "time_cat_pref": time_cat_pref,
        "combined_cat_pref": combined_cat_pref,
        "all_center": center(all_lat_lon),
        "recent_center": center(recent_lat_lon),
        "cat_centers": cat_centers,
    }


def spatial_bonus(poi_idx, cat, catalog, pref_ctx):
    lat = catalog["poi_lat"][poi_idx]
    lon = catalog["poi_lon"][poi_idx]
    score = 0.0

    all_center = pref_ctx["all_center"]
    if all_center is not None:
        score += 1.0 * proximity_score(haversine_km(all_center[0], all_center[1], lat, lon), scale_km=5.0)

    recent_center = pref_ctx["recent_center"]
    if recent_center is not None:
        score += 1.2 * proximity_score(haversine_km(recent_center[0], recent_center[1], lat, lon), scale_km=3.0)

    cat_center = pref_ctx["cat_centers"].get(cat)
    if cat_center is not None:
        score += 1.5 * proximity_score(haversine_km(cat_center[0], cat_center[1], lat, lon), scale_km=3.0)

    return score


