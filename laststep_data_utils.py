import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
    "trajectory_id",
    "UTC_time",
    "local_time",
    "POI_id",
    "POI_catname",
    "user_id",
    "day_of_week",
    "norm_in_day_time",
    "norm_day_shift",
    "latitude",
    "longitude",
]


@dataclass
class LastStepSample:
    trajectory_id: str
    target_idx_in_traj: int
    user_id: str
    target_poi_id: str
    target_catname: str
    target_utc_time: str
    target_local_time: str
    target_day_of_week: int
    target_norm_in_day_time: float
    target_norm_day_shift: float
    source_name: str
    source_rownum: int
    history_rows: list


def load_checkin_frame(csv_path: str, source_name: str, extra_cols=None):
    cols = list(BASE_COLUMNS)
    if extra_cols:
        for col in extra_cols:
            if col not in cols:
                cols.append(col)

    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    df["trajectory_id"] = df["trajectory_id"].astype(str)
    df["POI_id"] = df["POI_id"].astype(str)
    df["POI_catname"] = df["POI_catname"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["UTC_time"] = pd.to_datetime(df["UTC_time"], errors="coerce")
    df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
    df["_source_name"] = str(source_name)
    df["_source_rownum"] = list(range(len(df)))
    return df


def concat_history_frames(frames):
    if not frames:
        raise ValueError("at least one frame is required")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        ["user_id", "local_time", "_source_name", "_source_rownum"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def _build_user_history_index(history_df: pd.DataFrame):
    user_groups = {}
    position_map = {}
    for uid, g in history_df.groupby("user_id", sort=False):
        g = g.sort_values(
            ["local_time", "_source_name", "_source_rownum"],
            kind="mergesort",
        ).reset_index(drop=True)
        user_groups[str(uid)] = g
        for pos, row in enumerate(g.to_dict("records")):
            position_map[(str(uid), str(row["_source_name"]), int(row["_source_rownum"]))] = pos
    return user_groups, position_map


def build_laststep_samples(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    history_limit: int = 50,
    require_history: bool = True,
):
    user_groups, position_map = _build_user_history_index(history_df)
    target_df = target_df.sort_values(
        ["trajectory_id", "local_time", "_source_name", "_source_rownum"],
        kind="mergesort",
    )

    samples = []
    stats = {
        "num_trajectories": int(target_df["trajectory_id"].nunique()),
        "num_targets_total": 0,
        "num_targets_kept": 0,
        "num_targets_dropped_no_history": 0,
        "history_limit": int(history_limit),
    }

    for traj_id, g in target_df.groupby("trajectory_id", sort=False):
        g = g.sort_values(
            ["local_time", "_source_name", "_source_rownum"],
            kind="mergesort",
        ).reset_index(drop=True)
        target = g.iloc[-1]
        stats["num_targets_total"] += 1

        uid = str(target["user_id"])
        key = (uid, str(target["_source_name"]), int(target["_source_rownum"]))
        if key not in position_map:
            raise KeyError(f"target row not found in history index: {key}")

        user_hist = user_groups[uid]
        pos = position_map[key]
        hist = user_hist.iloc[max(0, pos - history_limit):pos]
        history_rows = hist.to_dict("records")

        if require_history and not history_rows:
            stats["num_targets_dropped_no_history"] += 1
            continue

        samples.append(
            LastStepSample(
                trajectory_id=str(traj_id),
                target_idx_in_traj=int(len(g) - 1),
                user_id=uid,
                target_poi_id=str(target["POI_id"]),
                target_catname=str(target["POI_catname"]),
                target_utc_time=str(target["UTC_time"]),
                target_local_time=str(target["local_time"]),
                target_day_of_week=int(target["day_of_week"]),
                target_norm_in_day_time=float(target["norm_in_day_time"]),
                target_norm_day_shift=float(target["norm_day_shift"]),
                source_name=str(target["_source_name"]),
                source_rownum=int(target["_source_rownum"]),
                history_rows=history_rows,
            )
        )
        stats["num_targets_kept"] += 1

    return samples, stats


def sample_to_prompt(sample: LastStepSample):
    uid = sample.user_id

    def fmt_record(r):
        return (
            f'At time: day_of_week={int(r["day_of_week"])}, '
            f't_in_day={float(r["norm_in_day_time"]):.2f}, '
            f'day_shift={float(r["norm_day_shift"]):.2f}, '
            f'user {uid} visited "{str(r["POI_catname"])}".'
        )

    hist_text = "\n".join(fmt_record(r) for r in sample.history_rows)
    if not hist_text:
        hist_text = "(no earlier check-in records)"

    target = sample.history_rows[-1] if sample.history_rows else None
    target_time = sample.target_local_time
    return hist_text, target_time


def build_cls_examples(samples, label2id):
    examples = []
    dropped = 0
    for sample in samples:
        y = str(sample.target_catname)
        if y not in label2id:
            dropped += 1
            continue

        uid = sample.user_id
        hist_text = "\n".join(
            (
                f'{str(r["trajectory_id"])} | '
                f'{format_prompt_time(r["local_time"])} | '
                f'{int(r["day_of_week"])} | '
                f'{float(r["norm_in_day_time"]):.2f} | '
                f'{str(r["POI_catname"])}'
            )
            for r in sample.history_rows
        )
        if not hist_text:
            hist_text = "(no earlier check-in records)\n"
        else:
            hist_text += "\n"

        text = (
            f"You will be given earlier check-ins of user {uid}.\n"
            f"Your task is to predict the POI category of the target check-in.\n\n"
            f"<history>\n"
            f"Format: trajectory_id | time | day_of_week | norm_in_day_time | poi_category\n"
            f"{hist_text}"
            f"</history>\n\n"
            f"<current>\n"
            f"target_trajectory_id | {sample.trajectory_id}\n"
            f"time | {format_prompt_time(sample.target_local_time)}\n"
            f"day_of_week | {sample.target_day_of_week}\n"
            f"norm_in_day_time | {sample.target_norm_in_day_time:.2f}\n\n"
            f"Return the target POI category only.\n"
            f"</current>\n\n"
            f"Answer:"
        )
        examples.append(
            {
                "text": text,
                "labels": int(label2id[y]),
                "trajectory_id": sample.trajectory_id,
                "target_idx_in_traj": sample.target_idx_in_traj,
                "target_poi_id": sample.target_poi_id,
                "true_catname": y,
                "user_id": uid,
                "history_len": len(sample.history_rows),
            }
        )
    return examples, dropped


def save_json(path: str, obj):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_prompt_time(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")
