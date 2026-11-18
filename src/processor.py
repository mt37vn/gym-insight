import pandas as pd
import numpy as np
from datetime import timedelta

from src.muscle_mapping import get_parent


def epley_1rm(weight: float, reps: int) -> float:
    if reps <= 0 or weight <= 0:
        return 0.0
    if reps == 1:
        return weight
    return weight * (1 + reps / 30)


def brzycki_1rm(weight: float, reps: int) -> float:
    if reps <= 0 or weight <= 0:
        return 0.0
    if reps == 1:
        return weight
    return weight * (36 / (37 - reps))


def estimate_1rm(weight: float, reps: int, formula: str = "epley") -> float:
    if formula == "brzycki":
        return brzycki_1rm(weight, reps)
    return epley_1rm(weight, reps)


def detect_prs(df: pd.DataFrame, formula: str = "epley") -> pd.DataFrame:
    working_sets = df[~df["warmup"]].copy()
    working_sets["estimated_1rm"] = working_sets.apply(
        lambda r: estimate_1rm(r["weight"], r["reps"], formula), axis=1
    )

    prs = (
        working_sets.sort_values("start")
        .groupby("exercise")
        .agg(
            best_weight=("weight", "max"),
            best_1rm=("estimated_1rm", "max"),
            best_volume=("volume_kg", "max"),
            total_sets=("exercise", "count"),
            first_date=("start", "min"),
            last_date=("start", "max"),
            best_weight_date=("start", lambda x: x.loc[working_sets.loc[x.index, "weight"].idxmax()] if len(x) > 0 else None),
            best_1rm_date=("start", lambda x: x.loc[working_sets.loc[x.index, "estimated_1rm"].idxmax()] if len(x) > 0 else None),
        )
        .reset_index()
    )

    prs["best_weight_date"] = pd.to_datetime(prs["best_weight_date"])
    prs["best_1rm_date"] = pd.to_datetime(prs["best_1rm_date"])
    prs["first_date"] = pd.to_datetime(prs["first_date"])
    prs["last_date"] = pd.to_datetime(prs["last_date"])

    return prs


def compute_weekly_volume(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df[~df["warmup"]]
        .groupby(["session_year", "session_week", "week_label"])
        .agg(
            total_volume=("volume_kg", "sum"),
            total_sets=("reps", "count"),
            total_reps=("reps", "sum"),
            avg_weight=("weight", "mean"),
            session_count=("session_date", "nunique"),
            total_duration=("session_duration_min", "sum"),
        )
        .reset_index()
    )
    weekly["week_start"] = weekly.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["session_year"]), int(r["session_week"]), 1),
        axis=1,
    )
    weekly.sort_values("week_start", inplace=True)
    return weekly


def compute_muscle_volume(df: pd.DataFrame, mapping: dict[str, tuple[list[str], list[str]]]) -> pd.DataFrame:
    working = df[~df["warmup"]].copy()

    rows = []
    for _, row in working.iterrows():
        ex = row["exercise"]
        primary, secondary = mapping.get(ex, (["Unknown"], []))
        for muscle in primary:
            rows.append({
                "exercise": ex,
                "muscle_group": muscle,
                "volume_kg": row["volume_kg"],
                "reps": row["reps"],
                "weight": row["weight"],
                "start": row["start"],
                "is_primary": True,
            })
        for muscle in secondary:
            rows.append({
                "exercise": ex,
                "muscle_group": muscle,
                "volume_kg": row["volume_kg"] * 0.5,
                "reps": row["reps"],
                "weight": row["weight"],
                "start": row["start"],
                "is_primary": False,
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            "muscle_group", "total_volume", "total_sets", "total_reps",
            "avg_weight", "exercise_count"
        ])

    summary = (
        result.groupby("muscle_group")
        .agg(
            total_volume=("volume_kg", "sum"),
            total_sets=("reps", "count"),
            total_reps=("reps", "sum"),
            avg_weight=("weight", "mean"),
            exercise_count=("exercise", "nunique"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
    )

    return summary


def rollup_muscle_volume(volume_summary: pd.DataFrame) -> pd.DataFrame:
    if volume_summary.empty:
        return volume_summary
    vol = volume_summary.copy()
    vol["parent"] = vol["muscle_group"].apply(get_parent)
    numeric_cols = ["total_volume", "total_sets", "total_reps"]
    agg = vol.groupby("parent")[numeric_cols].sum().reset_index()
    avg_w = vol.groupby("parent")["avg_weight"].apply(
        lambda x: (x * vol.loc[x.index, "total_sets"]).sum() / max(vol.loc[x.index, "total_sets"].sum(), 1)
    )
    agg["avg_weight"] = avg_w.values
    ex_c = vol.groupby("parent")["exercise_count"].sum()
    agg["exercise_count"] = ex_c.values
    agg.rename(columns={"parent": "muscle_group"}, inplace=True)
    agg.sort_values("total_volume", ascending=False, inplace=True)
    return agg


def compute_exercise_progression(
    df: pd.DataFrame, exercise: str
) -> pd.DataFrame:
    ex_df = df[(df["exercise"] == exercise) & ~df["warmup"]].copy()
    if ex_df.empty:
        return pd.DataFrame()

    ex_df["estimated_1rm"] = ex_df.apply(
        lambda r: estimate_1rm(r["weight"], r["reps"]), axis=1
    )

    daily = (
        ex_df.groupby("session_date")
        .agg(
            max_weight=("weight", "max"),
            max_1rm=("estimated_1rm", "max"),
            total_volume=("volume_kg", "sum"),
            total_reps=("reps", "sum"),
            avg_reps=("reps", "mean"),
            set_count=("reps", "count"),
        )
        .reset_index()
    )
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    daily.sort_values("session_date", inplace=True)
    return daily


def compute_frequency(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["session_year", "session_week", "week_label"])
        .agg(workout_days=("session_date", "nunique"))
        .reset_index()
        .sort_values(["session_year", "session_week"])
    )
    return weekly


def compute_workout_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby("program")
        .agg(
            total_sessions=("session_date", "nunique"),
            total_volume=("volume_kg", "sum"),
            total_duration=("session_duration_min", "sum"),
        )
        .reset_index()
        .sort_values("total_sessions", ascending=False)
    )
    return dist


def compute_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        df.groupby(["session_date", "workout"])
        .agg(
            duration_min=("session_duration_min", "first"),
            exercise_count=("exercise", "nunique"),
            total_volume=("volume_kg", "sum"),
            total_sets=("reps", "count"),
            program=("program", "first"),
        )
        .reset_index()
        .sort_values("session_date", ascending=False)
    )
    return sessions
