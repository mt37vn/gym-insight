from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

CACHE_PATH = Path(__file__).parent.parent / "data" / "processed.parquet"
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"

COLUMN_TYPES = {
    "workout": "string",
    "start": "float64",
    "end": "float64",
    "exercise": "string",
    "weight": "float64",
    "bodyweight": "float64",
    "extraWeight": "float64",
    "assistingWeight": "float64",
    "distanceM": "float64",
    "distanceKM": "float64",
    "reps": "float64",
    "rpm": "float64",
    "time-per-500": "float64",
    "calories": "float64",
    "time": "float64",
    "warmup": "string",
    "max": "string",
    "fail": "string",
    "checked": "string",
    "setComment": "string",
    "workoutComment": "string",
    "form": "float64",
    "sleep": "float64",
    "calories_2": "float64",
    "stress": "float64",
}

RENAMED_COLUMNS = {
    "time-per-500": "time_per_500",
    "calories.1": "calories_diet",
}

EXERCISES_USE_BODYWEIGHT = {
    "Hanging Leg Raise", "Lying Leg Raise", "Crunch", "Bench Dip",
    "Pull Up", "Chin Up", "Dip", "Push Up", "Bodyweight Squat",
    "Lunge", "Plank", "Side Plank", "Bicycle Crunch", "Russian Twist",
    "Leg Raise", "Decline Crunch", "Cable Crunch",
}


def load_csv(path: str | Path = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DATA_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")

    df = pd.read_csv(path, dtype="string", low_memory=False)
    df.rename(columns=RENAMED_COLUMNS, inplace=True)
    return df


NUMERIC_COLS = {
    "weight", "bodyweight", "extraWeight", "assistingWeight",
    "distanceM", "distanceKM", "reps", "rpm", "time_per_500",
    "calories", "time", "form", "sleep", "calories_diet", "stress",
    "start", "end",
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["start"] = pd.to_datetime(df["start"], unit="ms", errors="coerce")
    df["end"] = pd.to_datetime(df["end"], unit="ms", errors="coerce")
    df["session_date"] = df["start"].dt.date
    df["session_week"] = df["start"].dt.isocalendar().week.astype(int)
    df["session_year"] = df["start"].dt.isocalendar().year.astype(int)
    df["week_label"] = df["start"].dt.strftime("%Y-W%V")

    df["reps"] = df["reps"].fillna(0).astype(int)
    df["weight"] = df["weight"].fillna(0)

    bw_mask = df["exercise"].isin(EXERCISES_USE_BODYWEIGHT) & (df["weight"] == 0)
    df.loc[bw_mask, "weight"] = df.loc[bw_mask, "bodyweight"]

    df["warmup"] = df["warmup"].astype(str).str.lower().isin(["true", "1"])
    df["is_max"] = df["max"].astype(str).str.lower().isin(["true", "1"])
    df["is_fail"] = df["fail"].astype(str).str.lower().isin(["true", "1"])

    df["session_duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
    df["session_duration_min"] = df["session_duration_min"].clip(lower=0, upper=300)

    df["volume_kg"] = df["weight"] * df["reps"]

    df.sort_values(["start", "checked"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def extract_program_name(workout: str) -> str:
    if pd.isna(workout):
        return "Unknown"
    if ":" in str(workout):
        return str(workout).split(":")[0].strip()
    return str(workout).strip()


def extract_workout_name(workout: str) -> str:
    if pd.isna(workout):
        return "Unknown"
    if ":" in str(workout):
        return str(workout).split(":")[1].strip()
    return str(workout).strip()


def add_program_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["program"] = df["workout"].apply(extract_program_name)
    df["workout_name"] = df["workout"].apply(extract_workout_name)
    return df


def load_and_process(path: str | Path = None) -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        return df

    df = load_csv(path)
    df = clean_data(df)
    df = add_program_cols(df)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def refresh_cache(path: str | Path = None) -> pd.DataFrame:
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    return load_and_process(path)


def get_exercise_list(df: pd.DataFrame) -> list[str]:
    return sorted(df["exercise"].unique().tolist())


def get_program_list(df: pd.DataFrame) -> list[str]:
    return sorted(df["program"].unique().tolist())


def get_date_range(df: pd.DataFrame) -> tuple:
    return df["start"].min(), df["start"].max()
