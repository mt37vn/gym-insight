import json
import math
from pathlib import Path

STANDARDS_PATH = Path(__file__).parent.parent / "data" / "strength_standards.json"

LEVELS = ["Untrained", "Beginner", "Novice", "Intermediate", "Advanced", "Elite"]

LIFT_NAMES = {
    "Bench Press": "Bench Press",
    "Deadlift": "Deadlift",
    "Squat": "Squat",
    "Seated Barbell Overhead Press": "Overhead Press",
    "Barbell Row": "Barbell Row",
    "Lat Pulldown": "Lat Pulldown",
}

# Strength standards as BMI-based ratios for each lift/label
# Format: {lift: {level: {"min_ratio", "max_ratio"}}}
# Ratio = estimated_1rm / bodyweight
STANDARDS = {
    "Bench Press": {
        "Untrained": {"min": 0.0, "max": 0.6},
        "Beginner": {"min": 0.6, "max": 0.85},
        "Novice": {"min": 0.85, "max": 1.1},
        "Intermediate": {"min": 1.1, "max": 1.4},
        "Advanced": {"min": 1.4, "max": 1.75},
        "Elite": {"min": 1.75, "max": 999},
    },
    "Deadlift": {
        "Untrained": {"min": 0.0, "max": 0.8},
        "Beginner": {"min": 0.8, "max": 1.15},
        "Novice": {"min": 1.15, "max": 1.5},
        "Intermediate": {"min": 1.5, "max": 1.85},
        "Advanced": {"min": 1.85, "max": 2.3},
        "Elite": {"min": 2.3, "max": 999},
    },
    "Squat": {
        "Untrained": {"min": 0.0, "max": 0.65},
        "Beginner": {"min": 0.65, "max": 0.95},
        "Novice": {"min": 0.95, "max": 1.25},
        "Intermediate": {"min": 1.25, "max": 1.55},
        "Advanced": {"min": 1.55, "max": 1.95},
        "Elite": {"min": 1.95, "max": 999},
    },
    "Overhead Press": {
        "Untrained": {"min": 0.0, "max": 0.35},
        "Beginner": {"min": 0.35, "max": 0.5},
        "Novice": {"min": 0.5, "max": 0.65},
        "Intermediate": {"min": 0.65, "max": 0.85},
        "Advanced": {"min": 0.85, "max": 1.05},
        "Elite": {"min": 1.05, "max": 999},
    },
    "Barbell Row": {
        "Untrained": {"min": 0.0, "max": 0.5},
        "Beginner": {"min": 0.5, "max": 0.7},
        "Novice": {"min": 0.7, "max": 0.95},
        "Intermediate": {"min": 0.95, "max": 1.2},
        "Advanced": {"min": 1.2, "max": 1.5},
        "Elite": {"min": 1.5, "max": 999},
    },
    "Lat Pulldown": {
        "Untrained": {"min": 0.0, "max": 0.5},
        "Beginner": {"min": 0.5, "max": 0.7},
        "Novice": {"min": 0.7, "max": 0.95},
        "Intermediate": {"min": 0.95, "max": 1.2},
        "Advanced": {"min": 1.2, "max": 1.5},
        "Elite": {"min": 1.5, "max": 999},
    },
}


# Age adjustment factor: multiply ratio by this
def age_adjustment(age: int) -> float:
    if age < 20:
        return 1.05
    elif age < 30:
        return 1.0
    elif age < 40:
        return 0.95
    elif age < 50:
        return 0.88
    elif age < 60:
        return 0.8
    else:
        return 0.7


EXERCISE_STANDARD_MAP = {
    # Bench press family
    "Bench Press": "Bench Press",
    "Smith Machine Bench Press": "Bench Press",
    "Decline Bench Press": "Bench Press",
    "Incline Dumbbell Press": "Bench Press",
    "Dumbbell Bench Press": "Bench Press",
    "Leverage Machine Chest Press": "Bench Press",
    "Machine Chest Press": "Bench Press",
    "Dumbbell Chest Press": "Bench Press",
    "Dumbbell Decline Chest Press": "Bench Press",
    "Incline Bench Press": "Bench Press",
    "Smith Machine Incline Bench Press": "Bench Press",
    "Dumbbell Incline Chest Fly": "Bench Press",
    "Close-Grip Bench Press": "Bench Press",
    "Smith Machine Close-Grip Bench Press": "Bench Press",
    "Push Up": "Bench Press",
    "Push-Up": "Bench Press",
    # Deadlift family
    "Deadlift": "Deadlift",
    "Trap Bar Deadlift With Low Handles": "Deadlift",
    "Trap Bar Deadlift With High Handles": "Deadlift",
    "Rack Pull": "Deadlift",
    "Good Morning": "Deadlift",
    "Stiff-Legged Deadlift": "Deadlift",
    "Romanian Deadlift": "Deadlift",
    "Dumbbell Romanian Deadlift": "Deadlift",
    "Smith Machine Romanian Deadlift": "Deadlift",
    # Squat family
    "Squat": "Squat",
    "Smith Machine Squat": "Squat",
    "Front Squat": "Squat",
    "Goblet Squat": "Squat",
    "Overhead Squat": "Squat",
    "Bulgarian Split Squat": "Squat",
    "Bulgarian Split Squat (Bodyweight)": "Squat",
    "Barbell Lunge": "Squat",
    "Barbell Walking Lunges": "Squat",
    "Dumbbell Lunge": "Squat",
    "Smith Machine Lunge": "Squat",
    "Leg Press": "Squat",
    "Seated Leg Press": "Squat",
    "Single Leg Leg Press": "Squat",
    "One-Legged Leg Press": "Squat",
    "One-Legged Seated Leg Press": "Squat",
    "Leg Extension": "Squat",
    "Hip Thrust": "Squat",
    # Overhead press family
    "Seated Barbell Overhead Press": "Overhead Press",
    "Seated Smith Machine Shoulder Press": "Overhead Press",
    "Dumbbell Shoulder Press": "Overhead Press",
    "Seated Dumbbell Shoulder Press": "Overhead Press",
    "Machine Shoulder Press": "Overhead Press",
    "Overhead Press": "Overhead Press",
    "Arnold Press": "Overhead Press",
    # Row family
    "Barbell Row": "Barbell Row",
    "Pendlay Row": "Barbell Row",
    "Dumbbell Row": "Barbell Row",
    "Leverage Seated Machine Row": "Barbell Row",
    "Leverage Seated Machine Row (2 Arms)": "Barbell Row",
    "Seated Machine Row": "Barbell Row",
    "Cable Close-Grip Seated Row": "Barbell Row",
    "Cable Wide Grip Seated Row": "Barbell Row",
    "Rowing (Machine)": "Barbell Row",
    "Barbell Rear Delt Row": "Barbell Row",
    "Cable Rear Delt Row": "Barbell Row",
    # Lat pulldown / pull-up family
    "Lat Pulldown": "Lat Pulldown",
    "Leverage Machine Lat Pulldown": "Lat Pulldown",
    "Machine Lat Pulldown": "Lat Pulldown",
    "Close-Grip Lat Pulldown": "Lat Pulldown",
    "Lat Pulldown With Supinated Grip": "Lat Pulldown",
    "Lat Pulldown With Neutral Grip": "Lat Pulldown",
    "Lat Pulldown with Neutral Grip": "Lat Pulldown",
    "Neutral Close-Grip Lat Pulldown": "Lat Pulldown",
    "One-Arm Lat Pulldown": "Lat Pulldown",
    "Assisted Pull-Up": "Lat Pulldown",
    "Pull Up": "Lat Pulldown",
    "Pull-Up": "Lat Pulldown",
    "Chin Up": "Lat Pulldown",
    "Chin-Up": "Lat Pulldown",
}

KEY_LIFT_PER_MUSCLE: dict[str, str] = {
    "Chest: Upper": "Incline Dumbbell Press",
    "Chest: Middle": "Bench Press",
    "Chest: Lower": "Decline Bench Press",
    "Back: Lats": "Lat Pulldown",
    "Back: Traps": "Face Pull",
    "Back: Lower": "Deadlift",
    "Back: Rhomboids": "Reverse Dumbbell Flyes on Incline Bench",
    "Shoulders: Front": "Seated Barbell Overhead Press",
    "Shoulders: Side": "Lateral Raise",
    "Shoulders: Rear": "Face Pull",
    "Core: Upper": "Cable Crunch",
    "Core: Lower": "Hanging Leg Raise",
    "Core: Obliques": "Cable High to Low Wood Chop",
    "Biceps": "Barbell Curl",
    "Triceps": "Triceps Pushdown With Rope",
    "Forearms": "Gripper",
    "Quads": "Squat",
    "Hamstrings": "Romanian Deadlift",
    "Glutes": "Hip Thrust",
    "Calves": "Standing Calf Raise",
    "Hip Flexors": "Hanging Leg Raise",
    "Neck": "Deadlift",
}

BODYWEIGHT_EXERCISES = {
    "Hanging Leg Raise", "Lying Leg Raise", "Crunch", "Bench Dip",
    "Pull Up", "Chin Up", "Push Up", "Plank", "Leg Raise",
    "Hanging Knee Raise", "Hanging Windshield Wiper",
    "Bulgarian Split Squat (Bodyweight)", "Push-Up", "Pull-Up",
    "Chin-Up", "Bar Hang",
}


def get_level(ratio: float, lift: str, bodyweight: float, age: int = 30) -> tuple[str, float]:
    adjusted_ratio = ratio / age_adjustment(age)
    lift_standards = STANDARDS.get(lift, STANDARDS["Bench Press"])
    for level in LEVELS:
        s = lift_standards[level]
        if s["min"] <= adjusted_ratio < s["max"]:
            progress = 0.0
            next_level = None
            idx = LEVELS.index(level)
            if idx < len(LEVELS) - 1:
                next_level = LEVELS[idx + 1]
                current_range = lift_standards[next_level]["min"] - lift_standards[level]["min"]
                if current_range > 0:
                    progress = (adjusted_ratio - lift_standards[level]["min"]) / current_range
            return level, min(progress, 1.0)
    return LEVELS[-1], 1.0


def level_score(level: str) -> int:
    return LEVELS.index(level)


def compute_overall_level(
    exercise_levels: dict[str, tuple[str, float]], bodyweight: float, age: int
) -> tuple[str, float]:
    if not exercise_levels:
        return ("Untrained", 0.0)
    scores = [level_score(lvl) + progress for lvl, progress in exercise_levels.values()]
    avg = sum(scores) / len(scores)
    idx = int(avg)
    progress = avg - idx
    idx = min(idx, len(LEVELS) - 1)
    return (LEVELS[idx], progress)


def get_lift_display_name(exercise: str) -> str:
    return LIFT_NAMES.get(exercise, exercise)


def get_numeric_strength(weight: float, reps: int, bodyweight: float, age: int,
                         exercise_name: str = "") -> float:
    from src.processor import estimate_1rm
    if weight <= 0 or reps <= 0 or bodyweight <= 0:
        return 0.0
    est_1rm = estimate_1rm(weight, reps)
    ratio = est_1rm / max(bodyweight, 1)
    standard_key = EXERCISE_STANDARD_MAP.get(exercise_name, "Bench Press")
    level, progress = get_level(ratio, standard_key, bodyweight, age)
    return level_score(level) + progress


def get_muscle_strength_scores(df, mapping: dict, bodyweight: float, age: int) -> dict[str, float]:
    from src.processor import estimate_1rm
    working = df[~df["warmup"]].copy()

    exercise_peak: dict[str, float] = {}
    for _, row in working.iterrows():
        ex = row["exercise"]
        weight = row["weight"]
        reps = row["reps"]
        if weight <= 0 or reps <= 0 or ex in BODYWEIGHT_EXERCISES:
            continue
        standard_key = EXERCISE_STANDARD_MAP.get(ex, "Bench Press")
        est_1rm = estimate_1rm(weight, reps)
        ratio = est_1rm / max(bodyweight, 1)
        level, progress = get_level(ratio, standard_key, bodyweight, age)
        score = level_score(level) + progress
        if ex not in exercise_peak or score > exercise_peak[ex]:
            exercise_peak[ex] = score

    result: dict[str, float] = {}
    for muscle, key_ex in KEY_LIFT_PER_MUSCLE.items():
        if key_ex in exercise_peak:
            result[muscle] = exercise_peak[key_ex]
        else:
            fallback = [
                score for ex, score in exercise_peak.items()
                for muscle_name in mapping.get(ex, (["Unknown"], []))[0]
                if muscle_name == muscle
            ]
            result[muscle] = max(fallback) if fallback else 0.0

    return result
