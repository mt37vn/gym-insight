import json
from pathlib import Path

MAPPING_PATH = Path(__file__).parent.parent / "data" / "exercise_mapping.json"

PARENT_MAP = {
    "Chest: Upper": "Chest",
    "Chest: Middle": "Chest",
    "Chest: Lower": "Chest",
    "Back: Lats": "Back",
    "Back: Traps": "Back",
    "Back: Lower": "Back",
    "Back: Rhomboids": "Back",
    "Shoulders: Front": "Shoulders",
    "Shoulders: Side": "Shoulders",
    "Shoulders: Rear": "Shoulders",
    "Core: Upper": "Core",
    "Core: Lower": "Core",
    "Core: Obliques": "Core",
    "Quads": "Leg",
    "Hamstrings": "Leg",
    "Glutes": "Leg",
    "Calves": "Leg",
    "Hip Flexors": "Leg",
    "Biceps": "Arms",
    "Triceps": "Arms",
    "Forearms": "Arms",
    "Neck": "Neck",
}

ALL_PARENTS = ["Chest", "Back", "Leg", "Core", "Shoulders", "Arms", "Neck"]

PARENT_SUB_MAP: dict[str, list[str]] = {
    "Chest": ["Chest: Upper", "Chest: Middle", "Chest: Lower"],
    "Back": ["Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids"],
    "Leg": ["Quads", "Hamstrings", "Glutes", "Calves", "Hip Flexors"],
    "Core": ["Core: Upper", "Core: Lower", "Core: Obliques"],
    "Shoulders": ["Shoulders: Front", "Shoulders: Side", "Shoulders: Rear"],
    "Arms": ["Biceps", "Triceps", "Forearms"],
}


def get_parent(muscle: str) -> str:
    return PARENT_MAP.get(muscle, muscle)


def get_sub(muscle: str) -> str | None:
    if ": " in muscle:
        return muscle.split(": ", 1)[1]
    return None


def rollup(volume_series: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k, v in volume_series.items():
        parent = get_parent(k)
        result[parent] = result.get(parent, 0) + v
    return result


# Format: exercise_name -> ( [primary_muscles], [secondary_muscles] )
DEFAULT_MAPPING = {
    # === CHEST ===
    "Bench Press":                               (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Smith Machine Bench Press":                 (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Decline Bench Press":                       (["Chest: Lower"], ["Triceps", "Shoulders: Front"]),
    "Incline Dumbbell Press":                    (["Chest: Upper"], ["Triceps", "Shoulders: Front"]),
    "Dumbbell Bench Press":                      (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Leverage Machine Chest Press":              (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Machine Chest Press":                       (["Chest: Middle"], ["Triceps"]),
    "Dumbbell Chest Press":                      (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Dumbbell Decline Chest Press":              (["Chest: Lower"], ["Triceps", "Shoulders: Front"]),
    "Incline Bench Press":                       (["Chest: Upper"], ["Triceps", "Shoulders: Front"]),
    "Smith Machine Incline Bench Press":         (["Chest: Upper"], ["Triceps", "Shoulders: Front"]),
    "Standing Cable Chest Fly (Upper Position)": (["Chest: Upper"], ["Shoulders: Front"]),
    "Standing Cable Chest Fly (Middle Position)":(["Chest: Middle"], []),
    "Standing Cable Chest Fly (Lower Position)": (["Chest: Lower"], []),
    "Dumbbell Incline Chest Fly":                (["Chest: Upper"], []),
    "Incline Cable Chest Fly":                   (["Chest: Upper"], []),
    "Incline Dumbbell Chest Fly":                (["Chest: Upper"], []),
    "Dumbbell Chest Fly":                        (["Chest: Middle"], []),
    "Machine Chest Fly":                         (["Chest: Middle"], []),
    "Cable Crossover":                           (["Chest: Middle"], []),
    "Push Up":                                   (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Push-Up":                                   (["Chest: Middle"], ["Triceps", "Shoulders: Front"]),
    "Dip":                                       (["Chest: Lower"], ["Triceps", "Shoulders: Front"]),
    "Bar Dip":                                   (["Chest: Lower"], ["Triceps", "Shoulders: Front"]),
    "Dip Machine":                               (["Chest: Lower"], ["Triceps"]),
    "Pullover":                                  (["Chest: Middle"], ["Back: Lats"]),
    # === BACK ===
    "Lat Pulldown":                              (["Back: Lats"], ["Biceps"]),
    "Leverage Machine Lat Pulldown":             (["Back: Lats"], ["Biceps"]),
    "Machine Lat Pulldown":                      (["Back: Lats"], ["Biceps"]),
    "Close-Grip Lat Pulldown":                   (["Back: Lats"], ["Biceps"]),
    "Lat Pulldown With Supinated Grip":          (["Back: Lats"], ["Biceps"]),
    "Lat Pulldown With Neutral Grip":            (["Back: Lats"], ["Biceps"]),
    "Lat Pulldown with Neutral Grip":            (["Back: Lats"], ["Biceps"]),
    "Neutral Close-Grip Lat Pulldown":           (["Back: Lats"], ["Biceps"]),
    "One-Arm Lat Pulldown":                      (["Back: Lats"], ["Biceps"]),
    "Straight Arm Lat Pulldown":                 (["Back: Lats"], ["Triceps"]),
    "Assisted Pull-Up":                          (["Back: Lats"], ["Biceps"]),
    "Pull Up":                                   (["Back: Lats"], ["Biceps"]),
    "Pull-Up":                                   (["Back: Lats"], ["Biceps"]),
    "Chin Up":                                   (["Back: Lats"], ["Biceps"]),
    "Chin-Up":                                   (["Back: Lats"], ["Biceps"]),
    "Barbell Row":                               (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Pendlay Row":                               (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Dumbbell Row":                              (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Leverage Seated Machine Row":               (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Leverage Seated Machine Row (2 Arms)":      (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Seated Machine Row":                        (["Back: Lats"], ["Biceps"]),
    "Cable Close-Grip Seated Row":               (["Back: Lats"], ["Biceps"]),
    "Cable Wide Grip Seated Row":                (["Back: Lats"], ["Biceps"]),
    "Rowing (Machine)":                          (["Back: Lats"], ["Biceps", "Shoulders: Rear"]),
    "Face Pull":                                 (["Back: Traps", "Shoulders: Rear"], []),
    "Barbell Upright Row":                       (["Back: Traps"], ["Shoulders: Side", "Shoulders: Front"]),
    "Y Rise":                                    (["Back: Traps"], ["Shoulders: Rear"]),
    "Deadlift":                                  (["Back: Lower"], ["Glutes", "Hamstrings", "Core: Lower"]),
    "Trap Bar Deadlift With Low Handles":        (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Trap Bar Deadlift With High Handles":       (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Rack Pull":                                 (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Good Morning":                              (["Back: Lower"], ["Hamstrings", "Glutes"]),
    "Back Extension":                            (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Back Extension on Roman chair":             (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Machine Back Extension":                    (["Back: Lower"], ["Glutes"]),
    "Reverse Hyperextension":                    (["Back: Lower"], ["Glutes", "Hamstrings"]),
    "Reverse Dumbbell Flyes":                    (["Back: Rhomboids", "Shoulders: Rear"], []),
    "Reverse Machine Fly":                       (["Back: Rhomboids", "Shoulders: Rear"], []),
    "Cable Rear Delt Row":                       (["Back: Rhomboids", "Shoulders: Rear"], []),
    "Barbell Rear Delt Row":                     (["Back: Rhomboids", "Shoulders: Rear"], []),
    "Reverse Dumbbell Flyes on Incline Bench":   (["Back: Rhomboids", "Shoulders: Rear"], []),
    # === SHOULDERS ===
    "Seated Barbell Overhead Press":             (["Shoulders: Front"], ["Triceps"]),
    "Seated Smith Machine Shoulder Press":       (["Shoulders: Front"], ["Triceps"]),
    "Dumbbell Shoulder Press":                   (["Shoulders: Front"], ["Triceps"]),
    "Seated Dumbbell Shoulder Press":            (["Shoulders: Front"], ["Triceps"]),
    "Machine Shoulder Press":                    (["Shoulders: Front"], ["Triceps"]),
    "Overhead Press":                            (["Shoulders: Front"], ["Triceps"]),
    "Arnold Press":                              (["Shoulders: Front"], ["Shoulders: Side", "Triceps"]),
    "Front Raise":                               (["Shoulders: Front"], []),
    "Lateral Raise":                             (["Shoulders: Side"], []),
    "Dumbbell Lateral Raise":                    (["Shoulders: Side"], []),
    "Cable Lateral Raise":                       (["Shoulders: Side"], []),
    # === CORE ===
    "Crunch":                                    (["Core: Upper"], []),
    "Machine Crunch":                            (["Core: Upper"], []),
    "Cable Crunch":                              (["Core: Upper"], []),
    "Ab bench":                                  (["Core: Upper"], []),
    "Kneeling Ab Wheel":                         (["Core: Upper"], []),
    "Hanging Leg Raise":                         (["Core: Lower"], ["Hip Flexors"]),
    "Lying Leg Raise":                           (["Core: Lower"], ["Hip Flexors"]),
    "Hanging Knee Raise":                        (["Core: Lower"], []),
    "Leg Raise":                                 (["Core: Lower"], []),
    "Plank":                                     (["Core: Upper"], []),
    "Pallof Press":                              (["Core: Upper"], []),
    "Cable High to Low Wood Chop":               (["Core: Obliques"], []),
    "Cable Low to High Wood Chop":               (["Core: Obliques"], []),
    "Cable Horizontal Wood Chop":                (["Core: Obliques"], []),
    "Oblique Crunch":                           (["Core: Obliques"], []),
    "Dumbbell Side Bend":                        (["Core: Obliques"], []),
    "Hanging Windshield Wiper":                  (["Core: Obliques"], ["Core: Lower"]),
    # === BICEPS ===
    "Barbell Curl":                              (["Biceps"], []),
    "Dumbbell Curl":                             (["Biceps"], []),
    "Incline Dumbbell Curl":                     (["Biceps"], []),
    "Machine Biceps Curl":                       (["Biceps"], []),
    "Cable Bicep Curl":                          (["Biceps"], []),
    "Cable Curl With Bar":                       (["Biceps"], []),
    "Cable Curl With Rope":                      (["Biceps"], []),
    "Barbell Preacher Curl":                     (["Biceps"], []),
    "Reverse Preacher Curl":                     (["Biceps"], ["Forearms"]),
    "Hammer Curl":                               (["Biceps"], ["Forearms"]),
    # === TRICEPS ===
    "Triceps Pushdown With Rope":                (["Triceps"], []),
    "Triceps Pushdown With Bar":                 (["Triceps"], []),
    "Machine Triceps Press":                     (["Triceps"], []),
    "Overhead Triceps Extension":                (["Triceps"], []),
    "Overhead Cable Triceps Extension":          (["Triceps"], []),
    "Overhead Cable Triceps Extension (with bar)":(["Triceps"], []),
    "Dumbbell Lying Triceps Extension":          (["Triceps"], []),
    "Close-Grip Bench Press":                    (["Triceps"], ["Chest: Middle", "Shoulders: Front"]),
    "Smith Machine Close-Grip Bench Press":       (["Triceps"], ["Chest: Middle", "Shoulders: Front"]),
    "Bench Dip":                                 (["Triceps"], ["Chest: Lower"]),
    # === FOREARMS ===
    "Bar Hang":                                  (["Forearms"], ["Back: Lats"]),
    "Gripper":                                   (["Forearms"], []),
    # === QUADS ===
    "Squat":                                     (["Quads"], ["Glutes", "Core: Lower"]),
    "Smith Machine Squat":                       (["Quads"], ["Glutes", "Core: Lower"]),
    "Front Squat":                               (["Quads"], ["Glutes", "Core: Lower"]),
    "Goblet Squat":                              (["Quads"], ["Glutes", "Core: Lower"]),
    "Overhead Squat":                            (["Quads"], ["Glutes", "Core: Lower", "Shoulders: Front"]),
    "Bulgarian Split Squat":                     (["Quads"], ["Glutes", "Hamstrings"]),
    "Bulgarian Split Squat (Bodyweight)":        (["Quads"], ["Glutes", "Hamstrings"]),
    "Barbell Lunge":                             (["Quads"], ["Glutes", "Hamstrings"]),
    "Barbell Walking Lunges":                    (["Quads"], ["Glutes", "Hamstrings"]),
    "Dumbbell Lunge":                            (["Quads"], ["Glutes", "Hamstrings"]),
    "Smith Machine Lunge":                       (["Quads"], ["Glutes", "Hamstrings"]),
    "Leg Press":                                 (["Quads"], ["Glutes", "Hamstrings"]),
    "Seated Leg Press":                          (["Quads"], ["Glutes", "Hamstrings"]),
    "Single Leg Leg Press":                      (["Quads"], ["Glutes", "Hamstrings"]),
    "One-Legged Leg Press":                      (["Quads"], ["Glutes", "Hamstrings"]),
    "One-Legged Seated Leg Press":               (["Quads"], ["Glutes", "Hamstrings"]),
    "Leg Extension":                              (["Quads"], []),
    "Sled Push":                                 (["Quads"], ["Calves", "Glutes"]),
    # === HAMSTRINGS ===
    "Seated Leg Curl":                           (["Hamstrings"], []),
    "Lying Leg Curl":                            (["Hamstrings"], []),
    "Romanian Deadlift":                         (["Hamstrings"], ["Glutes", "Back: Lower"]),
    "Dumbbell Romanian Deadlift":                (["Hamstrings"], ["Glutes", "Back: Lower"]),
    "Smith Machine Romanian Deadlift":           (["Hamstrings"], ["Glutes", "Back: Lower"]),
    "Stiff-Legged Deadlift":                     (["Hamstrings"], ["Glutes", "Back: Lower"]),
    # === GLUTES ===
    "Hip Thrust":                                (["Glutes"], ["Hamstrings"]),
    "Hip Abduction Machine":                     (["Glutes"], []),
    "Hip Adduction Machine":                     (["Glutes"], []),
    # === CALVES ===
    "Seated Calf Raise":                         (["Calves"], []),
    "Standing Calf Raise":                       (["Calves"], []),
    "Calf Press":                                (["Calves"], []),
    "Calf Raise in Leg Press":                   (["Calves"], []),
    "Calf press on Leg press":                   (["Calves"], []),
    "Smith Machine Calf Raise":                  (["Calves"], []),
    # === HIP FLEXORS ===
    # === NECK ===
    # === CARDIO / OTHER ===
    "Cycling":                                   (["Quads"], ["Hamstrings"]),
    "Running":                                   (["Quads"], ["Hamstrings", "Calves"]),
    "Swimming":                                  (["Back: Lats"], ["Shoulders: Front", "Chest: Middle"]),
}

SECONDARY_FACTOR = 0.5


def load_mapping() -> dict:
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH) as f:
            raw = json.load(f)
        result = {}
        for ex, (primary, secondary) in raw.items():
            result[ex] = (primary, secondary)
        return result
    return dict(DEFAULT_MAPPING)


def save_mapping(mapping: dict):
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: list(v) for k, v in mapping.items()}
    with open(MAPPING_PATH, "w") as f:
        json.dump(serializable, f, indent=2)


def get_unknown_exercises(df, mapping: dict) -> list[str]:
    if df is None:
        return []
    all_exercises = set(df["exercise"].unique())
    mapped = set(mapping.keys())
    unknown = sorted(all_exercises - mapped)
    return unknown


def classify_with_deepseek(exercise_name: str, deepseek_client) -> tuple[list[str], list[str]]:
    all_muscles = (
        "Chest: Upper, Chest: Middle, Chest: Lower, "
        "Back: Lats, Back: Traps, Back: Lower, Back: Rhomboids, "
        "Shoulders: Front, Shoulders: Side, Shoulders: Rear, "
        "Core: Upper, Core: Lower, Core: Obliques, "
        "Biceps, Triceps, Forearms, Quads, Hamstrings, Glutes, Calves, Hip Flexors, Neck"
    )
    prompt = (
        f"Classify the exercise '{exercise_name}' into 1-2 primary muscle groups "
        f"and 0-2 secondary (synergist) muscle groups. "
        f"Return only valid JSON: {{\"primary\": [\"Muscle1\", \"Muscle2\"], \"secondary\": [\"Muscle3\"]}}.\n"
        f"Available muscle groups (with sub-group where applicable):\n{all_muscles}"
    )
    try:
        response = deepseek_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-chat",
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = response.get("result", {})
        primary = result.get("primary", ["Unknown"])
        secondary = result.get("secondary", [])
        return primary, secondary
    except Exception:
        return (["Unknown"], [])


def update_mapping_with_unknown(df, mapping: dict, deepseek_client=None) -> dict:
    unknown = get_unknown_exercises(df, mapping)
    if not unknown:
        return mapping
    if deepseek_client is None:
        for ex in unknown:
            mapping[ex] = (["Unknown"], [])
        return mapping
    for ex in unknown:
        primary, secondary = classify_with_deepseek(ex, deepseek_client)
        mapping[ex] = (primary, secondary)
    save_mapping(mapping)
    return mapping


def build_muscle_to_exercise_map(mapping: dict) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for ex, (primary, secondary) in mapping.items():
        for muscle in primary + secondary:
            if muscle not in result:
                result[muscle] = set()
            result[muscle].add(ex)
    return result


def build_parent_to_exercise_map(mapping: dict) -> dict[str, set[str]]:
    sub_map = build_muscle_to_exercise_map(mapping)
    result: dict[str, set[str]] = {}
    for sub_group, exercises in sub_map.items():
        parent = get_parent(sub_group)
        if parent not in result:
            result[parent] = set()
        result[parent] |= exercises
    return result
