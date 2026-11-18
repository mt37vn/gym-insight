import pandas as pd
import numpy as np
from collections import defaultdict
from src.processor import estimate_1rm
from src.strength_level import get_muscle_strength_scores, EXERCISE_STANDARD_MAP, KEY_LIFT_PER_MUSCLE
from src.program_builder import Program, Workout, ExerciseSlot, Goal
from src.muscle_mapping import get_parent, PARENT_SUB_MAP


def compute_exercise_effectiveness(
    df: pd.DataFrame,
    muscle_strength: dict[str, float],
    bodyweight: float,
    age: int,
) -> dict[str, dict]:
    working = df[~df["warmup"]].copy()
    if working.empty:
        return {}

    working["estimated_1rm"] = working.apply(
        lambda r: estimate_1rm(r["weight"], r["reps"]), axis=1
    )

    weekly_ex = working.copy()
    weekly_ex["week"] = weekly_ex["start"].dt.isocalendar().week.astype(int)
    weekly_ex["year"] = weekly_ex["start"].dt.isocalendar().year.astype(int)

    weekly_vol = (
        weekly_ex.groupby(["exercise", "year", "week"])
        .agg(weekly_volume=("volume_kg", "sum"))
        .reset_index()
    )
    avg_weekly_volume = (
        weekly_vol.groupby("exercise")["weekly_volume"].mean().to_dict()
    )

    daily_max = (
        working.groupby(["exercise", "session_date"])
        .agg(max_1rm=("estimated_1rm", "max"))
        .reset_index()
        .sort_values("session_date")
    )

    results = {}
    for exercise in working["exercise"].unique():
        standard_key = EXERCISE_STANDARD_MAP.get(exercise, "Bench Press")
        strength_score = 0.0
        for muscle, key_lift in KEY_LIFT_PER_MUSCLE.items():
            if key_lift == exercise:
                strength_score = muscle_strength.get(muscle, 0.0)
                break
        else:
            for muscle, score in muscle_strength.items():
                if score > 0:
                    strength_score = score
                    break

        avg_vol = avg_weekly_volume.get(exercise, 0)
        efficiency = strength_score / (avg_vol / 1000 + 0.1)
        efficiency = max(0.0, min(5.0, efficiency))

        ex_daily = daily_max[daily_max["exercise"] == exercise].copy()
        n_points = len(ex_daily)

        if n_points >= 3:
            x = np.arange(n_points)
            y = ex_daily["max_1rm"].values
            if y.std() > 0:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            mean_1rm = y.mean()
            progression = (slope / max(mean_1rm, 1)) * 100
            progression = max(0.0, min(5.0, progression))
        else:
            progression = 0.0

        effectiveness = (efficiency + progression) / 2.0

        results[exercise] = {
            "effectiveness": round(effectiveness, 2),
            "efficiency": round(efficiency, 2),
            "progression": round(progression, 2),
            "strength_score": round(strength_score, 2),
            "avg_weekly_volume_kg": round(avg_vol, 0),
            "data_points": n_points,
        }

    return results


def compute_muscle_effectiveness(
    effectiveness: dict[str, dict],
    mapping: dict[str, tuple[list[str], list[str]]],
) -> dict[str, list[dict]]:
    muscle_exercises: dict[str, list[dict]] = defaultdict(list)
    for ex, data in effectiveness.items():
        primary, secondary = mapping.get(ex, (["Unknown"], []))
        for muscle in primary:
            muscle_exercises[muscle].append({**data, "exercise": ex, "role": "primary"})
        for muscle in secondary:
            muscle_exercises[muscle].append({**data, "exercise": ex, "role": "secondary"})
    return dict(muscle_exercises)


def suggest_alternatives(
    muscle: str,
    mapping: dict[str, tuple[list[str], list[str]]],
    effectiveness: dict[str, dict],
) -> list[str]:
    alternatives = []
    for ex, (primary, _) in sorted(mapping.items()):
        if muscle in primary:
            eff = effectiveness.get(ex, {}).get("effectiveness", 0)
            alternatives.append((eff, ex))
    alternatives.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in alternatives]


def generate_adaptive_program(
    df: pd.DataFrame,
    mapping: dict[str, tuple[list[str], list[str]]],
    profile: dict,
    effectiveness: dict[str, dict] = None,
) -> Program:
    bodyweight = profile.get("bodyweight_kg", 75)
    age = profile.get("age", 30)

    if effectiveness is None:
        muscle_strength = get_muscle_strength_scores(df, mapping, bodyweight, age)
        effectiveness = compute_exercise_effectiveness(df, muscle_strength, bodyweight, age)

    muscle_eff = compute_muscle_effectiveness(effectiveness, mapping)

    lagging: list[tuple[str, float]] = []
    for muscle, exercises in muscle_eff.items():
        if not exercises:
            continue
        primary_eff = [e["effectiveness"] for e in exercises if e["role"] == "primary"]
        avg_eff = np.mean(primary_eff) if primary_eff else np.mean([e["effectiveness"] for e in exercises])
        lagging.append((muscle, avg_eff))
    lagging.sort(key=lambda x: x[1])

    low_effectiveness = {m for m, e in lagging if e < 3.0}
    lagging_map = dict(lagging)

    goal = profile.get("current_goal", "hypertrophy")
    program = Program(
        name="Adaptive Program",
        goal=Goal(goal),
        weeks=8,
        workouts_per_week=4,
        description="AI-generated program targeting weak areas",
    )

    used_exercises: set = set()

    day_templates = [
        {"push": 0.7, "pull": 0.0, "leg": 0.0, "core": 0.3},
        {"push": 0.0, "pull": 0.7, "leg": 0.0, "core": 0.3},
        {"push": 0.0, "pull": 0.0, "leg": 1.0, "core": 0.0},
        {"push": 0.6, "pull": 0.0, "leg": 0.0, "core": 0.4},
    ]

    PUSH_MUSCLES = ["Chest: Upper", "Chest: Middle", "Chest: Lower",
                    "Shoulders: Front", "Shoulders: Side", "Triceps"]
    PULL_MUSCLES = ["Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids",
                    "Shoulders: Rear", "Biceps", "Forearms"]
    LEG_MUSCLES = ["Quads", "Hamstrings", "Glutes", "Calves", "Hip Flexors"]
    CORE_MUSCLES = ["Core: Upper", "Core: Lower", "Core: Obliques"]

    day_names = ["Upper Push", "Upper Pull", "Lower Body", "Upper Push 2"]

    for day_idx in range(min(program.workouts_per_week, len(day_templates))):
        template = day_templates[day_idx]
        workout = Workout(
            name=day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx + 1}",
            day_of_week=day_idx,
        )

        target_muscles = []
        if template.get("push", 0) > 0:
            target_muscles.extend(PUSH_MUSCLES)
        if template.get("pull", 0) > 0:
            target_muscles.extend(PULL_MUSCLES)
        if template.get("leg", 0) > 0:
            target_muscles.extend(LEG_MUSCLES)
        if template.get("core", 0) > 0:
            target_muscles.extend(CORE_MUSCLES)

        priority = []
        seen = set()
        for m in target_muscles:
            if m not in seen:
                is_lagging = 1 if m in low_effectiveness else 0
                lag_score = -lagging_map.get(m, 0)
                priority.append((m, -is_lagging, lag_score))
                seen.add(m)
        priority.sort(key=lambda x: (x[1], x[2]))

        exercises_added = 0
        for muscle, _, _ in priority:
            if exercises_added >= 8:
                break
            alternatives = suggest_alternatives(muscle, mapping, effectiveness)
            chosen = None
            for alt in alternatives:
                if alt not in used_exercises or len(alternatives) == 1:
                    chosen = alt
                    break
            if chosen is None and alternatives:
                chosen = alternatives[0]

            if chosen:
                used_exercises.add(chosen)
                workout.exercises.append(
                    ExerciseSlot(
                        name=chosen,
                        sets=4 if goal == "hypertrophy" else 5,
                        reps=10 if goal == "hypertrophy" else 5,
                        rpe=8.0,
                        rest_seconds=90 if goal == "hypertrophy" else 120,
                        notes=f"Target: {muscle}",
                    )
                )
                exercises_added += 1

        program.workouts.append(workout)

    return program
