import json
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator


class Goal(str, Enum):
    STRENGTH = "strength"
    HYPERTROPHY = "hypertrophy"
    ENDURANCE = "endurance"
    GENERAL = "general_fitness"


class ExerciseSlot(BaseModel):
    name: str
    sets: int = Field(ge=1, le=20, default=3)
    reps: int = Field(ge=1, le=100, default=10)
    rpe: Optional[float] = Field(None, ge=1, le=10)
    rir: Optional[int] = Field(None, ge=0, le=5)
    rest_seconds: int = Field(ge=0, le=600, default=90)
    notes: str = ""
    is_warmup: bool = False

    def volume_kg(self, weight_kg: float = 0) -> float:
        return self.sets * self.reps * weight_kg


class Workout(BaseModel):
    name: str
    day_of_week: int = Field(ge=0, le=6, default=0)
    exercises: List[ExerciseSlot] = []
    notes: str = ""


class Program(BaseModel):
    name: str
    goal: Goal = Goal.GENERAL
    weeks: int = Field(ge=1, le=52, default=8)
    workouts_per_week: int = Field(ge=1, le=7, default=4)
    workouts: List[Workout] = []
    description: str = ""
    progression_model: str = ""

    @model_validator(mode="after")
    def validate_workouts(self):
        if self.workouts and len(self.workouts) != self.workouts_per_week:
            if len(self.workouts) > 0:
                pass
        return self


class ProgramEvaluation(BaseModel):
    overall_score: float = Field(ge=0, le=10)
    volume_balance_score: float = Field(ge=0, le=10)
    progression_score: float = Field(ge=0, le=10)
    exercise_selection_score: float = Field(ge=0, le=10)
    recovery_score: float = Field(ge=0, le=10)
    goal_alignment_score: float = Field(ge=0, le=10)
    strengths: List[str] = []
    weaknesses: List[str] = []
    recommendations: List[str] = []


MUSCLE_GROUP_EXERCISES = {
    "Chest": ["Bench Press", "Incline Bench Press", "Decline Bench Press", "Dumbbell Press", "Cable Fly", "Push Up"],
    "Back": ["Barbell Row", "Dumbbell Row", "Lat Pulldown", "Pull Up", "Seated Cable Row", "Deadlift"],
    "Shoulders: Front": ["Overhead Press", "Arnold Press", "Front Raise", "Machine Shoulder Press"],
    "Shoulders: Side": ["Lateral Raise", "Upright Row", "Machine Lateral Raise"],
    "Shoulders: Rear": ["Face Pull", "Reverse Fly", "Rear Delt Row"],
    "Biceps": ["Barbell Curl", "Dumbbell Curl", "Hammer Curl", "Cable Curl", "Incline Curl"],
    "Triceps": ["Triceps Pushdown", "Overhead Extension", "Skull Crusher", "Close Grip Press", "Dip"],
    "Quads": ["Squat", "Leg Press", "Bulgarian Split Squat", "Hack Squat", "Lunge", "Leg Extension"],
    "Hamstrings": ["Romanian Deadlift", "Leg Curl", "Good Morning", "Nordic Curl"],
    "Glutes": ["Hip Thrust", "Glute Bridge", "Cable Pull Through", "Step Up"],
    "Calves": ["Standing Calf Raise", "Seated Calf Raise", "Calf Press"],
    "Core": ["Plank", "Leg Raise", "Cable Crunch", "Ab Wheel", "Pallof Press", "Russian Twist"],
}


def evaluate_program(program: Program) -> ProgramEvaluation:
    scores = {}
    strengths = []
    weaknesses = []
    recommendations = []

    total_sets = sum(
        ex.sets for w in program.workouts for ex in w.exercises if not ex.is_warmup
    )
    avg_sets_per_workout = total_sets / max(len(program.workouts), 1)

    if program.goal == Goal.HYPERTROPHY:
        if 15 <= avg_sets_per_workout <= 25:
            scores["volume"] = 9
            strengths.append("Appropriate volume for hypertrophy")
        elif 10 <= avg_sets_per_workout <= 30:
            scores["volume"] = 7
        else:
            scores["volume"] = 4
            weaknesses.append(f"Volume ({avg_sets_per_workout:.0f} sets/workout) may not be optimal for hypertrophy")
    elif program.goal == Goal.STRENGTH:
        if 10 <= avg_sets_per_workout <= 20:
            scores["volume"] = 9
            strengths.append("Appropriate volume for strength")
        else:
            scores["volume"] = 6

    exercise_categories = {}
    for w in program.workouts:
        for ex in w.exercises:
            if ex.is_warmup:
                continue
            for muscle, exercises in MUSCLE_GROUP_EXERCISES.items():
                if any(e.lower() in ex.name.lower() for e in exercises):
                    exercise_categories.setdefault(muscle, 0)
                    exercise_categories[muscle] += ex.sets

    if exercise_categories:
        push = exercise_categories.get("Chest", 0) + exercise_categories.get("Shoulders: Front", 0) + exercise_categories.get("Triceps", 0)
        pull = exercise_categories.get("Back", 0) + exercise_categories.get("Biceps", 0)
        if push > 0 and pull > 0:
            ratio = push / pull
            if 0.7 <= ratio <= 1.5:
                scores["balance"] = 9
                strengths.append("Good push/pull balance")
            elif 0.5 <= ratio <= 2.0:
                scores["balance"] = 6
            else:
                scores["balance"] = 3
                weaknesses.append(f"Push/pull ratio of {ratio:.1f}:1 may cause imbalances")

        lower = sum(exercise_categories.get(m, 0) for m in ["Quads", "Hamstrings", "Glutes", "Calves"])
        upper = sum(exercise_categories.get(m, 0) for m in ["Chest", "Back", "Shoulders: Front", "Shoulders: Side", "Shoulders: Rear", "Biceps", "Triceps"])
        if lower > 0 and upper > 0:
            ul_ratio = upper / lower
            if 0.5 <= ul_ratio <= 3.0:
                if "balance" not in scores:
                    scores["balance"] = 7
            else:
                weaknesses.append(f"Upper/lower ratio of {ul_ratio:.1f}:1 may be imbalanced")
    else:
        scores["balance"] = 5

    scores.setdefault("volume", 6)
    scores.setdefault("balance", 6)
    scores.setdefault("progression", 7)
    scores.setdefault("selection", 7)
    scores.setdefault("recovery", 7)
    scores.setdefault("goal_alignment", 7)

    overall = sum(scores.values()) / len(scores)
    return ProgramEvaluation(
        overall_score=round(overall, 1),
        volume_balance_score=scores["balance"],
        progression_score=scores["progression"],
        exercise_selection_score=scores["selection"],
        recovery_score=scores["recovery"],
        goal_alignment_score=scores["goal_alignment"],
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
    )
