import pandas as pd
import math

from src.muscle_mapping import get_parent

# === ANTAGONIST PAIRS (at specific sub-group or parent level) ===
ANTAGONIST_PAIRS: list[tuple[str, str, str, str]] = [
    ("Chest", "Back", "Chest", "Back"),
    ("Shoulders: Front", "Shoulders: Rear", "Front Delts", "Rear Delts"),
    ("Biceps", "Triceps", "Biceps", "Triceps"),
    ("Quads", "Hamstrings", "Quads", "Hamstrings"),
]

# === PUSH / PULL / LEG CLASSIFICATION ===
PUSH_MUSCLES = {
    "Chest: Upper", "Chest: Middle", "Chest: Lower",
    "Shoulders: Front", "Shoulders: Side",
    "Triceps",
}

PULL_MUSCLES = {
    "Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids",
    "Shoulders: Rear",
    "Biceps", "Forearms",
}

LEG_MUSCLES = {"Quads", "Hamstrings", "Glutes", "Calves", "Hip Flexors"}
CORE_MUSCLES = {"Core: Upper", "Core: Lower", "Core: Obliques"}

# === UPPER / LOWER ===
LOWER_MUSCLES = LEG_MUSCLES
UPPER_MUSCLES = {
    "Chest: Upper", "Chest: Middle", "Chest: Lower",
    "Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids",
    "Shoulders: Front", "Shoulders: Side", "Shoulders: Rear",
    "Core: Upper", "Core: Lower", "Core: Obliques",
    "Biceps", "Triceps", "Forearms",
    "Neck",
}


def _vol_dict(volume_summary: pd.DataFrame) -> dict[str, float]:
    return volume_summary.set_index("muscle_group")["total_volume"].to_dict()


def _classify(vol: dict[str, float], muscles: set[str]) -> float:
    return sum(v for k, v in vol.items() if k in muscles)


def _match_muscle(vol: dict[str, float], key: str) -> float:
    return sum(v for k, v in vol.items() if k == key or get_parent(k) == key)


def compute_push_pull_leg(volume_summary: pd.DataFrame) -> dict:
    if volume_summary.empty:
        return {"push": 0, "pull": 0, "leg": 0, "core": 0}
    vol = _vol_dict(volume_summary)
    p = _classify(vol, PUSH_MUSCLES)
    pu = _classify(vol, PULL_MUSCLES)
    l = _classify(vol, LEG_MUSCLES)
    c = _classify(vol, CORE_MUSCLES)
    total = p + pu + l + c or 1
    return {
        "push": p, "pull": pu, "leg": l, "core": c,
        "push_pct": p / total, "pull_pct": pu / total,
        "leg_pct": l / total, "core_pct": c / total,
        "push_pull_ratio": p / max(pu, 1),
    }


def compute_upper_lower(volume_summary: pd.DataFrame) -> dict:
    if volume_summary.empty:
        return {"upper": 1, "lower": 1}
    vol = _vol_dict(volume_summary)
    u = _classify(vol, UPPER_MUSCLES)
    l = _classify(vol, LOWER_MUSCLES)
    return {
        "upper": u, "lower": l,
        "ratio": u / max(l, 1),
        "upper_pct": u / (u + l) if (u + l) > 0 else 0.5,
    }


def compute_antagonist_balances(volume_summary: pd.DataFrame) -> list[dict]:
    if volume_summary.empty:
        return []
    vol = _vol_dict(volume_summary)
    results = []
    for key_a, key_b, label_a, label_b in ANTAGONIST_PAIRS:
        v1 = _match_muscle(vol, key_a)
        v2 = _match_muscle(vol, key_b)
        total = v1 + v2 or 1
        ratio = v1 / max(v2, 1)
        deviation = abs(math.log2((v1 + 1) / (v2 + 1)))
        ideal = 1.0
        pct_1 = v1 / total
        pct_2 = v2 / total
        results.append({
            "muscle_1": label_a,
            "muscle_2": label_b,
            "key_1": key_a,
            "key_2": key_b,
            "vol_1": v1,
            "vol_2": v2,
            "ratio": ratio,
            "deviation": deviation,
            "pct_1": pct_1,
            "pct_2": pct_2,
            "status": "balanced" if deviation <= 0.3 else (
                f"{label_a} dominant" if ratio > 1 else f"{label_b} dominant"
            ),
            "score": max(0, 1 - deviation),
        })
    return results


def compute_all_issues(
    push_pull_leg: dict,
    upper_lower: dict,
    antagonist_balances: list[dict],
) -> list[dict]:
    issues = []

    ppl = push_pull_leg
    ratio = ppl.get("push_pull_ratio", 1)
    if ratio > 1.3:
        issues.append({
            "type": "push_dominant",
            "severity": "warning",
            "title": "Push > Pull",
            "message": f"Push volume ({ppl['push']:,.0f} kg) is {ratio:.1f}x pull ({ppl['pull']:,.0f} kg). Add more pulling work.",
        })
    elif ratio < 0.7:
        issues.append({
            "type": "pull_dominant",
            "severity": "warning",
            "title": "Pull > Push",
            "message": f"Pull volume ({ppl['pull']:,.0f} kg) is {1/ratio:.1f}x push ({ppl['push']:,.0f} kg). Add more pushing work.",
        })

    leg_pct = ppl.get("leg_pct", 0)
    if leg_pct < 0.15:
        issues.append({
            "type": "legs_under",
            "severity": "info",
            "title": "Legs Lagging",
            "message": f"Leg volume is only {leg_pct*100:.0f}% of total ({ppl['leg']:,.0f} kg). Consider more leg training.",
        })

    ul = upper_lower
    ratio_ul = ul.get("ratio", 2)
    if ratio_ul > 3:
        issues.append({
            "type": "upper_dominant",
            "severity": "info",
            "title": "Upper > Lower",
            "message": f"Upper body volume is {ratio_ul:.1f}x lower body.",
        })
    elif ratio_ul < 0.5:
        issues.append({
            "type": "lower_dominant",
            "severity": "info",
            "title": "Lower > Upper",
            "message": f"Lower body volume is {1/ratio_ul:.1f}x upper body.",
        })

    for pair in antagonist_balances:
        if pair["deviation"] > 0.5:
            m1, m2 = pair["muscle_1"], pair["muscle_2"]
            ratio_pair = pair["ratio"]
            issues.append({
                "type": f"imbalance_{pair['key_1']}_{pair['key_2']}",
                "severity": "warning",
                "title": f"{m1} vs {m2}",
                "message": f"{m1} volume ({pair['vol_1']:,.0f} kg) is {ratio_pair:.1f}x {m2} ({pair['vol_2']:,.0f} kg). Consider balancing.",
            })

    return issues


def compute_overall_score(
    push_pull_leg: dict,
    upper_lower: dict,
    antagonist_balances: list[dict],
) -> float:
    deviations = []

    for pair in antagonist_balances:
        deviations.append(pair["deviation"])

    ppl = push_pull_leg
    ppr = ppl.get("push_pull_ratio", 1)
    deviations.append(abs(math.log2(max(ppr, 0.01))))

    ul = upper_lower
    ulr = ul.get("ratio", 2)
    deviations.append(abs(math.log2(max(ulr, 0.01))))

    if not deviations:
        return 1.0
    avg = sum(deviations) / len(deviations)
    return max(0.0, min(1.0, 1.0 - avg))


# ====================================================================
# STRENGTH-BASED BALANCE
# Each muscle's strength score = avg numeric level of its exercises.
# Balance compares opponent muscles by strength, not volume.
# ====================================================================

PUSH_MUSCLE_NAMES = {"Chest: Upper", "Chest: Middle", "Chest: Lower",
                     "Shoulders: Front", "Shoulders: Side", "Triceps"}
PULL_MUSCLE_NAMES = {"Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids",
                     "Shoulders: Rear", "Biceps", "Forearms"}
LEG_MUSCLE_NAMES = {"Quads", "Hamstrings", "Glutes", "Calves", "Hip Flexors"}
CORE_MUSCLE_NAMES = {"Core: Upper", "Core: Lower", "Core: Obliques"}


def compute_strength_antagonist_balances(
    muscle_strength: dict[str, float]
) -> list[dict]:
    results = []
    for key_a, key_b, label_a, label_b in ANTAGONIST_PAIRS:
        scores_a = [s for m, s in muscle_strength.items()
                    if m == key_a or get_parent(m) == key_a]
        scores_b = [s for m, s in muscle_strength.items()
                    if m == key_b or get_parent(m) == key_b]
        avg_a = sum(scores_a) / len(scores_a) if scores_a else 0
        avg_b = sum(scores_b) / len(scores_b) if scores_b else 0
        diff = avg_a - avg_b
        results.append({
            "muscle_1": label_a,
            "muscle_2": label_b,
            "key_1": key_a,
            "key_2": key_b,
            "strength_1": round(avg_a, 2),
            "strength_2": round(avg_b, 2),
            "diff": round(diff, 2),
            "status": "balanced" if abs(diff) < 0.5 else (
                f"{label_a} stronger" if diff > 0 else f"{label_b} stronger"
            ),
            "weaker": label_a if diff < 0 else label_b if diff > 0 else None,
        })
    return results


def compute_strength_category_scores(
    muscle_strength: dict[str, float]
) -> dict:
    def avg_for(names: set[str]) -> float:
        scores = [s for m, s in muscle_strength.items() if m in names]
        return sum(scores) / len(scores) if scores else 0

    push_s = avg_for(PUSH_MUSCLE_NAMES)
    pull_s = avg_for(PULL_MUSCLE_NAMES)
    leg_s = avg_for(LEG_MUSCLE_NAMES)
    core_s = avg_for(CORE_MUSCLE_NAMES)
    upper_s = avg_for(PUSH_MUSCLE_NAMES | PULL_MUSCLE_NAMES | CORE_MUSCLE_NAMES)
    lower_s = avg_for(LEG_MUSCLE_NAMES)

    return {
        "push": round(push_s, 2),
        "pull": round(pull_s, 2),
        "leg": round(leg_s, 2),
        "core": round(core_s, 2),
        "upper": round(upper_s, 2),
        "lower": round(lower_s, 2),
    }


def compute_strength_issues(
    antagonist: list[dict],
    categories: dict,
    muscle_strength: dict[str, float],
    muscle_volume: dict[str, float],
) -> list[dict]:
    issues = []

    for pair in antagonist:
        if pair["weaker"] and abs(pair["diff"]) >= 0.5:
            issues.append({
                "type": f"strength_{pair['key_1']}_{pair['key_2']}",
                "severity": "warning",
                "title": f"{pair['muscle_1']} vs {pair['muscle_2']} (Strength)",
                "message": (
                    f"{pair['weaker']} is weaker ({abs(pair['diff']):.1f} level gap). "
                    f"Strength: {pair['muscle_1']} {pair['strength_1']:.1f} vs "
                    f"{pair['muscle_2']} {pair['strength_2']:.1f}"
                ),
            })

    for muscle, strength in muscle_strength.items():
        parent = get_parent(muscle)
        vol = muscle_volume.get(muscle, 0)
        if strength < 1.5 and vol > 50000:
            issues.append({
                "type": f"inefficient_{muscle}",
                "severity": "warning",
                "title": f"{muscle}: High volume, Low strength",
                "message": (
                    f"{muscle} has {vol:,.0f} kg volume but strength score is only "
                    f"{strength:.1f} (Novice). Consider reducing junk volume and "
                    f"focusing on progressive overload."
                ),
            })

    return issues


def compute_strength_overall_score(
    antagonist: list[dict],
    categories: dict,
) -> float:
    gaps = [abs(p["diff"]) for p in antagonist]
    scores_list = [
        categories["push"], categories["pull"], categories["leg"],
        categories["upper"], categories["lower"],
    ]
    spread = max(scores_list) - min(scores_list) if scores_list else 0
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    deviation = (avg_gap + spread * 0.5) / 3.0
    return max(0.0, min(1.0, 1.0 - deviation))
