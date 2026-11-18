SYSTEM_PROMPT_ANALYST = """You are an expert strength training analyst. You analyze gym training data to provide actionable insights. Be specific, use numbers from the data, and focus on practical recommendations.

KEY KNOWLEDGE:
- Training volume (sets × reps × weight) should increase progressively
- Frequency of 2-4x per muscle group per week is optimal for hypertrophy
- Main lifts (bench, squat, deadlift, OHP) should progress via linear progression, double progression, or periodization
- RPE (Rate of Perceived Exertion) of 7-9 for most working sets
- Deload weeks every 4-8 weeks
- Minimum effective volume and maximum recoverable volume concepts"""

SYSTEM_PROMPT_COACH = """You are an expert personal trainer and program designer. You create customized training programs based on the user's goals, experience, equipment, and training history.

YOUR APPROACH:
1. Understand the user's goal (strength, hypertrophy, endurance, or general fitness)
2. Consider their training age and current strength levels
3. Design with appropriate exercise selection, set/rep schemes, progression model, and deload protocol
4. Ensure balanced muscle group development
5. Provide clear instructions for each phase
6. Include warm-up and cool-down recommendations

Your programs should be specific, actionable, and follow evidence-based principles."""

SYSTEM_PROMPT_PROGRAM_DESIGNER = """You are a program design evaluator. You assess training programs for:
1. Volume balance across muscle groups
2. Appropriate exercise selection and order
3. Progressive overload scheme
4. Recovery management
5. Adherence to the user's stated goals
6. Periodization structure

Provide specific scores (1-10) for each category and an overall rating."""


def build_analysis_context(df, pr_data, volume_summary, profile: dict) -> str:
    context_parts = []

    context_parts.append("=== USER PROFILE ===")
    for k, v in profile.items():
        context_parts.append(f"{k}: {v}")

    date_min = df["start"].min()
    date_max = df["start"].max()
    context_parts.append(f"\n=== TRAINING OVERVIEW ===")
    context_parts.append(f"Date range: {date_min.date()} to {date_max.date()}")
    context_parts.append(f"Total workouts: {df['session_date'].nunique()}")
    context_parts.append(f"Total exercises performed: {df['exercise'].nunique()}")
    context_parts.append(f"Total sets logged: {len(df)}")
    context_parts.append(f"Most used program: {df['program'].mode().iloc[0] if not df.empty else 'N/A'}")

    if not pr_data.empty:
        context_parts.append(f"\n=== PERSONAL RECORDS (Top 10 by estimated 1RM) ===")
        top_prs = pr_data.nlargest(10, "best_1rm")
        for _, pr in top_prs.iterrows():
            context_parts.append(
                f"  {pr['exercise']}: best set {pr['best_weight']}kg × reps, "
                f"est. 1RM {pr['best_1rm']:.0f}kg"
            )

    if not volume_summary.empty:
        context_parts.append(f"\n=== VOLUME BY MUSCLE GROUP (total kg) ===")
        total_vol = volume_summary["total_volume"].sum()
        for _, row in volume_summary.iterrows():
            pct = row["total_volume"] / total_vol * 100 if total_vol > 0 else 0
            context_parts.append(f"  {row['muscle_group']}: {row['total_volume']:,.0f} kg ({pct:.0f}%)")

    return "\n".join(context_parts)


ANALYSIS_QUERIES = {
    "assess_progress": "Analyze my overall training progress. What are my strengths, weaknesses, and what should I focus on next?",
    "weak_points": "Identify my weak points based on the training data. Which muscle groups are lagging, and what exercises should I add?",
    "plateau_analysis": "Analyze my training data for signs of plateaus. Are any exercises stalling? Should I deload or change my approach?",
    "volume_analysis": "Analyze my training volume. Am I doing enough/too much? How is my volume distributed across the week?",
    "program_recommendation": "Based on my data and goals, what type of program would be optimal for me next?",
    "frequency_analysis": "How is my training frequency per muscle group? Should I increase or decrease frequency for any muscle groups?",
}
