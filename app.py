import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="GymInsight",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_loader import load_and_process, refresh_cache
from src.muscle_mapping import load_mapping, PARENT_SUB_MAP, ALL_PARENTS, build_muscle_to_exercise_map, build_parent_to_exercise_map
from src.deepseek_client import DeepseekClient
from src.processor import (
    compute_weekly_volume, compute_exercise_progression, compute_frequency,
    compute_workout_distribution, compute_muscle_volume, rollup_muscle_volume,
    detect_prs, estimate_1rm,
)
from src.strength_level import (
    get_level, LEVELS, level_score, compute_overall_level,
    get_muscle_strength_scores, STANDARDS, get_lift_display_name,
)
from src.balance import (
    compute_strength_antagonist_balances, compute_strength_category_scores,
    compute_strength_issues, compute_strength_overall_score,
)
from src.prompts import (
    SYSTEM_PROMPT_ANALYST, SYSTEM_PROMPT_COACH, SYSTEM_PROMPT_PROGRAM_DESIGNER,
    build_analysis_context, ANALYSIS_QUERIES,
)
from src.program_builder import (
    Program, Workout, ExerciseSlot, Goal, evaluate_program, MUSCLE_GROUP_EXERCISES,
)
from src.adaptive_program import (
    compute_exercise_effectiveness, compute_muscle_effectiveness,
    suggest_alternatives, generate_adaptive_program,
)
from src.deepseek_client import DEFAULT_MODEL

DATA_PATH = Path(__file__).parent / "data" / "training_data.csv"


def _mini_chat(tab_key: str, context_text: str):
    client = st.session_state.get("deepseek")
    if client is None or not client.is_configured():
        return

    chat_key = f"app_chat_{tab_key}"

    with st.expander(f"🤖 Ask AI about this data", expanded=False):
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [
                {"role": "system", "content": context_text},
            ]

        msgs = st.session_state[chat_key]
        if msgs[0]["content"] != context_text:
            msgs[0]["content"] = context_text

        for msg in msgs:
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(
            f"Ask about this...",
            key=f"chat_input_{tab_key}",
        ):
            msgs.append({"role": "user", "content": prompt})
            model = st.session_state.get("selected_model", "deepseek-chat")
            response = client.chat(
                messages=msgs,
                model=model,
                temperature=0.7,
                max_tokens=1024,
                use_cache=True,
                conversation_id=f"app_{tab_key}",
            )
            if response["error"]:
                st.error(response["error"])
            else:
                content = response["result"]["content"]
                msgs.append({"role": "assistant", "content": content})
                st.rerun()


def _strength_bar(muscle_strength: dict[str, float], title: str) -> go.Figure:
    records = []
    for m, s in sorted(muscle_strength.items(), key=lambda x: x[0]):
        if s > 0:
            records.append({"Muscle": m, "Strength": s,
                            "Level": LEVELS[min(int(s), len(LEVELS) - 1)]})
    if not records:
        return go.Figure()
    df_plot = pd.DataFrame(records)
    fig = px.bar(df_plot, x="Strength", y="Muscle", orientation="h",
                 title=title, color="Strength", color_continuous_scale="RdYlGn",
                 range_color=[0, 5],
                 labels={"Strength": "Score (0–5)", "Muscle": ""})
    fig.update_layout(height=max(300, len(records) * 22))
    return fig


def _gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#ffebee"},
                {"range": [40, 70], "color": "#fff3e0"},
                {"range": [70, 100], "color": "#e8f5e9"},
            ],
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def init_session_state():
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True

    try:
        csv_path = str(DATA_PATH) if DATA_PATH.exists() else None
        df = load_and_process(csv_path)
        st.session_state.df = df
    except FileNotFoundError as e:
        st.session_state.df = pd.DataFrame()
        st.session_state.df_error = str(e)

    st.session_state.mapping = load_mapping()
    st.session_state.deepseek = DeepseekClient()

    from src.program_builder import Program
    st.session_state.current_program = Program(
        name="My Program",
        weeks=8,
        workouts_per_week=4,
    )

    st.session_state.profile = {
        "age": 41,
        "gender": "male",
        "bodyweight_kg": 75.0,
        "height_cm": 173,
        "training_experience_years": 2,
        "current_goal": "hypertrophy",
    }

    st.session_state.selected_model = "deepseek-v4-flash"
    st.session_state.adaptive_program = None

    if "dashboard_filters" not in st.session_state:
        st.session_state.dashboard_filters = {
            "date_range": None,
            "exercises": [],
            "muscle_groups": [],
        }


init_session_state()

st.sidebar.title("GymInsight 💪")

if st.session_state.df.empty:
    st.sidebar.warning("No training data loaded")
    with st.sidebar:
        uploaded = st.file_uploader("Upload CSV from StrengthLog", type=["csv"])
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            st.session_state.df = load_and_process(tmp_path)
            st.rerun()
else:
    df = st.session_state.df
    total_workouts = df["session_date"].nunique()
    date_min = df["start"].min().strftime("%b %d, %Y")
    date_max = df["start"].max().strftime("%b %d, %Y")
    total_exercises = df["exercise"].nunique()

    st.sidebar.markdown(
        f"**{total_workouts}** workouts | **{total_exercises}** exercises"
    )
    st.sidebar.markdown(f"{date_min} – {date_max}")
    st.sidebar.divider()

    st.sidebar.page_link("app.py", label="🏠 Home", use_container_width=True)
    st.sidebar.page_link("pages/6_Settings.py", label="⚙️ Settings", use_container_width=True)

if st.session_state.df.empty:
    st.title("GymInsight")
    st.markdown("Your personal training analytics & AI coach")
    st.info(
        "No training data found. Upload a CSV file exported from StrengthLog "
        "using the sidebar to get started."
    )
    st.stop()

df = st.session_state.df
profile = st.session_state.profile
mapping = st.session_state.mapping
bw = profile.get("bodyweight_kg", 75)
age = profile.get("age", 30)

st.title("GymInsight")
st.markdown("Your personal training analytics & AI coach")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Workouts", df["session_date"].nunique())
with col2:
    st.metric("Total Volume", f"{df['volume_kg'].sum() / 1000:.0f}K kg")
with col3:
    st.metric("Unique Exercises", df["exercise"].nunique())
with col4:
    pr_data = detect_prs(df)
    levels = []
    for lift_name in ["Bench Press", "Deadlift", "Squat", "Seated Barbell Overhead Press"]:
        pr_row = pr_data[pr_data["exercise"] == lift_name]
        if not pr_row.empty:
            ratio = pr_row.iloc[0]["best_1rm"] / max(bw, 1)
            level, _ = get_level(ratio, lift_name, bw, age)
            levels.append(level)
    if levels:
        avg_idx = sum(level_score(l) for l in levels) / len(levels)
        overall = LEVELS[min(int(avg_idx), len(LEVELS) - 1)]
        st.metric("Overall Level", overall)

tabs = st.tabs([
    "📊 Dashboard",
    "🏋️ Strength Level",
    "⚖️ Muscle Balance",
    "🤖 AI Analysis",
    "📋 Program Builder",
    "🔄 Adaptive Program",
])

# ====================================================================
# TAB 0: DASHBOARD
# ====================================================================
with tabs[0]:
    muscle_to_ex = build_muscle_to_exercise_map(mapping)
    parent_to_ex = build_parent_to_exercise_map(mapping)

    filters = st.session_state.dashboard_filters

    all_sub_groups = []
    for subs in PARENT_SUB_MAP.values():
        all_sub_groups.extend(subs)
    all_sub_groups = sorted(set(all_sub_groups))

    all_parents_sorted = sorted(ALL_PARENTS)

    muscle_options = []
    muscle_options.extend(
        f"{p} (↓{len(PARENT_SUB_MAP.get(p, []))})" for p in all_parents_sorted if p in PARENT_SUB_MAP
    )
    muscle_options.extend(all_sub_groups)

    def expand_muscle_selection(selected: list[str]) -> list[str]:
        expanded = set()
        for s in selected:
            for p in all_parents_sorted:
                label = f"{p} (↓{len(PARENT_SUB_MAP.get(p, []))})"
                if s == label or s == p:
                    expanded.update(PARENT_SUB_MAP.get(p, []))
                elif s in all_sub_groups:
                    expanded.add(s)
        return sorted(expanded)

    date_min = df["start"].min()
    date_max = df["start"].max()

    with st.expander("🔍 Filters", expanded=True):
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        with col_f1:
            date_vals = st.slider(
                "Date range",
                min_value=date_min.to_pydatetime(),
                max_value=date_max.to_pydatetime(),
                value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
                format="YYYY-MM",
                key="dash_date_range",
            )
        with col_f2:
            muscle_labels = sorted(
                f"{p} (↓{len(PARENT_SUB_MAP.get(p, []))})"
                for p in all_parents_sorted if p in PARENT_SUB_MAP
            )
            muscle_labels += all_sub_groups

            selected_muscle_labels = st.multiselect(
                "Muscle groups",
                options=muscle_labels,
                default=filters["muscle_groups"],
                placeholder="All muscle groups",
                key="dash_muscle_filter",
            )
        with col_f3:
            if st.button("↺ Reset filters", use_container_width=True, key="dash_reset"):
                st.session_state.dashboard_filters = {
                    "date_range": None, "exercises": [], "muscle_groups": [],
                }
                st.rerun()

    selected_sub_groups = expand_muscle_selection(selected_muscle_labels)
    if selected_sub_groups:
        valid_exercises_for_muscle = set()
        for mg in selected_sub_groups:
            valid_exercises_for_muscle |= muscle_to_ex.get(mg, set())
    else:
        valid_exercises_for_muscle = None

    all_exercises_sorted = sorted(df["exercise"].unique())
    if valid_exercises_for_muscle:
        exercise_options = sorted(
            e for e in all_exercises_sorted if e in valid_exercises_for_muscle
        )
    else:
        exercise_options = all_exercises_sorted

    selected_exercises = st.multiselect(
        "Exercises",
        options=exercise_options,
        default=filters["exercises"],
        placeholder="All exercises",
        key="dash_exercise_filter",
    )

    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["start"] >= pd.Timestamp(date_vals[0]))
        & (filtered_df["start"] <= pd.Timestamp(date_vals[1]))
    ]
    if selected_sub_groups and valid_exercises_for_muscle:
        filtered_df = filtered_df[filtered_df["exercise"].isin(valid_exercises_for_muscle)]
    if selected_exercises:
        filtered_df = filtered_df[filtered_df["exercise"].isin(selected_exercises)]

    st.session_state.dashboard_filters["muscle_groups"] = selected_muscle_labels
    st.session_state.dashboard_filters["exercises"] = selected_exercises

    filtered_df_empty = filtered_df.empty or filtered_df[~filtered_df["warmup"]].empty

    if filtered_df_empty:
        st.info("No data matches the current filters. Adjust your selection.")
        st.stop()

    sub_tabs = st.tabs(["Volume", "Progression", "Frequency", "Workouts"])

    with sub_tabs[0]:
        weekly = compute_weekly_volume(filtered_df)
        if not weekly.empty:
            view = st.radio("View", ["All Time", "Last 16 Weeks", "Last 8 Weeks"],
                            horizontal=True, label_visibility="collapsed", key="dash_vol_view")
            if view == "Last 16 Weeks":
                weekly = weekly.tail(16)
            elif view == "Last 8 Weeks":
                weekly = weekly.tail(8)

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(weekly, x="week_label", y="total_volume",
                             title="Weekly Volume (kg)",
                             labels={"week_label": "Week", "total_volume": "Total Volume (kg)"},
                             color_discrete_sequence=["#4CAF50"])
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.line(weekly, x="week_label", y="avg_weight",
                               title="Average Weight Per Set (kg)",
                               labels={"week_label": "Week", "avg_weight": "Avg Weight (kg)"},
                               markers=True, color_discrete_sequence=["#FF9800"])
                fig2.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                fig3 = px.bar(weekly, x="week_label", y="session_count",
                              title="Sessions Per Week",
                              labels={"week_label": "Week", "session_count": "Sessions"},
                              color_discrete_sequence=["#2196F3"])
                fig3.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)
            with col_d:
                fig4 = px.bar(weekly, x="week_label", y="total_sets",
                              title="Sets Per Week",
                              labels={"week_label": "Week", "total_sets": "Sets"},
                              color_discrete_sequence=["#9C27B0"])
                fig4.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig4, use_container_width=True)

    with sub_tabs[1]:
        st.subheader("Exercise Progression")
        prog_exercises = sorted(filtered_df["exercise"].unique())
        selected_exercise = st.selectbox("Select Exercise", prog_exercises, key="dash_prog_ex")
        prog = compute_exercise_progression(filtered_df, selected_exercise)
        if not prog.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.line(prog, x="session_date", y="max_weight",
                              title=f"Weight Progression",
                              markers=True, color_discrete_sequence=["#4CAF50"])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.line(prog, x="session_date", y="max_1rm",
                               title=f"Estimated 1RM",
                               markers=True, color_discrete_sequence=["#FF5722"])
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No progression data for this exercise.")

    with sub_tabs[2]:
        st.subheader("Training Frequency")
        freq = compute_frequency(filtered_df)
        if not freq.empty:
            view_f = st.radio("View", ["All Time", "Last 16 Weeks"],
                              horizontal=True, key="freq_view", label_visibility="collapsed")
            fdf = freq.tail(16) if view_f == "Last 16 Weeks" else freq
            fig = px.bar(fdf, x="week_label", y="workout_days",
                         title="Training Days Per Week",
                         labels={"week_label": "Week", "workout_days": "Days Trained"},
                         color_discrete_sequence=["#2196F3"])
            fig.update_layout(height=450, xaxis_tickangle=-45)
            fig.add_hline(y=4, line_dash="dash", line_color="red",
                          annotation_text="Target: 4 days")
            st.plotly_chart(fig, use_container_width=True)
            avg_days = fdf["workout_days"].mean()
            consistency = (fdf["workout_days"] >= 3).mean() * 100
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Days/Week", f"{avg_days:.1f}")
            col_b.metric("Consistency (≥3 days)", f"{consistency:.0f}%")
            col_c.metric("Total Weeks", len(fdf))

    with sub_tabs[3]:
        st.subheader("Workout Distribution")
        dist = compute_workout_distribution(filtered_df)
        if not dist.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.pie(dist, values="total_sessions", names="program",
                             title="Sessions by Program",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.pie(dist, values="total_volume", names="program",
                              title="Volume by Program",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recent PRs")
        prs = detect_prs(filtered_df)
        if not prs.empty:
            cols = ["exercise", "best_weight", "best_1rm", "best_volume", "last_date"]
            display = prs[cols].copy()
            display["best_weight"] = display["best_weight"].round(1)
            display["best_1rm"] = display["best_1rm"].round(0).astype(int)
            display["best_volume"] = display["best_volume"].round(0).astype(int)
            display["last_date"] = pd.to_datetime(display["last_date"]).dt.strftime("%b %d")
            display.columns = ["Exercise", "Best Weight", "Est. 1RM", "Best Volume", "Last"]
            st.dataframe(
                display.sort_values("Est. 1RM", ascending=False).head(20),
                use_container_width=True, hide_index=True,
            )

# ====================================================================
# TAB 1: STRENGTH LEVEL
# ====================================================================
with tabs[1]:
    pr_data = detect_prs(df)
    key_lifts = ["Bench Press", "Deadlift", "Squat", "Seated Barbell Overhead Press",
                 "Barbell Row", "Lat Pulldown"]
    lift_levels = {}
    for lift in key_lifts:
        pr_row = pr_data[pr_data["exercise"] == lift]
        if not pr_row.empty:
            est_1rm = pr_row.iloc[0]["best_1rm"]
            ratio = est_1rm / max(bw, 1)
            level, progress = get_level(ratio, lift, bw, age)
            lift_levels[lift] = {
                "est_1rm": est_1rm, "ratio": ratio, "level": level,
                "progress": progress, "best_weight": pr_row.iloc[0]["best_weight"],
                "last_date": pr_row.iloc[0]["last_date"],
            }

    overall_level, overall_progress = compute_overall_level(
        {k: (v["level"], v["progress"]) for k, v in lift_levels.items()},
        bw, age,
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Overall Strength Level", overall_level,
                  f"Progress: {overall_progress * 100:.0f}% to next level")
    with col_b:
        if lift_levels:
            avg_r = sum(v["ratio"] for v in lift_levels.values()) / len(lift_levels)
            st.metric("Avg Strength-to-Weight Ratio", f"{avg_r:.2f}x BW")
    with col_c:
        st.metric("Bodyweight", f"{bw} kg")

    sub_tabs = st.tabs(["Key Lifts", "All Exercises", "Standards Comparison"])

    with sub_tabs[0]:
        rows = []
        for lift in key_lifts:
            if lift in lift_levels:
                info = lift_levels[lift]
                rows.append({
                    "Lift": get_lift_display_name(lift),
                    "Best Set": f'{info["best_weight"]:.1f} kg',
                    "Est. 1RM": f'{info["est_1rm"]:.0f} kg',
                    "× BW": f'{info["ratio"]:.2f}x',
                    "Level": info["level"],
                    "Progress": f'{info["progress"] * 100:.0f}%',
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            fig = go.Figure()
            for lift in key_lifts:
                if lift in lift_levels:
                    li = lift_levels[lift]
                    fig.add_trace(go.Bar(
                        name=get_lift_display_name(lift),
                        y=[get_lift_display_name(lift)],
                        x=[level_score(li["level"]) + li["progress"]],
                        orientation="h",
                        text=f'{li["level"]} ({li["progress"]*100:.0f}%)',
                        textposition="inside",
                    ))
            fig.update_layout(
                title="Strength Levels (0=Untrained, 5=Elite)",
                xaxis=dict(tickmode="array", tickvals=list(range(len(LEVELS))),
                           ticktext=LEVELS, range=[0, len(LEVELS)]),
                height=350, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with sub_tabs[1]:
        pr_data["level"] = pr_data.apply(
            lambda r: get_level(r["best_1rm"] / max(bw, 1), r["exercise"], bw, age)[0]
            if r["best_1rm"] > 0 else "N/A", axis=1,
        )
        display = pr_data[["exercise", "best_weight", "best_1rm", "total_sets", "level", "last_date"]].copy()
        display["best_weight"] = display["best_weight"].round(1)
        display["best_1rm"] = display["best_1rm"].round(0).astype(int)
        display["last_date"] = pd.to_datetime(display["last_date"]).dt.strftime("%b %d, %Y")
        display.columns = ["Exercise", "Best Weight", "Est. 1RM", "Sets", "Level", "Last Date"]
        st.dataframe(display.sort_values("Est. 1RM", ascending=False),
                     use_container_width=True, hide_index=True)

    with sub_tabs[2]:
        if lift_levels:
            fig = go.Figure()
            for lift in key_lifts:
                if lift in lift_levels:
                    display_name = get_lift_display_name(lift)
                    standards = STANDARDS.get(lift, STANDARDS["Bench Press"])
                    user_ratio = lift_levels[lift]["ratio"]
                    fig.add_trace(go.Scatter(
                        x=LEVELS, y=[standards[l]["min"] for l in LEVELS],
                        mode="lines+markers", name=display_name, line=dict(width=2),
                    ))
                    fig.add_hline(y=user_ratio, line_dash="dash", line_color="red",
                                  annotation_text=f"{display_name}: {user_ratio:.2f}x")
            fig.update_layout(title="Strength Standards by Level (× Bodyweight)",
                              xaxis_title="Level", yaxis_title="Ratio (Lift / BW)",
                              height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# TAB 2: MUSCLE BALANCE
# ====================================================================
with tabs[2]:
    vol_summary = compute_muscle_volume(df, mapping)
    parent_summary = rollup_muscle_volume(vol_summary)
    vol_dict = vol_summary.set_index("muscle_group")["total_volume"].to_dict()
    muscle_strength = get_muscle_strength_scores(df, mapping, bw, age)
    str_pairs = compute_strength_antagonist_balances(muscle_strength)
    str_cats = compute_strength_category_scores(muscle_strength)
    str_issues = compute_strength_issues(str_pairs, str_cats, muscle_strength, vol_dict)
    str_score = compute_strength_overall_score(str_pairs, str_cats)

    bal_tabs = st.tabs(["Overall", "Antagonist Pairs", "Push/Pull/Leg", "Upper/Lower", "Volume Reference"])

    with bal_tabs[0]:
        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            c = "#4CAF50" if str_score >= 0.7 else "#FF9800" if str_score >= 0.4 else "#F44336"
            lbl = "Balanced" if str_score >= 0.7 else "Fair" if str_score >= 0.4 else "Needs Work"
            st.plotly_chart(_gauge(str_score, f"Strength Balance — {lbl}", c), use_container_width=True)
        with col_b:
            st.subheader("Strength Level Averages")
            for cat, label in [("push", "Push"), ("pull", "Pull"), ("leg", "Leg"), ("core", "Core"),
                               ("upper", "Upper Body"), ("lower", "Lower Body")]:
                val = str_cats.get(cat, 0)
                st.markdown(f"**{label}:** {val:.1f} — {LEVELS[min(int(val), len(LEVELS)-1)]}")
        st.divider()
        st.subheader("Issues & Warnings")
        for iss in str_issues:
            icon = {"warning": "⚠️", "info": "ℹ️", "error": "🚨"}.get(iss["severity"], "ℹ️")
            st.markdown(f"**{icon} {iss['title']}**")
            st.markdown(iss["message"])
            st.divider()
        st.subheader("Volume Distribution (for reference)")
        if not parent_summary.empty:
            fig = px.pie(parent_summary, values="total_volume", names="muscle_group",
                         color_discrete_sequence=px.colors.qualitative.Set3, hole=0.3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        ctx = (
            f"Overall strength balance score: {str_score*100:.0f}%. "
            f"Push avg: {str_cats.get('push',0):.1f}, Pull avg: {str_cats.get('pull',0):.1f}, "
            f"Leg avg: {str_cats.get('leg',0):.1f}, Core avg: {str_cats.get('core',0):.1f}. "
            f"Issues: {' | '.join(i['message'] for i in str_issues) if str_issues else 'None'}."
        )
        _mini_chat("balance_overall", ctx)

    with bal_tabs[1]:
        st.subheader("Antagonist Pair Strength Comparison")
        for p in str_pairs:
            s1, s2 = p["strength_1"], p["strength_2"]
            col_a, col_m, col_b = st.columns([1, 0.3, 1])
            with col_a:
                st.metric(f"{p['muscle_1']}", f"{LEVELS[min(int(s1), len(LEVELS)-1)]}", f"score: {s1:.1f}")
            with col_m:
                st.markdown("<div style='text-align:center;font-size:2rem;margin-top:1.5rem'>↔</div>",
                            unsafe_allow_html=True)
            with col_b:
                st.metric(f"{p['muscle_2']}", f"{LEVELS[min(int(s2), len(LEVELS)-1)]}", f"score: {s2:.1f}")
            if p["weaker"]:
                st.warning(f"{p['weaker']} is **{abs(p['diff']):.1f}** levels weaker")
            st.divider()
        fig = _strength_bar(muscle_strength, "All Muscle Strength Scores")
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        ctx = "Antagonist pair strength comparison:\n" + "\n".join(
            f"  {p['muscle_1']}: {p['strength_1']:.1f} vs {p['muscle_2']}: {p['strength_2']:.1f} — {p['status']}"
            for p in str_pairs
        )
        _mini_chat("balance_antagonist", ctx)

    with bal_tabs[2]:
        st.subheader("Push / Pull / Leg Strength")
        cats_data = [
            ("Push", str_cats.get("push", 0), "#FF5252"),
            ("Pull", str_cats.get("pull", 0), "#448AFF"),
            ("Leg", str_cats.get("leg", 0), "#66BB6A"),
            ("Core", str_cats.get("core", 0), "#AB47BC"),
        ]
        cols = st.columns(4)
        for i, (label, val, color) in enumerate(cats_data):
            with cols[i]:
                st.markdown(
                    f"<div style='background:{color}22;padding:1rem;border-radius:0.75rem;"
                    f"border-left:4px solid {color};text-align:center'>"
                    f"<span style='color:{color};font-weight:700'>{label}</span><br>"
                    f"<span style='font-size:1.8rem;font-weight:800'>{val:.1f}</span><br>"
                    f"<span style='color:#94a3b8'>{LEVELS[min(int(val), len(LEVELS)-1)]}</span></div>",
                    unsafe_allow_html=True,
                )
        cat_df = pd.DataFrame([{"Category": c[0], "Strength": c[1]} for c in cats_data])
        fig = px.bar(cat_df, x="Category", y="Strength", title="Strength Averages",
                     color="Category", color_discrete_sequence=[c[2] for c in cats_data])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        PUSH_NAMES = {"Chest: Upper", "Chest: Middle", "Chest: Lower",
                      "Shoulders: Front", "Shoulders: Side", "Triceps"}
        PULL_NAMES = {"Back: Lats", "Back: Traps", "Back: Lower", "Back: Rhomboids",
                      "Shoulders: Rear", "Biceps", "Forearms"}
        LEG_NAMES = {"Quads", "Hamstrings", "Glutes", "Calves", "Hip Flexors"}
        CORE_NAMES = {"Core: Upper", "Core: Lower", "Core: Obliques"}
        names_map = {"Push": PUSH_NAMES, "Pull": PULL_NAMES, "Leg": LEG_NAMES, "Core": CORE_NAMES}
        show_cat = st.radio("Show muscles in:", list(names_map.keys()), horizontal=True, key="balance_show_cat")
        fig2 = _strength_bar(
            {k: v for k, v in muscle_strength.items() if k in names_map[show_cat]},
            f"{show_cat} Muscles",
        )
        if fig2.data:
            st.plotly_chart(fig2, use_container_width=True)

    with bal_tabs[3]:
        st.subheader("Upper vs Lower Body Strength")
        upper_val = str_cats.get("upper", 0)
        lower_val = str_cats.get("lower", 0)
        ratio = upper_val / max(lower_val, 0.01)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Upper Body", f"{LEVELS[min(int(upper_val), 5)]}", f"{upper_val:.1f}")
        col_b.metric("Lower Body", f"{LEVELS[min(int(lower_val), 5)]}", f"{lower_val:.1f}")
        col_c.metric("Ratio", f"{ratio:.2f}:1",
                     "Lower stronger" if ratio < 0.8 else "Balanced" if ratio <= 1.5 else "Upper stronger")
        ul_df = pd.DataFrame({"Category": ["Upper Body", "Lower Body"], "Strength": [upper_val, lower_val]})
        fig = px.bar(ul_df, x="Category", y="Strength", title="Upper / Lower Body Strength",
                     color="Category", color_discrete_sequence=["#2196F3", "#66BB6A"])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with bal_tabs[4]:
        st.subheader("Volume Reference (Parent Groups)")
        if not parent_summary.empty:
            display = parent_summary.copy()
            display["total_volume"] = display["total_volume"].round(0).astype(int)
            display["avg_weight"] = display["avg_weight"].round(1)
            display.columns = ["Muscle Group", "Volume (kg)", "Sets", "Reps", "Avg Weight", "Exercises"]
            st.dataframe(display, use_container_width=True, hide_index=True)
            fig = px.bar(display, x="Volume (kg)", y="Muscle Group", orientation="h",
                         title="Volume by Parent Group", color="Volume (kg)",
                         color_continuous_scale="Viridis")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        parent_choice = st.selectbox("Drill into:", ["All"] + sorted(ALL_PARENTS), key="balance_drill")
        if parent_choice != "All":
            subs = PARENT_SUB_MAP.get(parent_choice, [parent_choice])
            sub = vol_summary[vol_summary["muscle_group"].isin(subs)].copy()
            if not sub.empty:
                sub["total_volume"] = sub["total_volume"].round(0).astype(int)
                sub["avg_weight"] = sub["avg_weight"].round(1)
                sdisp = sub[["muscle_group", "total_volume", "total_sets",
                             "total_reps", "avg_weight", "exercise_count"]]
                sdisp.columns = ["Sub-group", "Volume", "Sets", "Reps", "Avg Wt", "Ex"]
                st.dataframe(sdisp, use_container_width=True, hide_index=True)

# ====================================================================
# TAB 3: AI ANALYSIS
# ====================================================================
with tabs[3]:
    deepseek = st.session_state.deepseek
    if not deepseek.is_configured():
        st.warning("Deepseek API key not configured. Go to **Settings** to set up your API key.")
    else:
        cache_info = deepseek.get_cached_context()
        with st.sidebar:
            st.subheader("🤖 Model & Context")
            st.session_state.selected_model = st.text_input(
                "Model", value=st.session_state.get("selected_model", DEFAULT_MODEL),
                placeholder="deepseek-chat",
            )
            st.divider()
            st.caption("**Token Usage (Session)**")
            st.metric("Total Tokens", f"{cache_info['total_tokens']:,}")
            st.metric("Cached Tokens", f"{cache_info['total_cached_tokens']:,}")
            st.metric("Cache Ratio", f"{cache_info['cache_ratio'] * 100:.0f}%")
            st.metric("Estimated Cost", f"${cache_info['total_cost']:.4f}")
            if st.button("Reset Stats", use_container_width=True):
                deepseek.reset_stats()
                st.rerun()
            st.divider()
            sys_prompt_option = st.selectbox(
                "Assistant Role",
                ["Training Analyst", "Personal Coach", "Program Designer"],
                index=0, key="ai_role",
            )
            role_map = {
                "Training Analyst": SYSTEM_PROMPT_ANALYST,
                "Personal Coach": SYSTEM_PROMPT_COACH,
                "Program Designer": SYSTEM_PROMPT_PROGRAM_DESIGNER,
            }
            st.session_state.system_prompt = role_map[sys_prompt_option]

        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = SYSTEM_PROMPT_ANALYST

        if "ai_messages" not in st.session_state:
            pr_data_ai = detect_prs(df)
            vol_summary_ai = compute_muscle_volume(df, mapping)
            context = build_analysis_context(df, pr_data_ai, vol_summary_ai, profile)
            st.session_state.ai_context = context
            st.session_state.ai_messages = [
                {"role": "system", "content": st.session_state.system_prompt},
                {"role": "system", "content": context},
            ]

        if st.button("🔄 Refresh Context", use_container_width=True, key="ai_refresh"):
            pr_data_ai = detect_prs(df)
            vol_summary_ai = compute_muscle_volume(df, mapping)
            context = build_analysis_context(df, pr_data_ai, vol_summary_ai, profile)
            st.session_state.ai_context = context
            st.session_state.ai_messages = [
                {"role": "system", "content": st.session_state.system_prompt},
                {"role": "system", "content": context},
            ]
            st.rerun()

        if "system_prompt" in st.session_state:
            st.session_state.ai_messages[0] = {
                "role": "system", "content": st.session_state.system_prompt,
            }

        st.subheader("Quick Actions")
        qcols = st.columns(4)
        for i, (key, query) in enumerate(ANALYSIS_QUERIES.items()):
            with qcols[i % 4]:
                if st.button(query.split(".")[0] if "." in query else query[:30],
                             help=query, use_container_width=True, key=f"qa_{key}"):
                    st.session_state.ai_messages.append({"role": "user", "content": query})
                    response = deepseek.chat(
                        messages=st.session_state.ai_messages,
                        model=st.session_state.selected_model,
                        use_cache=True, conversation_id="analysis",
                    )
                    if response["error"]:
                        st.error(f"API Error: {response['error']}")
                    else:
                        st.session_state.ai_messages.append(
                            {"role": "assistant", "content": response["result"]["content"]}
                        )
                        st.rerun()

        st.divider()
        for msg in st.session_state.ai_messages:
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your training..."):
            st.session_state.ai_messages.append({"role": "user", "content": prompt})
            response = deepseek.chat(
                messages=st.session_state.ai_messages,
                model=st.session_state.selected_model,
                use_cache=True, conversation_id="analysis",
            )
            if response["error"]:
                st.error(f"API Error: {response['error']}")
            else:
                result = response["result"]
                st.session_state.ai_messages.append(
                    {"role": "assistant", "content": result["content"]}
                )
                with st.expander("Response Details"):
                    st.json({
                        "model": result["model"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "cached_tokens": result["cached_tokens"],
                        "cost": f"${result['cost']:.4f}",
                        "elapsed": f"{result['elapsed']:.2f}s",
                    })
                st.rerun()

# ====================================================================
# TAB 4: PROGRAM BUILDER
# ====================================================================
with tabs[4]:
    pb_tabs = st.tabs(["Manual Builder", "AI Generator", "Evaluation"])

    with pb_tabs[0]:
        st.subheader("Manual Program Builder")
        current_program = st.session_state.current_program

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            prog_name = st.text_input("Program Name", value=current_program.name, key="pb_name")
        with col_b:
            goal = st.selectbox("Goal", options=[g.value for g in Goal],
                                index=[g.value for g in Goal].index(current_program.goal.value),
                                key="pb_goal")
        with col_c:
            weeks = st.number_input("Weeks", min_value=1, max_value=52, value=current_program.weeks, key="pb_weeks")

        wpw = st.number_input("Workouts Per Week", min_value=1, max_value=7,
                              value=current_program.workouts_per_week, key="pb_wpw")

        current_program.name = prog_name
        current_program.goal = Goal(goal)
        current_program.weeks = int(weeks)
        current_program.workouts_per_week = int(wpw)

        while len(current_program.workouts) < current_program.workouts_per_week:
            current_program.workouts.append(
                Workout(name=f"Day {len(current_program.workouts) + 1}",
                        day_of_week=len(current_program.workouts))
            )

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        all_ex_names = sorted(set(
            ex for ex_list in MUSCLE_GROUP_EXERCISES.values() for ex in ex_list
        ))

        for i, workout in enumerate(current_program.workouts):
            with st.expander(f"{day_names[workout.day_of_week]} - {workout.name}", expanded=(i == 0)):
                wcol1, wcol2 = st.columns([3, 1])
                with wcol1:
                    w_name = st.text_input("Workout Name", value=workout.name, key=f"pb_wname_{i}")
                with wcol2:
                    w_day = st.selectbox("Day", day_names, index=workout.day_of_week, key=f"pb_wday_{i}")
                workout.name = w_name
                workout.day_of_week = day_names.index(w_day)

                num_ex = st.number_input("Number of Exercises", min_value=0, max_value=15,
                                         value=len(workout.exercises), key=f"pb_num_ex_{i}")
                while len(workout.exercises) < num_ex:
                    workout.exercises.append(ExerciseSlot(name=all_ex_names[0] if all_ex_names else "New"))
                if len(workout.exercises) > num_ex:
                    workout.exercises = workout.exercises[:num_ex]

                for j, ex in enumerate(workout.exercises):
                    scol1, scol2, scol3, scol4, scol5 = st.columns([3, 1, 1, 1, 1])
                    with scol1:
                        ex_name = st.selectbox("Exercise", all_ex_names,
                                               index=all_ex_names.index(ex.name) if ex.name in all_ex_names else 0,
                                               key=f"pb_ex_name_{i}_{j}", label_visibility="collapsed")
                    with scol2:
                        sets = st.number_input("Sets", 1, 20, ex.sets, key=f"pb_ex_sets_{i}_{j}", label_visibility="collapsed")
                    with scol3:
                        reps = st.number_input("Reps", 1, 100, ex.reps, key=f"pb_ex_reps_{i}_{j}", label_visibility="collapsed")
                    with scol4:
                        rpe = st.number_input("RPE", 0.0, 10.0, ex.rpe or 0.0, step=0.5,
                                              key=f"pb_ex_rpe_{i}_{j}", label_visibility="collapsed")
                    with scol5:
                        rest = st.number_input("Rest (s)", 0, 600, ex.rest_seconds,
                                               key=f"pb_ex_rest_{i}_{j}", label_visibility="collapsed")
                    ex.name = ex_name
                    ex.sets = int(sets)
                    ex.reps = int(reps)
                    ex.rpe = rpe if rpe > 0 else None
                    ex.rest_seconds = int(rest)

        if st.button("Save Program", type="primary", use_container_width=True, key="pb_save"):
            st.session_state.current_program = current_program
            st.success("Program saved!")

    with pb_tabs[1]:
        st.subheader("AI Program Generator")
        deepseek_ai = st.session_state.deepseek
        if not deepseek_ai.is_configured():
            st.warning("Configure Deepseek API in Settings first.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                ai_goal = st.selectbox("Goal", ["strength", "hypertrophy", "endurance", "general_fitness"],
                                       index=1, key="pb_ai_goal")
                ai_days = st.slider("Days per week", 1, 7, 4, key="pb_ai_days")
            with col_b:
                ai_experience = st.selectbox("Experience", ["beginner", "novice", "intermediate", "advanced"],
                                             index=2, key="pb_ai_exp")
                ai_equipment = st.multiselect(
                    "Available Equipment",
                    ["barbell", "dumbbell", "cables", "machines", "kettlebell",
                     "bands", "smith machine", "trap bar", "ez bar"],
                    default=["barbell", "dumbbell", "cables", "machines"],
                    key="pb_ai_equip",
                )
            ai_preferences = st.text_area("Additional Preferences (optional)",
                                          placeholder="E.g., focus on legs, avoid exercises that hurt my shoulders",
                                          key="pb_ai_prefs")

            if st.button("Generate Program with AI", type="primary", use_container_width=True, key="pb_ai_gen"):
                prompt = f"""Create a {ai_goal} training program for a {ai_experience} level trainee.
Profile:
- Age: {profile.get('age', 30)}
- Bodyweight: {profile.get('bodyweight_kg', 80)} kg
- Training experience: {profile.get('training_experience_years', 2)} years

Program Requirements:
- Goal: {ai_goal}
- Days per week: {ai_days}
- Available equipment: {', '.join(ai_equipment)}
- Weeks: 8
Preferences: {ai_preferences if ai_preferences else 'None'}

Return the program as structured JSON with:
{{"name": "...", "description": "...", "weeks": 8, "workouts_per_week": {ai_days},
  "progression_model": "...", "workouts": [
    {{"name": "...", "day_of_week": 0-6, "exercises": [
      {{"name": "...", "sets": N, "reps": N, "rpe": N, "rest_seconds": N, "notes": ""}}
    ]}}
  ]
}}"""
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT_COACH},
                    {"role": "user", "content": prompt},
                ]
                with st.spinner("Generating program..."):
                    response = deepseek_ai.chat(
                        messages=msgs, model=st.session_state.get("selected_model", "deepseek-chat"),
                        temperature=0.3, max_tokens=4096,
                    )
                if response["error"]:
                    st.error(f"Error: {response['error']}")
                else:
                    content = response["result"]["content"]
                    try:
                        import json
                        json_start = content.index("{")
                        json_end = content.rindex("}") + 1
                        program_json = json.loads(content[json_start:json_end])
                        program = Program(**program_json)
                        st.session_state.current_program = program
                        st.success("Program generated! Go to the Evaluation tab.")
                        with st.expander("AI Response"):
                            st.markdown(content)
                    except (ValueError, json.JSONDecodeError) as e:
                        st.error(f"Could not parse program: {e}")
                        with st.expander("Raw AI Response"):
                            st.text(content)

    with pb_tabs[2]:
        st.subheader("Program Evaluation")
        current_program = st.session_state.current_program
        st.markdown(f"**{current_program.name}** — {current_program.goal.value}")
        st.markdown(f"{current_program.weeks} weeks, {current_program.workouts_per_week} days/week")

        if current_program.workouts:
            for w in current_program.workouts:
                day_label = day_names[w.day_of_week] if w.day_of_week < 7 else "?"
                with st.expander(f"{day_label} - {w.name}"):
                    if w.exercises:
                        data = [{
                            "Exercise": e.name, "Sets": e.sets, "Reps": e.reps,
                            "RPE": e.rpe if e.rpe else "-", "Rest": f"{e.rest_seconds}s",
                        } for e in w.exercises]
                        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No exercises")
        else:
            st.info("No workouts defined yet.")

        if st.button("Evaluate Program", type="primary", use_container_width=True, key="pb_eval"):
            eval_result = evaluate_program(current_program)
            st.session_state.program_eval = eval_result
            st.rerun()

        if "program_eval" in st.session_state:
            eval_result = st.session_state.program_eval
            col_a, col_b = st.columns([1, 2])
            with col_a:
                c = "#4CAF50" if eval_result.overall_score >= 7 else "#FF9800" if eval_result.overall_score >= 5 else "#F44336"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=eval_result.overall_score,
                    title={"text": "Overall Score"}, domain={"x": [0, 1], "y": [0, 1]},
                    gauge={"axis": {"range": [0, 10]}, "bar": {"color": c}},
                    number={"suffix": "/10"},
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                scores = {
                    "Balance": eval_result.volume_balance_score,
                    "Progression": eval_result.progression_score,
                    "Selection": eval_result.exercise_selection_score,
                    "Recovery": eval_result.recovery_score,
                    "Goal": eval_result.goal_alignment_score,
                }
                scores_df = pd.DataFrame([{"Category": k, "Score": v} for k, v in scores.items()])
                fig2 = px.bar(scores_df, x="Category", y="Score", color="Score",
                              color_continuous_scale="RdYlGn", range_color=[0, 10])
                fig2.update_layout(height=250)
                st.plotly_chart(fig2, use_container_width=True)
            if eval_result.strengths:
                st.subheader("✅ Strengths")
                for s in eval_result.strengths:
                    st.markdown(f"- {s}")
            if eval_result.weaknesses:
                st.subheader("⚠️ Weaknesses")
                for w in eval_result.weaknesses:
                    st.markdown(f"- {w}")
            if eval_result.recommendations:
                st.subheader("📝 Recommendations")
                for r in eval_result.recommendations:
                    st.markdown(f"- {r}")

# ====================================================================
# TAB 5: ADAPTIVE PROGRAM
# ====================================================================
with tabs[5]:
    st.subheader("Exercise Effectiveness Analysis")
    st.markdown("Analyzes how effective each exercise is based on strength progress and volume efficiency.")

    if st.button("Compute Exercise Effectiveness", type="primary", use_container_width=True, key="ap_compute"):
        with st.spinner("Analyzing exercise effectiveness..."):
            muscle_strength = get_muscle_strength_scores(df, mapping, bw, age)
            effectiveness = compute_exercise_effectiveness(df, muscle_strength, bw, age)

        st.session_state.effectiveness_data = effectiveness
        st.rerun()

    if "effectiveness_data" in st.session_state and st.session_state.effectiveness_data:
        eff_data = st.session_state.effectiveness_data

        eff_df = pd.DataFrame([
            {
                "Exercise": ex,
                "Effectiveness": d["effectiveness"],
                "Efficiency": d["efficiency"],
                "Progression": d["progression"],
                "Strength": d["strength_score"],
                "Avg Weekly Vol": f"{d['avg_weekly_volume_kg']:.0f} kg",
                "Data Points": d["data_points"],
            }
            for ex, d in sorted(eff_data.items(), key=lambda x: -x[1]["effectiveness"])
        ])

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Exercises Analyzed", len(eff_df))
            avg_eff = eff_df["Effectiveness"].mean()
            st.metric("Avg Effectiveness", f"{avg_eff:.2f}/5")
        with col_b:
            fig = px.bar(eff_df.head(15), x="Effectiveness", y="Exercise",
                         orientation="h", title="Top 15 Exercises by Effectiveness",
                         color="Effectiveness", color_continuous_scale="RdYlGn",
                         range_color=[0, 5])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Effectiveness Table"):
            st.dataframe(eff_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Muscle Group Effectiveness")
        muscle_eff = compute_muscle_effectiveness(eff_data, mapping)
        muscle_rows = []
        for muscle, exercises in sorted(muscle_eff.items()):
            primary_eff = [e["effectiveness"] for e in exercises if e["role"] == "primary"]
            avg = np.mean(primary_eff) if primary_eff else np.mean([e["effectiveness"] for e in exercises])
            muscle_rows.append({"Muscle": muscle, "Avg Effectiveness": round(avg, 2)})

        if muscle_rows:
            mdf = pd.DataFrame(muscle_rows).sort_values("Avg Effectiveness")
            fig2 = px.bar(mdf, x="Avg Effectiveness", y="Muscle", orientation="h",
                          title="Muscle Group Effectiveness (Lagging at top)",
                          color="Avg Effectiveness", color_continuous_scale="RdYlGn_r",
                          range_color=[0, 5])
            fig2.update_layout(height=max(300, len(mdf) * 20))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Generate Adaptive Program")
        st.markdown("Creates a 4-day program prioritizing lagging muscle groups.")

        if st.button("Generate Program", type="primary", use_container_width=True, key="ap_generate"):
            with st.spinner("Building adaptive program..."):
                prog = generate_adaptive_program(df, mapping, profile, eff_data)
                st.session_state.adaptive_program = prog
            st.rerun()

    if st.session_state.get("adaptive_program"):
        prog = st.session_state.adaptive_program
        st.success(f"Program generated: **{prog.name}** ({prog.goal.value})")
        st.markdown(f"{prog.weeks} weeks, {prog.workouts_per_week} days/week")

        for w in prog.workouts:
            with st.expander(f"{w.name}"):
                if w.exercises:
                    data = [{
                        "Exercise": e.name, "Sets": e.sets, "Reps": e.reps,
                        "RPE": e.rpe if e.rpe else "-", "Rest": f"{e.rest_seconds}s",
                        "Notes": e.notes,
                    } for e in w.exercises]
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                else:
                    st.caption("No exercises")

        if st.button("Use as Current Program", use_container_width=True, key="ap_use"):
            st.session_state.current_program = prog
            st.success("Adaptive program set as current program! Go to Program Builder → Evaluation tab.")
    elif "effectiveness_data" not in st.session_state:
        st.info("Click 'Compute Exercise Effectiveness' above to start.")
