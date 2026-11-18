import streamlit as st
import os
import json
from pathlib import Path

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

from src.data_loader import refresh_cache
from src.deepseek_client import DEFAULT_MODEL, DEFAULT_BASE_URL
from src.muscle_mapping import load_mapping, save_mapping, PARENT_SUB_MAP, ALL_PARENTS, get_sub


def render():
    if "df" not in st.session_state or st.session_state.df.empty:
        st.warning("No training data loaded.")
        return

    st.title("⚙️ Settings")
    st.markdown("Configure API, profile, exercise mapping, and data management.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "API Configuration", "Profile", "Exercise Mapping", "Data Management"
    ])

    with tab1:
        st.subheader("Deepseek API Configuration")

        current_key = os.getenv("DEEPSEEK_API_KEY", "")
        current_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)

        api_key = st.text_input(
            "API Key",
            value=current_key,
            type="password",
            placeholder="sk-...",
            help="Your Deepseek API key. Stored in .env file.",
        )
        base_url = st.text_input(
            "Base URL",
            value=current_url,
            placeholder="https://api.deepseek.com",
        )

        chosen_model = st.text_input(
            "Default Model",
            value=st.session_state.get("selected_model", DEFAULT_MODEL),
            placeholder="deepseek-chat",
        )

        if st.button("Save API Settings", use_container_width=True):
            env_path = Path(__file__).parent.parent / ".env"
            env_content = f"DEEPSEEK_API_KEY={api_key}\nDEEPSEEK_BASE_URL={base_url}\n"
            with open(env_path, "w") as f:
                f.write(env_content)
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["DEEPSEEK_BASE_URL"] = base_url
            st.session_state.selected_model = chosen_model
            st.session_state.deepseek.api_key = api_key
            st.session_state.deepseek.base_url = base_url
            st.session_state.deepseek._client = None
            st.success("API settings saved!")

        st.divider()
        st.subheader("Connection Test")
        if st.button("Test Connection", use_container_width=True):
            deepseek = st.session_state.deepseek
            if deepseek.is_configured():
                with st.spinner("Testing connection..."):
                    response = deepseek.chat(
                        messages=[
                            {"role": "user",
                             "content": "Say 'Connection successful!' if you receive this."}
                        ],
                        model=chosen_model,
                        max_tokens=20,
                    )
                if response.get("error"):
                    st.error(f"Connection failed: {response['error']}")
                else:
                    st.success("Connection successful!")
                    st.write(response["result"]["content"])
            else:
                st.warning("No API key configured.")

    with tab2:
        st.subheader("User Profile")
        profile = st.session_state.profile

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=120,
                                  value=int(profile.get("age", 30)))
            gender = st.selectbox("Gender", ["male", "female"],
                                  index=0 if profile.get("gender") == "male" else 1)
            bw = st.number_input("Bodyweight (kg)", min_value=30.0, max_value=300.0,
                                 value=float(profile.get("bodyweight_kg", 80)), step=0.5)
        with col2:
            height = st.number_input("Height (cm)", min_value=100, max_value=250,
                                     value=int(profile.get("height_cm", 175)))
            experience = st.number_input("Training Experience (years)",
                                         min_value=0, max_value=50,
                                         value=int(profile.get("training_experience_years", 2)))
            goal = st.selectbox(
                "Current Goal",
                ["strength", "hypertrophy", "endurance", "general_fitness"],
                index=["strength", "hypertrophy", "endurance", "general_fitness"]
                .index(profile.get("current_goal", "hypertrophy")),
            )

        if st.button("Save Profile", use_container_width=True):
            st.session_state.profile = {
                "age": int(age),
                "gender": gender,
                "bodyweight_kg": float(bw),
                "height_cm": int(height),
                "training_experience_years": int(experience),
                "current_goal": goal,
            }
            profile_path = Path(__file__).parent.parent / "data" / "settings.json"
            with open(profile_path, "w") as f:
                json.dump(st.session_state.profile, f, indent=2)
            st.success("Profile saved!")

    with tab3:
        st.subheader("Exercise Mapping Editor")
        st.markdown("Map each exercise to its primary and secondary **muscle sub-groups**.")

        mapping = st.session_state.mapping
        all_exercises = sorted(st.session_state.df["exercise"].unique())

        all_sub_options = []
        for parent, subs in PARENT_SUB_MAP.items():
            for s in subs:
                all_sub_options.append(s)
        for p in ALL_PARENTS:
            if p not in {s.split(":")[0].strip() if ": " in s else s
                         for s in all_sub_options}:
                all_sub_options.append(p)
        all_sub_options = sorted(set(all_sub_options))

        search = st.text_input("🔍 Search exercises", placeholder="Type to filter...")
        filtered = [e for e in all_exercises
                    if search.lower() in e.lower()] if search else all_exercises

        ex_to_edit = st.selectbox("Select Exercise to Edit", filtered)

        if ex_to_edit:
            current = mapping.get(ex_to_edit, (["Unknown"], []))

            col1, col2 = st.columns(2)
            with col1:
                new_primary = st.multiselect(
                    "Primary Muscle Sub-groups",
                    all_sub_options,
                    default=[m for m in current[0] if m in all_sub_options],
                )
            with col2:
                new_secondary = st.multiselect(
                    "Secondary Muscle Sub-groups",
                    all_sub_options,
                    default=[m for m in current[1] if m in all_sub_options],
                )

            if st.button("Save Mapping", use_container_width=True):
                mapping[ex_to_edit] = (new_primary, new_secondary)
                save_mapping(mapping)
                st.session_state.mapping = mapping
                st.success(f"Saved mapping for {ex_to_edit}")
                st.rerun()

        unmapped = set(st.session_state.df["exercise"].unique()) - set(mapping.keys())
        if unmapped:
            st.warning(f"{len(unmapped)} exercises not yet mapped")
            with st.expander("Unmapped Exercises"):
                st.write(sorted(unmapped))
                if st.button("Classify with AI (requires Deepseek)"):
                    deepseek = st.session_state.deepseek
                    if deepseek.is_configured():
                        from src.muscle_mapping import update_mapping_with_unknown
                        with st.spinner("Classifying exercises..."):
                            mapping = update_mapping_with_unknown(
                                st.session_state.df, mapping, deepseek
                            )
                            st.session_state.mapping = mapping
                            st.success("Updated mappings!")
                            st.rerun()
                    else:
                        st.warning("Configure Deepseek API first.")

    with tab4:
        st.subheader("Data Management")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Refresh Data**")
            st.markdown("Re-process the CSV from scratch.")
            if st.button("🔄 Refresh Cache", use_container_width=True):
                try:
                    csv_path = Path(__file__).parent.parent / "data" / "training_data.csv"
                    df = refresh_cache(str(csv_path) if csv_path.exists() else None)
                    st.session_state.df = df
                    st.success("Cache refreshed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("**Upload New Data**")
            uploaded = st.file_uploader("Replace training CSV", type=["csv"])
            if uploaded:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                csv_dest = Path(__file__).parent.parent / "data" / "training_data.csv"
                import shutil
                shutil.copy2(tmp_path, csv_dest)
                df = refresh_cache(str(csv_dest))
                st.session_state.df = df
                st.success(f"Uploaded and processed {uploaded.name}")
                st.rerun()

        st.divider()
        st.markdown("**Statistics**")
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Exercises", df["exercise"].nunique())
        col3.metric("Workouts", df["session_date"].nunique())
        col4.metric("Programs", df["program"].nunique())

        if st.button("Reset All Data", type="secondary", use_container_width=True):
            confirm = st.checkbox("I understand this will delete all cached data")
            if confirm and st.button("Confirm Reset", type="primary"):
                cache_path = Path(__file__).parent.parent / "data" / "processed.parquet"
                if cache_path.exists():
                    cache_path.unlink()
                st.session_state.df = refresh_cache()
                st.success("Data reset complete!")
                st.rerun()


render()
