"""Student Task Prioritizer dashboard. Run: streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # engine modules use flat imports
import task_store as ts
from fuzzy_engine import get_engine, map_days_to_urgency, map_difficulty, map_workload

DIFFICULTY = {1: "Very easy", 2: "Easy", 3: "Moderate", 4: "Hard", 5: "Very hard"}

st.set_page_config(page_title="Student Task Prioritizer", layout="wide")
st.title("Student Task Prioritizer")
st.caption("Fuzzy priority (0-100) from deadline urgency, task difficulty and current workload.")


@st.cache_resource
def engine():
    return get_engine()


with st.sidebar:
    st.header("Add a task")
    with st.form("add_task", clear_on_submit=True):
        title = st.text_input("Task name")
        days = st.number_input("Days remaining (0 = today, negative = overdue)", -60, 365, 3, step=1)
        diff = st.select_slider("Difficulty", list(DIFFICULTY), 3, format_func=lambda k: f"{k} - {DIFFICULTY[k]}")
        others = st.number_input("Other pending assignments", 0, 30, 0, step=1)
        length = st.number_input("Estimated length (0 = not sure)", 0.0, 100.0, 0.0)
        if st.form_submit_button("Add task"):
            try:
                ts.add_task(title, days, diff, others, length or None)
                st.success(f"Added '{title.strip()}'.")
            except ValueError as e:
                st.error(str(e))

try:
    tasks, dropped = ts.load_tasks()
except ValueError as e:
    st.error(f"Could not read the task file: {e}")
    st.stop()
if dropped:
    st.warning(f"{dropped} invalid row(s) in data/my_tasks.csv were ignored; the file is untouched.")
if tasks.empty:
    st.info("No tasks yet. Add one in the sidebar, or start from the sample set.")
    if st.button("Load sample tasks"):
        ts.load_sample()
        st.rerun()
    st.stop()

try:
    ranked = ts.rank_tasks(tasks)
except (ValueError, KeyError) as e:
    st.error(f"Could not score tasks: {e}")
    st.stop()

k1, k2, k3 = st.columns(3)
k1.metric("Tasks", len(ranked))
k2.metric("Do first", ranked.iloc[0]["title"], f"{ranked.iloc[0]['fuzzy_priority']:.1f} / 100", delta_color="off")
k3.metric("Due within a day", int((ranked["days_remaining"] <= 1).sum()))

tab_rank, tab_why, tab_cmp = st.tabs(["Ranked tasks", "Why this score", "Compare methods"])
LEVEL = {"Very High": "🔴 Very High", "High": "🟠 High", "Medium": "🟡 Medium", "Low": "🟢 Low", "Very Low": "⚪ Very Low"}

with tab_rank:
    st.dataframe(
        ranked[["fuzzy_rank", "title", "days_remaining", "difficulty", "other_assignments",
                "fuzzy_priority", "fuzzy_category"]].assign(fuzzy_category=ranked["fuzzy_category"].map(lambda c: LEVEL.get(c, c))),
        hide_index=True, use_container_width=True,
        column_config={
            "fuzzy_priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.1f"),
            "fuzzy_rank": "Rank", "title": "Task", "days_remaining": "Days left",
            "difficulty": "Difficulty", "other_assignments": "Other tasks", "fuzzy_category": "Level",
        },
    )
    st.bar_chart(ranked.set_index("title")["fuzzy_priority"], y_label="Fuzzy priority (0-100)")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    gone = c1.selectbox("Remove a task", ranked.index, format_func=lambda i: ranked.at[i, "title"])
    if c2.button("Remove"):
        ts.delete_task(gone)
        st.rerun()

with tab_why:
    i = st.selectbox("Task", ranked.index, format_func=lambda i: ranked.at[i, "title"], key="why")
    r = ranked.loc[i]
    length_val = None if pd.isna(r["perceived_length"]) else r["perceived_length"]
    u = map_days_to_urgency(r["days_remaining"])
    d = map_difficulty(r["difficulty"])
    w = map_workload(r["other_assignments"], length_val)
    info = engine().explain(u, d, w)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Priority", f"{info['crisp_priority']:.1f}", info["priority_category"], delta_color="off")
    m2.metric("Urgency", f"{u:.1f} / 10")
    m3.metric("Difficulty", f"{d:.1f} / 10")
    m4.metric("Workload", f"{w:.1f} / 10")
    st.subheader("Fuzzy categories involved")
    cols = st.columns(3)
    for col, name in zip(cols, ["urgency", "difficulty", "workload"]):
        deg = {k.replace("_", " ").title(): v for k, v in info["membership_degrees"][name].items() if v > 0}
        col.markdown(f"**{name.title()}**")
        col.dataframe(pd.Series(deg, name="Membership").to_frame(), use_container_width=True)
    st.subheader(f"Rules that fired ({info['active_rules_count']})")
    st.dataframe(pd.DataFrame(info["active_rules"]).drop(columns="rationale", errors="ignore"),
                 hide_index=True, use_container_width=True)

with tab_cmp:
    top = lambda col: ranked.loc[ranked[col].idxmin(), "title"]
    st.markdown(
        f"**Do first** - fuzzy: {top('fuzzy_rank')} | deadline-only: {top('edf_rank')} | "
        f"weighted: {top('weighted_rank')}"
    )
    cmp = ranked[["title", "days_remaining", "fuzzy_priority", "fuzzy_rank",
                  "weighted_score", "weighted_rank", "edf_rank", "shift_fuzzy_vs_edf"]]
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.bar_chart(
        ranked.set_index("title")[["fuzzy_priority", "weighted_score", "edf_score"]]
        .rename(columns={"fuzzy_priority": "Fuzzy", "weighted_score": "Weighted", "edf_score": "Deadline-only"}),
        y_label="Score (0-100)",
    )
    st.caption(
        "Weighted = 0.5 × urgency + 0.3 × difficulty + 0.2 × workload. "
        "EDF sorts by days left, overdue first. "
        "'shift_fuzzy_vs_edf' > 0 means fuzzy ranks the task higher than deadline-only would."
    )
