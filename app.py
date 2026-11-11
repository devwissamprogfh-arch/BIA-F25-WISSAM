

"""SmartAcademicScheduler - Streamlit interface."""

import time
import tracemalloc
from pathlib import Path

import pandas as pd
import streamlit as st

from core.problem import load_problem
from core.validator import validate_schedule

from algorithms.genetic_algorithm import (
    GAConfig,
    genetic_algorithm,
)
from algorithms.particle_swarm import (
    PSOConfig,
    particle_swarm_optimization,
)
from algorithms.ant_colony import (
    ACOConfig,
    ant_colony_optimization,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartAcademicScheduler",
    page_icon="🎓",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "المحاضرات": {
        "Small": "data/lecture/small",
        "Medium": "data/lecture/medium",
        "Large": "data/lecture/large",
    },
    "الامتحانات": {
        "Small": "data/exam/small",
        "Medium": "data/exam/medium",
        "Large": "data/exam/large",
    },
}

ALGORITHMS = {
    "الخوارزمية الجينية (GA)": "GA",
    "سرب الجسيمات (PSO)": "PSO",
    "مستعمرة النمل (ACO)": "ACO",
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_problem(dataset_path):
    """Load and cache a problem instance."""
    return load_problem(dataset_path)


BUDGETS = {
    "سريع (للتجربة)": "quick",
    "قياسي (للتقرير)": "standard",
}


def history_series(history):
    """Return (iterations, best_fitness, best_hard) lists from any history format."""
    its, fit, hard = [], [], []
    for index, item in enumerate(history or []):
        if not isinstance(item, dict):
            its.append(index)
            fit.append(float(item))
            hard.append(None)
            continue
        value = item.get("best_fitness", item.get("iteration_best_fitness"))
        if value is None:
            continue
        its.append(item.get("generation", item.get("iteration", index)))
        fit.append(float(value))
        hard.append(item.get("best_hard_violations"))
    return its, fit, hard


def convergence_stats(history):
    """Convergence indicators computed from the Iteration -> Best Fitness history."""
    its, fit, hard = history_series(history)
    if not fit:
        return {}
    final = fit[-1]
    start = fit[0]
    last_it = its[-1] if its[-1] else 1

    it_best = next(it for it, f in zip(its, fit) if f <= final + 1e-9)

    gain = start - final
    if gain > 1e-9:
        it_95 = next(
            it for it, f in zip(its, fit) if start - f >= 0.95 * gain
        )
    else:
        it_95 = its[0]

    it_feasible = next(
        (it for it, h in zip(its, hard) if h is not None and h == 0),
        None,
    )

    return {
        "Initial Fitness": start,
        "Improvement %": (gain / start * 100.0) if start > 0 else 0.0,
        "Iter. to Best": it_best,
        "Iter. to 95% of gain": it_95,
        "Iter. to Feasible": it_feasible,
        "Budget (iterations)": last_it,
        "Best found at % of budget": it_best / last_it * 100.0,
    }


def run_algorithm(algorithm, problem, seed, budget="quick"):
    """Run one algorithm. ``quick`` is a small budget, ``standard`` the report budget."""

    quick = budget == "quick"

    if algorithm == "GA":
        config = GAConfig(
            population_size=20 if quick else 40,
            generations=20 if quick else 100,
            seed=seed,
        )
        return genetic_algorithm(problem, config)

    if algorithm == "PSO":
        config = PSOConfig(
            swarm_size=10 if quick else 30,
            iterations=20 if quick else 100,
            seed=seed,
        )
        return particle_swarm_optimization(problem, config)

    if algorithm == "ACO":
        config = ACOConfig(
            ants=3 if quick else 12,
            iterations=5 if quick else 30,
            seed=seed,
        )
        return ant_colony_optimization(problem, config)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_once(algorithm, problem, seed, budget, measure_memory):
    """Run once and measure wall time, CPU time, peak memory and validity."""

    if measure_memory:
        tracemalloc.start()

    wall0 = time.perf_counter()
    cpu0 = time.process_time()

    result = run_algorithm(algorithm, problem, seed, budget)

    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0

    peak_mb = None
    if measure_memory:
        peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

    validation = validate_schedule(problem, result.best_schedule)
    validation_text = validation.text().strip()
    valid = validation_text.startswith("VALID:")

    row = {
        "Seed": seed,
        "Best Fitness": result.best_fitness,
        "Hard Violations": result.best_hard_violations,
        "Soft Penalty": result.best_soft_penalty,
        "Valid": valid,
        "Wall Time (s)": wall,
        "CPU Time (s)": cpu,
        "Peak Memory (MB)": peak_mb,
    }
    row.update(convergence_stats(result.history))

    return row, result, validation_text


def stability_summary(runs_df):
    """Mean / std / min / max over repeated runs (stability of the results)."""

    def stats(column):
        values = runs_df[column].dropna().astype(float)
        if values.empty:
            return None
        mean = values.mean()
        std = values.std(ddof=0)
        return {
            "Metric": column,
            "Mean": mean,
            "Std": std,
            "Min": values.min(),
            "Max": values.max(),
            "CV %": (std / mean * 100.0) if mean else 0.0,
        }

    columns = [
        "Best Fitness",
        "Hard Violations",
        "Soft Penalty",
        "Wall Time (s)",
        "CPU Time (s)",
        "Peak Memory (MB)",
        "Iter. to Best",
        "Iter. to 95% of gain",
    ]

    rows = [s for s in (stats(c) for c in columns if c in runs_df) if s]
    return pd.DataFrame(rows)


def timetable_frames(problem, schedule):
    """Build (event list, grid-cell dictionary) for a schedule."""
    grid = problem.grid
    spd = grid.slots_per_day
    rows = []
    for i, event in enumerate(problem.events):
        s, r = schedule.slots[i], schedule.rooms[i]
        if s < 0 or r < 0:
            continue
        rows.append({
            "Event": problem.event_ids[i],
            "Course": event.course_id,
            "Teacher": event.teacher_id,
            "Day": grid.day_names[grid.day_of(s)],
            "Start": grid.label(s).split(" ")[1],
            "Slots": event.duration,
            "Room": problem.room_ids[r],
            "Students": event.size,
            "_slot": s,
        })
    events_df = pd.DataFrame(rows).sort_values(["_slot", "Room"])
    return events_df.drop(columns="_slot").reset_index(drop=True), rows


def weekly_grid(problem, rows, room_filter):
    """Hours x days table; every cell lists 'Room: Course' for the events in it."""
    grid = problem.grid
    spd = grid.slots_per_day
    cells = {}
    for row in rows:
        if room_filter != "الكل" and row["Room"] != room_filter:
            continue
        s = row["_slot"]
        for g in range(s, s + row["Slots"]):
            cells.setdefault(g, []).append(f"{row['Room']}: {row['Course']}")

    index = [grid.label(k).split(" ")[1] for k in range(spd)]
    data = {}
    for d, day in enumerate(grid.day_names):
        data[day] = [" | ".join(sorted(cells.get(d * spd + k, []))) for k in range(spd)]
    return pd.DataFrame(data, index=index)


def show_results(name, algorithm_label, problem_type, problem_size,
                 budget, runs, results, texts, problem):
    """Show only what the report needs: quality, time/resources, convergence,
    stability and the weekly timetable of the best run."""

    runs_df = pd.DataFrame(runs)
    n_runs = len(runs_df)
    n_valid = int(runs_df["Valid"].sum())

    best_idx = int(
        runs_df.sort_values(["Hard Violations", "Best Fitness"]).index[0]
    )
    best_run = runs_df.loc[best_idx]
    feasible_its = runs_df["Iter. to Feasible"].dropna()

    st.subheader(f"{name} - {problem_type} / {problem_size}")

    # ---- 1) quality -------------------------------------------------------------
    st.markdown("#### 1) جودة الحل (أفضل تشغيلة)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Fitness", f"{best_run['Best Fitness']:.2f}")
    c2.metric("Hard Violations", int(best_run["Hard Violations"]))
    c3.metric("Soft Penalty", f"{best_run['Soft Penalty']:.2f}")
    c4.metric("تشغيلات صالحة", f"{n_valid} / {n_runs}")

    if best_run["Valid"]:
        st.success("لا يوجد أي تعارض: الحل اجتاز التحقق المستقل.")
    else:
        st.error(
            f"الحل يحتوي على {int(best_run['Hard Violations'])} "
            "انتهاكات صارمة (تعارضات) ولم يجتز التحقق."
        )
    with st.expander("تفاصيل التحقق"):
        st.code(texts[best_idx] or "لا توجد تفاصيل إضافية.")

    # ---- 2) time / resources / convergence speed ------------------------------------
    st.markdown("#### 2) الزمن والموارد وسرعة التقارب")
    perf = {
        "Wall time (s)": runs_df["Wall Time (s)"].mean(),
        "CPU time (s)": runs_df["CPU Time (s)"].mean(),
        "Peak memory (MB)": runs_df["Peak Memory (MB)"].max(),
        "Iter. to best": runs_df["Iter. to Best"].mean(),
        "Iter. to 95% gain": runs_df["Iter. to 95% of gain"].mean(),
        "Iter. to feasible": feasible_its.mean() if len(feasible_its) else None,
        "Budget (iter.)": runs_df["Budget (iterations)"].iloc[0],
    }
    st.dataframe(
        pd.DataFrame([perf]).round(2),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("القيم متوسط التشغيلات (الذاكرة: الأعلى). "
               "Iter. to feasible فارغة = لم يتحقق حل صالح.")

    # ---- 3) convergence curve ------------------------------------------------------
    st.markdown("#### 3) منحنى التقارب: Iteration → Best Fitness")
    curves = {}
    for run, result in zip(runs, results):
        its, fit, _ = history_series(result.history)
        curves[f"seed {run['Seed']}"] = pd.Series(fit, index=its)
    curves_df = pd.DataFrame(curves)
    curves_df.index.name = "Iteration"
    st.line_chart(curves_df)

    # ---- 4) stability ------------------------------------------------------------
    st.markdown("#### 4) استقرار النتائج عند إعادة التجربة")
    stability_df = pd.DataFrame()
    if n_runs > 1:
        stability_df = stability_summary(runs_df)
        keep = ["Best Fitness", "Hard Violations", "Wall Time (s)"]
        st.dataframe(
            stability_df[stability_df["Metric"].isin(keep)].round(3),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Std = الانحراف المعياري، CV% = Std / Mean × 100 "
                   "(الأقل = أكثر استقراراً).")
    else:
        st.info("اختر «عدد مرات التكرار» 2 أو أكثر لقياس الاستقرار.")

    # ---- 5) weekly timetable ---------------------------------------------------------
    st.markdown("#### 5) الجدول (أفضل تشغيلة): الأيام × الساعات × القاعات")
    events_df, rows = timetable_frames(problem, results[best_idx].best_schedule)

    room_choice = st.selectbox(
        "عرض القاعة",
        ["الكل"] + list(problem.room_ids),
        key="room_filter",
    )
    st.dataframe(
        weekly_grid(problem, rows, room_choice),
        use_container_width=True,
    )
    st.caption("كل خلية: «القاعة: المقرر» للأحداث الجارية في تلك الساعة "
               "(الحدث الأطول من ساعة يظهر في كل ساعاته).")

    with st.expander("قائمة الأحداث التفصيلية (حدث، أستاذ، يوم، ساعة، قاعة)"):
        st.dataframe(events_df, use_container_width=True, hide_index=True)

    # ---- one-row summary + exports -----------------------------------------------------
    summary = {
        "Algorithm": name,
        "Problem": problem_type,
        "Size": problem_size,
        "Budget": budget,
        "Runs": n_runs,
        "Feasible runs": n_valid,
        "Best Fitness (best run)": best_run["Best Fitness"],
        "Fitness mean": runs_df["Best Fitness"].mean(),
        "Fitness std": runs_df["Best Fitness"].std(ddof=0),
        "Hard Violations (best run)": int(best_run["Hard Violations"]),
        "Soft Penalty (best run)": best_run["Soft Penalty"],
        "Wall Time mean (s)": runs_df["Wall Time (s)"].mean(),
        "CPU Time mean (s)": runs_df["CPU Time (s)"].mean(),
        "Peak Memory max (MB)": runs_df["Peak Memory (MB)"].max(),
        "Iter. to Best mean": runs_df["Iter. to Best"].mean(),
        "Iter. to 95% gain mean": runs_df["Iter. to 95% of gain"].mean(),
        "Iter. to Feasible mean": feasible_its.mean() if len(feasible_its) else None,
    }

    return summary, runs_df, curves_df, stability_df, events_df


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎓 SmartAcademicScheduler")
st.markdown(
    """
    ### نظام ذكي لجدولة المحاضرات والامتحانات

    واجهة لتشغيل خوارزميات التحسين، والتحقق من الحلول،
    ومقارنة الجودة وزمن التنفيذ والموارد والتقارب والاستقرار.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("⚙️ إعدادات التجربة")

problem_type = st.sidebar.selectbox(
    "نوع المشكلة",
    list(DATASETS.keys()),
)

problem_size = st.sidebar.selectbox(
    "حجم البيانات",
    ["Small", "Medium", "Large"],
)

algorithm_label = st.sidebar.selectbox(
    "الخوارزمية",
    list(ALGORITHMS.keys()),
)

budget_label = st.sidebar.selectbox(
    "ميزانية التشغيل",
    list(BUDGETS.keys()),
)

seed = st.sidebar.number_input(
    "البذرة العشوائية الأولى (Seed)",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

repeats = st.sidebar.number_input(
    "عدد مرات التكرار (لقياس الاستقرار)",
    min_value=1,
    max_value=30,
    value=3,
    step=1,
)

measure_memory = st.sidebar.checkbox(
    "قياس استهلاك الذاكرة (أبطأ قليلاً)",
    value=True,
)

st.sidebar.caption(
    "يُشغَّل التكرار k بالبذرة seed + k. "
    "الميزانية «القياسية» هي إعدادات التقرير وقد تستغرق دقائق "
    "على الأحجام الكبيرة."
)

dataset_path = DATASETS[problem_type][problem_size]


# --------------------------------------------------
# Main interface
# --------------------------------------------------

left, right = st.columns([2, 1])

with left:
    st.subheader("إعداد المشكلة")
    st.write(f"**نوع الجدولة:** {problem_type}")
    st.write(f"**حجم البيانات:** {problem_size}")
    st.write(f"**الخوارزمية:** {algorithm_label}")
    st.write(f"**الميزانية:** {budget_label}")
    st.write(f"**مجموعة البيانات:** `{dataset_path}`")



st.divider()

run_clicked = st.button(
    "▶ تشغيل الخوارزمية",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    try:
        full_dataset_path = ROOT / dataset_path

        if not full_dataset_path.is_dir():
            st.error(
                f"لم يتم العثور على مجلد البيانات:\n"
                f"{full_dataset_path}"
            )
            st.stop()

        with st.spinner("جارٍ تحميل المشكلة..."):
            problem = get_problem(str(full_dataset_path))

        st.write(
            f"عدد الأحداث في المشكلة: **{problem.n_events}**"
        )

        algorithm_code = ALGORITHMS[algorithm_label]
        budget = BUDGETS[budget_label]

        runs, results, texts = [], [], []
        progress = st.progress(0.0, text="بدء التجارب...")

        for k in range(int(repeats)):
            progress.progress(
                k / int(repeats),
                text=f"التشغيلة {k + 1} من {int(repeats)} "
                     f"(seed = {int(seed) + k})",
            )
            row, result, text = run_once(
                algorithm_code,
                problem,
                int(seed) + k,
                budget,
                measure_memory,
            )
            runs.append(row)
            results.append(result)
            texts.append(text)

        progress.progress(1.0, text="اكتملت التجارب")

        st.divider()

        summary, runs_df, curves_df, stability_df, events_df = show_results(
            algorithm_label,
            algorithm_label,
            problem_type,
            problem_size,
            budget_label,
            runs,
            results,
            texts,
            problem,
        )

        st.session_state["last_result"] = summary

        st.divider()
        summary_df = pd.DataFrame([summary])
        tag = f"{algorithm_code}_{problem_type}_{problem_size}".replace(" ", "")

        with st.expander("📊 ملخص من سطر واحد + تصدير (للمقارنة في التقرير)"):
            st.dataframe(
                summary_df.round(3),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(runs_df.round(3), use_container_width=True, hide_index=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "⬇️ الملخص",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"summary_{tag}.csv",
            mime="text/csv",
        )
        d2.download_button(
            "⬇️ كل التشغيلات",
            data=runs_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"runs_{tag}.csv",
            mime="text/csv",
        )
        d3.download_button(
            "⬇️ منحنيات التقارب",
            data=curves_df.to_csv().encode("utf-8-sig"),
            file_name=f"convergence_{tag}.csv",
            mime="text/csv",
        )
        d4.download_button(
            "⬇️ الجدول",
            data=events_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"schedule_{tag}.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error("حدث خطأ أثناء تشغيل التجربة.")
        st.exception(exc)

else:
    st.info(
        "اختر نوع الجدولة وحجم البيانات والخوارزمية من القائمة "
        "الجانبية، ثم اضغط «تشغيل الخوارزمية»."
    )

    st.markdown("#### ما الذي ستعرضه الواجهة؟")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("جودة الحل", "Fitness / Violations")
    c2.metric("الزمن والموارد", "Wall / CPU / Memory")
    c3.metric("التقارب", "Iteration → Best Fitness")
    c4.metric("الاستقرار", "Mean / Std / CV%")