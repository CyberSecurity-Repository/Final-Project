"""The four Streamlit dashboard sections + the pipeline-run control.

Each ``render_*`` function is a thin presentation layer over
:mod:`retail_clickstream_ai.dashboard.cache` (cached artifact reads) and
:mod:`retail_clickstream_ai.dashboard.data` (validation/prediction helpers).
No section ever starts the Flow or calls OpenAI on its own — only
:func:`render_pipeline_control`'s button, after explicit confirmation, does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from retail_clickstream_ai import paths
from retail_clickstream_ai.config import load_settings
from retail_clickstream_ai.dashboard import cache
from retail_clickstream_ai.dashboard import data as svc
from retail_clickstream_ai.dashboard import pipeline_control as run_ctl
from retail_clickstream_ai.flow import DEFAULT_INPUT
from retail_clickstream_ai.validation.errors import ContractValidationError

_CAVEAT = (
    "This is **2008** data from a **single** online clothing shop. It demonstrates a "
    "workflow end to end — it is **not** evidence about present-day shoppers, other "
    "retailers, or any real-time business decision."
)


def safe_section(name: str, render: Callable[[], None]) -> None:
    """Run one section's renderer, turning known artifact problems into a concise message."""
    try:
        render()
    except svc.ArtifactMissingError as exc:
        st.warning(f"**{name} isn't available yet.** {exc}")
    except svc.ModelIntegrityError as exc:
        st.error(f"**{name} is blocked.** {exc}")
    except (KeyError, ValueError) as exc:
        st.error(f"**{name} couldn't be rendered** — an artifact looks incompatible: {exc}")
    except Exception as exc:  # noqa: BLE001 - last-resort concise message, never a stack trace
        st.error(f"**{name} hit an unexpected error:** {type(exc).__name__}: {exc}")


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _md(text: str) -> str:
    """Escape literal `$` in artifact-sourced markdown before `st.markdown`.

    Streamlit's markdown renders `$...$` as inline LaTeX; the reports contain
    literal dollar amounts (e.g. "US$18 to US$82"), which would otherwise be
    mangled into a math expression instead of displayed as text.
    """
    return text.replace("$", "\\$")


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview() -> None:
    st.header("Overview")
    st.info(_CAVEAT, icon="⚠️")

    st.subheader("Business question")
    st.markdown(
        "Given the clicks a shopper has made **so far in a session**, predict the "
        "**main product category** (trousers / skirts / blouses / sale) they are "
        "most likely to view **next**."
    )

    contract = cache.dataset_contract()
    ds = contract["dataset"]
    raw = ds["raw_file"]
    cleaned = contract["cleaned_data"]
    target = contract["target"]

    st.subheader("Dataset")
    cols = st.columns(4)
    cols[0].metric("Clicks", f"{raw['row_count']:,}")
    cols[1].metric("Sessions", f"{contract['session_key']['observed_session_count']:,}")
    cols[2].metric("Labelled examples", f"{target['rows_with_label']:,}")
    cols[3].metric("Time period", ds["documented_time_period"].split(" (")[0])
    st.caption(
        f"**Source:** {ds['name']} — {ds['publisher']} · **License:** {ds['license']} · "
        f"[UCI dataset 553]({ds['source_urls']['uci']})"
    )
    st.caption(
        f"Raw file `{raw['name']}` (SHA-256 `{raw['sha256'][:16]}…`, {raw['row_count']:,} rows) "
        f"→ cleaned `{cleaned['file']}` (SHA-256 `{cleaned['sha256'][:16]}…`, "
        f"{cleaned['row_count']:,} rows)."
    )

    st.subheader("Latest Flow status")
    _render_flow_status()

    with st.expander("⚙️ Run full pipeline (advanced)"):
        render_pipeline_control()


def _render_flow_status() -> None:
    latest = svc.find_latest_run()
    if latest.kind == "none":
        st.info(
            "No Flow run has been recorded under `artifacts/runs/` yet. The sections "
            "below read the **committed canonical artifacts** directly."
        )
        _render_canonical_artifact_table()
        return
    if latest.kind == "corrupt":
        st.warning(
            f"The most recent run record (`{latest.path}`, last modified {latest.modified_at}) "
            "could not be parsed as JSON. Showing the committed canonical artifacts instead."
        )
        _render_canonical_artifact_table()
        return

    payload = latest.payload or {}
    if latest.kind == "success":
        st.success(f"Run **{latest.run_id}** — SUCCESS (recorded {latest.modified_at})")
        selection = payload.get("selection", {})
        cols = st.columns(3)
        cols[0].metric("Selected model", str(selection.get("selected_candidate_id", "—")))
        metric_name = str(selection.get("primary_metric", "macro_f1"))
        metric_value = selection.get("primary_metric_value")
        cols[1].metric(
            f"Test {metric_name}", f"{metric_value:.4f}" if metric_value is not None else "—"
        )
        cols[2].metric("Total duration", f"{payload.get('total_duration_seconds', 0):.1f}s")
        stages = pd.DataFrame(payload.get("stages", []))
        if not stages.empty:
            st.dataframe(
                stages[["stage", "status", "duration_seconds"]],
                hide_index=True,
                width="stretch",
            )
    else:
        st.error(f"Run **{latest.run_id}** — FAILED at `{payload.get('failed_step', '?')}`")
        st.markdown(f"**Problem:** {payload.get('cause', '—')}")
        st.markdown(f"**Remediation:** {payload.get('remediation', '—')}")
        st.caption(
            f"Downstream blocked: {payload.get('downstream_blocked', True)} · "
            f"recorded {latest.modified_at}"
        )
        _render_canonical_artifact_table()


def _render_canonical_artifact_table() -> None:
    rows = []
    for status in [*cache.analyst_artifact_statuses(), *cache.scientist_artifact_statuses()]:
        rows.append(
            {
                "artifact": status.name,
                "present": "✅" if status.exists else "❌",
                "modified": status.modified_at or "—",
                "sha256": (status.sha256[:16] + "…") if status.sha256 else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def render_eda() -> None:
    st.header("Exploratory Data Analysis")
    st.caption(
        "Every figure and number below is read verbatim from the committed Analyst artifacts."
    )

    insights = cache.insights_text()
    summary = svc.extract_markdown_section(insights, "Key findings")
    if summary:
        st.markdown(_md(summary))
    with st.expander("Full insights report (`insights.md`)"):
        st.markdown(_md(insights))

    manifest = {f["figure_id"]: f for f in cache.eda_figure_manifest()["figures"]}

    st.subheader("Main-category frequency")
    cat = cache.eda_table("main_category_frequency").set_index("label")
    st.bar_chart(cat["share_pct"])
    st.caption(manifest.get("fig_category_frequency", {}).get("caption", ""))

    st.subheader("Next-category transition probabilities (%)")
    trans = cache.eda_table("transition_matrix_pct")
    labels = {
        row.code: row.label for row in cache.eda_table("main_category_frequency").itertuples()
    }
    trans[["cur", "nxt"]] = trans["pair"].str.split("->", expand=True).astype(int)
    pivot = trans.pivot(index="cur", columns="nxt", values="probability_pct")
    pivot.index = [labels[i] for i in pivot.index]
    pivot.columns = [labels[i] for i in pivot.columns]
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}"), width="stretch"
    )
    st.caption(manifest.get("fig_transitions", {}).get("caption", ""))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Session depth")
        buckets = cache.eda_table("clicks_per_session_buckets").set_index("bucket")
        st.bar_chart(buckets["sessions"])
        st.caption(manifest.get("fig_session_depth", {}).get("caption", ""))
    with col_b:
        st.subheader("Top countries")
        countries = cache.eda_table("top_countries").head(8).set_index("label")
        st.bar_chart(countries["share_pct"])
        st.caption(manifest.get("fig_top_countries", {}).get("caption", ""))

    st.subheader("Clicks per month")
    months = cache.eda_table("month_coverage").set_index("month")
    st.line_chart(months["clicks"])
    st.caption(manifest.get("fig_month_activity", {}).get("caption", ""))

    with st.expander("Open the full EDA report (`eda_report.html`)"):
        # Safe: this is our own deterministically-rendered report, never
        # user/LLM-supplied HTML — see `st.iframe`'s untrusted-HTML warning.
        st.iframe(cache.eda_report_html(), height=600)


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #
def render_evaluation() -> None:
    st.header("Model evaluation")
    metrics = cache.scientist_metrics()

    st.subheader("Baseline vs. candidate comparison (validation)")
    candidates = pd.DataFrame(metrics["candidates"]).T
    candidates.index.name = "candidate"
    st.dataframe(candidates[["macro_f1", "accuracy", "weighted_f1", "log_loss"]], width="stretch")
    st.bar_chart(candidates["macro_f1"])

    winner = metrics["selected_candidate_id"]
    st.success(
        f"**Selected model:** `{winner}` ({metrics['selected_model']}) — "
        f"{metrics['selection_rule']}"
    )

    st.subheader("Held-out test performance (August, evaluated once)")
    test = metrics["test"]
    cols = st.columns(4)
    cols[0].metric("Macro F1 (primary)", f"{test['macro_f1']:.4f}")
    cols[1].metric("Accuracy", _pct(test["accuracy"]))
    cols[2].metric("Weighted F1", f"{test['weighted_f1']:.4f}")
    cols[3].metric("Log loss", f"{test['log_loss']:.4f}" if test["log_loss"] is not None else "—")

    labels = metrics["class_labels"]
    per_class = pd.DataFrame(test["per_class"]).T
    per_class.index = [f"{c}={labels[c]}" for c in per_class.index]
    st.dataframe(per_class[["precision", "recall", "f1", "support"]], width="stretch")

    weak = svc.weakest_test_class(metrics)
    st.warning(
        f"**Weakest class:** {weak}={labels[weak]} (test F1 "
        f"{test['per_class'][weak]['f1']:.3f}) — the hardest category to predict as the next click."
    )

    st.subheader("Confusion matrices")
    fig_cols = st.columns(2)
    fig_dir = paths.scientist_artifacts() / "figures"
    with fig_cols[0]:
        st.image(str(fig_dir / "confusion_matrix_validation.png"), caption="Validation (July)")
    with fig_cols[1]:
        st.image(str(fig_dir / "confusion_matrix_test.png"), caption="Test (August, held out)")

    with st.expander("Full evaluation report (`evaluation_report.md`)"):
        st.markdown(_md(cache.evaluation_report_text()))
    with st.expander("Model card (`model_card.md`) — limitations & security notes"):
        st.markdown(_md(cache.model_card_text()))


# --------------------------------------------------------------------------- #
# Predict next category
# --------------------------------------------------------------------------- #
def render_predict() -> None:
    st.header("Predict next category")
    st.caption(
        "Fields below come directly from the trained pipeline's own feature contract "
        "(`model_metadata.json`) — the same 12 predictors it was trained on."
    )

    status = cache.scientist_artifact_statuses()
    model_status = next(s for s in status if s.name == "model.joblib")
    if not model_status.exists:
        st.warning("`model.joblib` is not available yet — inference is disabled.")
        return

    ok, observed, expected = svc.verify_model_artifact()
    if not ok:
        st.error(
            "**Model artifact hash mismatch — refusing to load an untrusted model.**\n\n"
            f"- On disk: `{observed}`\n- Expected (`model_metadata.json`): `{expected}`\n\n"
            "Re-produce `model.joblib` from a trusted Scientist run before predicting."
        )
        return

    specs = cache.prediction_field_specs()
    current_specs = [s for s in specs if s.group == "current"]
    history_specs = [s for s in specs if s.group == "history"]

    with st.form("predict_form"):
        st.markdown("**Current click**")
        values: dict[str, int] = {}
        cols = st.columns(2)
        for i, spec in enumerate(current_specs):
            values[spec.name] = _render_field(cols[i % 2], spec)

        st.markdown("**Session history so far** (optional context)")
        cols2 = st.columns(2)
        for i, spec in enumerate(history_specs):
            values[spec.name] = _render_field(cols2[i % 2], spec)

        submitted = st.form_submit_button("Predict next category", type="primary")

    if not submitted:
        return

    try:
        model = cache.verified_model()
        result = svc.predict_next_category(values, model=model)
    except ContractValidationError as exc:
        st.error("**Invalid input — prediction was not run:**\n\n" + exc.report.render())
        return
    except svc.ModelIntegrityError as exc:
        st.error(f"**Refused to load the model:** {exc}")
        return

    st.success(
        f"**Predicted next category:** {result.predicted_label} (code {result.predicted_class})"
    )
    proba_df = pd.DataFrame(
        [{"category": label, "probability": p} for _, label, p in result.probabilities]
    ).set_index("category")
    st.bar_chart(proba_df["probability"])
    st.dataframe(proba_df.style.format("{:.1%}"), width="stretch")
    st.caption(
        "Plain-language caveat: this is a workflow demonstration trained on **2008** data from "
        "one shop. Probabilities describe a *descriptive* browsing pattern, not a guarantee, "
        "and should never drive a real merchandising or personalization decision."
    )


def _render_field(container: Any, spec: svc.FieldSpec) -> int:
    if spec.kind == "categorical":
        options = spec.options or ()
        labels = dict(options)
        codes = [code for code, _ in options]
        default_index = codes.index(spec.default) if spec.default in codes else 0
        return int(
            container.selectbox(
                spec.label,
                options=codes,
                index=default_index,
                format_func=lambda c: labels.get(c, str(c)),
                help=spec.help,
            )
        )
    return int(
        container.number_input(
            spec.label,
            min_value=spec.min_value,
            max_value=spec.max_value,
            value=spec.default,
            step=1,
            help=spec.help,
        )
    )


# --------------------------------------------------------------------------- #
# Pipeline run control
# --------------------------------------------------------------------------- #
def render_pipeline_control() -> None:
    st.caption(
        "Runs the real CrewAI Flow (`prepare → Analyst crew → gate → Scientist crew → "
        "gate → manifest`) against the committed raw dataset. This **overwrites** the "
        "canonical `artifacts/analyst/` and `artifacts/scientist/` files with a fresh run."
    )

    settings = load_settings()
    engine = st.radio(
        "Engine",
        options=("deterministic", "crew"),
        horizontal=True,
        format_func=lambda e: (
            "Deterministic (no LLM, free)" if e == "deterministic" else "Crew (real OpenAI calls)"
        ),
        key="pipeline_engine",
    )
    is_paid = engine == "crew"
    key_missing = is_paid and not settings.has_openai_config
    if key_missing:
        st.warning(
            "`OPENAI_API_KEY` / `OPENAI_MODEL_NAME` are not configured, so the **crew** "
            "engine is disabled. Set them in `.env` to run the real LLM crews, or switch "
            "to the deterministic engine (no OpenAI calls)."
        )

    confirm_label = (
        "I understand this starts a real run and will make **paid OpenAI calls**."
        if is_paid
        else "I understand this starts a real (free, local) pipeline run."
    )
    confirmed = st.checkbox(confirm_label, key="pipeline_confirm")

    active = run_ctl.is_active()
    disabled = active or key_missing or not confirmed
    if st.button("Start run", disabled=disabled, key="pipeline_start"):
        run_id = run_ctl.start_run(input_path=DEFAULT_INPUT, engine=engine)
        if run_id is None:
            st.warning("A run is already in progress — please wait for it to finish.")
        else:
            st.session_state["pipeline_last_run_id"] = run_id
            st.rerun()

    _render_run_status()


def _render_run_status() -> None:
    snap = run_ctl.snapshot()
    run_id = snap.get("run_id") or st.session_state.get("pipeline_last_run_id")
    if not run_id:
        return

    if snap.get("active"):
        st.info(f"Run **{run_id}** is in progress (engine={snap.get('engine')})…")
        log_lines = run_ctl.tail_log(run_id)
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        if st.button("Refresh status", key="pipeline_refresh"):
            st.rerun()
        return

    result = snap.get("result")
    if result == "success":
        cache.clear_all()
        st.success(
            f"Run **{run_id}** finished successfully. Manifest: `{snap.get('manifest_path')}`"
        )
        st.caption("Artifact caches were cleared — the sections above now reflect this run.")
    elif result == "failed":
        st.error(f"Run **{run_id}** failed at `{snap.get('failed_stage') or '?'}`.")
        if snap.get("failure_cause"):
            st.markdown(f"**Problem:** {snap['failure_cause']}")
        if snap.get("failure_remediation"):
            st.markdown(f"**Remediation:** {snap['failure_remediation']}")
        if snap.get("error"):
            st.caption(f"Unexpected error: {snap['error']}")
