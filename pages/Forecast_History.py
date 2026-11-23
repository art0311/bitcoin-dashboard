"""Read-only forecast history for the BTC/ETH Streamlit development dashboard.

All writes happen in trusted GitHub Actions jobs, never in this public page.
Keep SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from forecast_storage import StorageError, list_forecasts

st.set_page_config(page_title="AI Forecast History", layout="wide")
st.title("📚 AI Forecast History")
st.caption(
    "Real predictions recorded before their outcomes were known. "
    "All dates use UTC. Models are experimental, not validated trading signals."
)


@st.cache_data(ttl=300, show_spinner=False)
def load_history() -> list[dict]:
    # The existing storage API supports at most 500 recent records per request.
    # Do not imply this is a complete lifetime history once that limit is reached.
    return list_forecasts(st.secrets, limit=500)


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for col in ("forecast_date", "target_date", "evaluated_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    for col in ("starting_price", "predicted_price", "actual_price"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    frame["status"] = frame["actual_price"].notna().map({True: "Evaluated", False: "Pending"})
    valid = (frame["starting_price"] > 0) & (frame["actual_price"] > 0)
    frame["actual_move_pct"] = ((frame["actual_price"] / frame["starting_price"]) - 1) * 100
    frame["predicted_move_pct"] = ((frame["predicted_price"] / frame["starting_price"]) - 1) * 100
    frame["price_error_pct"] = ((frame["predicted_price"] - frame["actual_price"]).abs() / frame["actual_price"]) * 100
    frame.loc[~valid, "price_error_pct"] = float("nan")
    frame["direction_correct"] = pd.NA
    directional = valid & frame["model"].ne("No-change baseline")
    # Zero-valued predicted moves represent no directional call.
    directional &= frame["predicted_move_pct"].ne(0)
    frame.loc[directional, "direction_correct"] = (
        (frame.loc[directional, "predicted_move_pct"] > 0)
        == (frame.loc[directional, "actual_move_pct"] > 0)
    ).astype(bool)
    return frame


try:
    rows = load_history()
except StorageError:
    st.error("Couldn't retrieve forecasts from Supabase. Check the app configuration or retry later.")
    st.stop()
except Exception:
    st.error("Forecast history is temporarily unavailable. Please check the Streamlit app logs.")
    st.stop()

if not rows:
    st.info("No saved forecasts yet. The scheduled forecasting job will add them here.")
    st.stop()

frame = to_dataframe(rows)
if len(rows) == 500:
    st.warning(
        "Showing only the 500 most recently recorded forecasts. "
        "Historical totals and accuracy on this page exclude older records. "
        "We'll add paginated database queries before the history grows beyond this limit."
    )

st.subheader("Choose what to inspect")
f1, f2, f3 = st.columns(3)
with f1:
    asset = st.selectbox("Asset", ["All", "BTC", "ETH"])
with f2:
    horizons = ["All", "1", "7", "30", "60"]
    horizon = st.selectbox("Forecast horizon (days)", horizons)
with f3:
    status = st.selectbox("Forecast status", ["All", "Pending", "Evaluated"])

filtered = frame.copy()
if asset != "All":
    filtered = filtered.loc[filtered["asset"] == asset]
if horizon != "All":
    filtered = filtered.loc[filtered["horizon_days"] == int(horizon)]
if status != "All":
    filtered = filtered.loc[filtered["status"] == status]

completed = filtered.loc[filtered["actual_price"].notna()].copy()
model_completed = completed.loc[completed["model"].ne("No-change baseline")]
valid_direction = model_completed.loc[model_completed["direction_correct"].notna()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stored forecasts", f"{len(filtered):,}")
m2.metric("Awaiting outcomes", f"{int(filtered['actual_price'].isna().sum()):,}")
m3.metric("Evaluated", f"{len(completed):,}")
if valid_direction.empty:
    m4.metric("Correct direction", "Not available")
else:
    m4.metric("Correct direction", f"{100 * valid_direction['direction_correct'].astype(float).mean():.1f}%")
st.caption(
    "One forecast row represents one model for one asset and horizon. "
    "The directional metric combines eligible model predictions from the current filters; "
    "it excludes the no-change baseline. Small samples are unreliable."
)

st.subheader("Recorded forecasts")
if filtered.empty:
    st.info("No records match the selected filters.")
else:
    display = filtered.sort_values("forecast_date", ascending=False).copy()
    display["Issued (UTC)"] = display["forecast_date"].dt.strftime("%Y-%m-%d %H:%M")
    display["Target (UTC)"] = display["target_date"].dt.strftime("%Y-%m-%d")
    display["Evaluated at (UTC)"] = display["evaluated_at"].dt.strftime("%Y-%m-%d %H:%M")
    display["Horizon"] = display["horizon_days"].map(lambda v: f"{int(v)}d" if pd.notna(v) else "—")
    display = display.rename(columns={
        "asset": "Asset", "model": "Model", "model_version": "Version",
        "starting_price": "Starting price", "predicted_price": "Predicted price",
        "actual_price": "Actual price", "price_error_pct": "Price error (%)", "status": "Status",
    })
    shown_cols = ["Issued (UTC)", "Asset", "Model", "Horizon", "Starting price", "Predicted price", "Target (UTC)", "Actual price", "Price error (%)", "Status", "Version"]
    st.dataframe(
        display[shown_cols], hide_index=True, use_container_width=True,
        column_config={
            "Starting price": st.column_config.NumberColumn(format="$%.2f"),
            "Predicted price": st.column_config.NumberColumn(format="$%.2f"),
            "Actual price": st.column_config.NumberColumn(format="$%.2f"),
            "Price error (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.download_button(
        "Download filtered history (CSV)",
        data=display[shown_cols].to_csv(index=False).encode("utf-8"),
        file_name="forecast_history.csv", mime="text/csv",
        help="Optional export; daily forecasting does not require manual downloads.",
    )

st.subheader("Performance by model")
if completed.empty:
    st.info(
        "No evaluated forecasts in this selection yet. Once a target candle has closed "
        "and the evaluator runs successfully, accuracy will appear here."
    )
else:
    grouped = completed.groupby(["asset", "horizon_days", "model"], dropna=False)
    stats = grouped.agg(tests=("id", "count"), mean_price_error=("price_error_pct", "mean")).reset_index()
    direction = (
        completed.loc[completed["direction_correct"].notna()]
        .assign(direction_value=lambda d: d["direction_correct"].astype(float))
        .groupby(["asset", "horizon_days", "model"], dropna=False)["direction_value"]
        .mean().mul(100).rename("direction_pct").reset_index()
    )
    stats = stats.merge(direction, how="left", on=["asset", "horizon_days", "model"])
    stats = stats.rename(columns={
        "asset": "Asset", "horizon_days": "Days", "model": "Model",
        "tests": "Evaluated forecasts", "mean_price_error": "Mean price error (%)",
        "direction_pct": "Direction correct (%)",
    })
    st.dataframe(
        stats.sort_values(["Asset", "Days", "Model"]),
        hide_index=True, use_container_width=True,
        column_config={
            "Mean price error (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Direction correct (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        "Lower mean price error is better. These are actual recorded forecasts, "
        "not historical backtests. Compare models within the same asset and horizon."
    )

st.divider()
st.caption(
    "Read-only page. Forecasts are saved and evaluated by GitHub Actions; "
    "this page never inserts or edits database records. "
    "Forecast accuracy is not a guarantee of future results."
)
