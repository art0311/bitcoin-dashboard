"""Read-only BTC/ETH forecast history and model performance.

Writes are exclusively performed by trusted GitHub Actions jobs. Supabase
credentials stay in Streamlit Secrets. No model training occurs on this page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from forecast_storage import StorageError, list_forecasts

st.set_page_config(page_title="AI Forecast History", layout="wide")
st.title("📚 AI Forecast History")
st.caption("Recorded live predictions and their subsequent outcomes. Dates use UTC; models are experimental, not validated trading signals.")

PAGE_SIZE = 500
MAX_ROWS = 20000
MODEL_ORDER = {
    "Ridge Regression": 0,
    "Random Forest": 1,
    "7-day momentum benchmark": 2,
    "No-change baseline": 3,
}
MAX_ISSUE_DELAY_HOURS = 6.0  # Operational cutoff after origin UTC close
SORT_OPTIONS = [
    "Newest issue date (BTC → ETH, 1d → 60d)",
    "Oldest issue date",
    "Forecast horizon (1d → 60d)",
    "Target date (soonest first)",
    "Highest prediction error (evaluated first)",
]

@st.cache_data(ttl=300, show_spinner=False)
def load_history() -> tuple[list[dict], bool]:
    records = []
    for offset in range(0, MAX_ROWS, PAGE_SIZE):
        batch = list_forecasts(st.secrets, limit=PAGE_SIZE, offset=offset)
        records.extend(batch)
        if len(batch) < PAGE_SIZE:
            return records, False
    # An entire additional page might exist. Don't claim completeness at cap.
    return records, True


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in ("forecast_date", "target_date", "evaluated_at"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in ("starting_price", "predicted_price", "actual_price", "horizon_days"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df["actual_price"].notna().map({True: "Evaluated", False: "Pending"})
    valid = (df["starting_price"] > 0) & (df["actual_price"] > 0)
    df["actual_move_pct"] = (df["actual_price"] / df["starting_price"] - 1) * 100
    df["predicted_move_pct"] = (df["predicted_price"] / df["starting_price"] - 1) * 100
    df["price_error_pct"] = ((df["predicted_price"] - df["actual_price"]).abs() / df["actual_price"]) * 100
    df.loc[~valid, "price_error_pct"] = float("nan")
    df["direction_correct"] = pd.NA
    directional = valid & df["model"].ne("No-change baseline") & df["predicted_move_pct"].ne(0)
    df.loc[directional, "direction_correct"] = (
        (df.loc[directional, "predicted_move_pct"] > 0)
        == (df.loc[directional, "actual_move_pct"] > 0)
    ).astype(bool)
    df["model_rank"] = df["model"].map(MODEL_ORDER).fillna(99).astype(int)
    df["asset_rank"] = df["asset"].map({"BTC": 0, "ETH": 1}).fillna(99).astype(int)
    df["issue_day"] = df["forecast_date"].dt.floor("D")
    # Target is the UTC END of the h-th candle after the last completed close.
    origin_close = df["target_date"] - pd.to_timedelta(df["horizon_days"], unit="D")
    df["issue_delay_hours"] = (df["forecast_date"] - origin_close).dt.total_seconds() / 3600
    df["lead_hours"] = (df["target_date"] - df["forecast_date"]).dt.total_seconds() / 3600
    coherent = (df["forecast_date"].notna() & df["target_date"].notna()
        & df["horizon_days"].isin([1, 7, 30, 60])
        & df["lead_hours"].gt(0) & df["issue_delay_hours"].ge(0))
    df["timing_eligible"] = coherent & df["issue_delay_hours"].le(MAX_ISSUE_DELAY_HOURS)
    df["timing"] = "Late / check"
    df.loc[df["timing_eligible"], "timing"] = "On time"
    return df


def sort_forecasts(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    # Using daily UTC grouping means different models issued seconds apart stay together.
    if choice == SORT_OPTIONS[1]:
        keys, asc = ["issue_day", "asset_rank", "horizon_days", "model_rank", "forecast_date"], [True, True, True, True, True]
    elif choice == SORT_OPTIONS[2]:
        keys, asc = ["horizon_days", "issue_day", "asset_rank", "model_rank"], [True, False, True, True]
    elif choice == SORT_OPTIONS[3]:
        keys, asc = ["target_date", "asset_rank", "horizon_days", "model_rank"], [True, True, True, True]
    elif choice == SORT_OPTIONS[4]:
        # NaNs (pending forecasts) are always last, not treated as zero error.
        keys, asc = ["price_error_pct", "issue_day", "asset_rank", "model_rank"], [False, False, True, True]
    else:
        keys, asc = ["issue_day", "asset_rank", "horizon_days", "model_rank", "forecast_date"], [False, True, True, True, False]
    return df.sort_values(keys, ascending=asc, na_position="last", kind="stable")

try:
    records, capped = load_history()
except StorageError:
    st.error("Couldn't retrieve forecasts from Supabase. Check app configuration and retry.")
    st.stop()
except Exception:
    st.error("Forecast history temporarily unavailable. Check Streamlit app logs.")
    st.stop()

if not records:
    st.info("No saved forecasts yet. The scheduled job will add them here.")
    st.stop()

frame = to_dataframe(records)
if capped:
    st.warning(f"Showing at most {MAX_ROWS:,} recent records; counts and performance may exclude older forecasts.")

st.subheader("Choose what to inspect")
c1, c2, c3 = st.columns(3)
with c1:
    asset = st.selectbox("Asset", ["All", "BTC", "ETH"])
with c2:
    horizon = st.selectbox("Forecast horizon (days)", ["All", "1", "7", "30", "60"])
with c3:
    status = st.selectbox("Forecast status", ["All", "Pending", "Evaluated"])
filtered = frame.copy()
if asset != "All":
    filtered = filtered.loc[filtered["asset"] == asset]
if horizon != "All":
    filtered = filtered.loc[filtered["horizon_days"] == int(horizon)]
if status != "All":
    filtered = filtered.loc[filtered["status"] == status]

completed = filtered.loc[filtered["actual_price"].notna()].copy()
comparable = completed.loc[completed["timing_eligible"]].copy()
scored = comparable.loc[comparable["direction_correct"].notna()]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stored forecasts", f"{len(filtered):,}")
m2.metric("Awaiting outcomes", f"{int(filtered['actual_price'].isna().sum()):,}")
m3.metric("Evaluated", f"{len(completed):,}")
m4.metric("Correct direction (on-time)", "Not available" if scored.empty else f"{scored['direction_correct'].astype(float).mean()*100:.1f}%")
late_count = int((~completed["timing_eligible"]).sum())
if late_count:
    st.warning(f"{late_count} evaluated forecast(s) were issued more than {MAX_ISSUE_DELAY_HOURS:g} hours after the origin UTC close, or have inconsistent timing. They remain visible but are excluded from the main accuracy statistics.")
st.caption("Horizon refers to daily candles after the LAST completed UTC close, not hours since issuance. Lead-to-target shows actual time remaining. 'On time' means issued within six hours after the originating candle closed. Related model outcomes are not independent tests.")

st.subheader("Recorded forecasts")
sort_by = st.selectbox("Sort by", SORT_OPTIONS, index=0)
if filtered.empty:
    st.info("No records match these filters.")
else:
    display = sort_forecasts(filtered, sort_by).copy()
    display["Issued (UTC)"] = display["forecast_date"].dt.strftime("%Y-%m-%d %H:%M")
    display["Target (UTC)"] = display["target_date"].dt.strftime("%Y-%m-%d")
    display["Horizon"] = display["horizon_days"].map(lambda v: f"{int(v)}d" if pd.notna(v) else "—")
    display["Lead (hours)"] = display["lead_hours"]
    display = display.rename(columns={
        "asset": "Asset", "model": "Model", "model_version": "Version",
        "starting_price": "Starting price", "predicted_price": "Predicted price",
        "actual_price": "Actual price", "price_error_pct": "Price error (%)", "status": "Status",
    })
    cols = ["Issued (UTC)", "Asset", "Horizon", "Model", "Starting price", "Predicted price", "Target (UTC)", "Actual price", "Price error (%)", "Status", "Lead (hours)", "timing", "Version"]
    st.dataframe(display[cols], hide_index=True, use_container_width=True,
        column_config={
            "Starting price": st.column_config.NumberColumn(format="$%.2f"),
            "Predicted price": st.column_config.NumberColumn(format="$%.2f"),
            "Actual price": st.column_config.NumberColumn(format="$%.2f"),
            "Price error (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Lead (hours)": st.column_config.NumberColumn(format="%.1f"),
        })
    st.download_button("Download filtered history (CSV)",
        data=display[cols].to_csv(index=False).encode("utf-8"),
        file_name="forecast_history.csv", mime="text/csv",
        help="Optional export; automated history does not need manual downloads.")

st.subheader("Performance by model (on-time only)")
if comparable.empty:
    st.info("No on-time evaluated forecasts yet. Late-issued records remain above but do not count toward comparable model accuracy. The first batches were issued well after the origin UTC close.")
else:
    keys = ["asset", "horizon_days", "model"]
    stats = comparable.groupby(keys).agg(
        tests=("id", "count"), mean_price_error=("price_error_pct", "mean")
    ).reset_index()
    directions = (comparable.loc[comparable["direction_correct"].notna()]
        .assign(direction_value=lambda d: d["direction_correct"].astype(float))
        .groupby(keys)["direction_value"].mean().mul(100).rename("direction_pct").reset_index())
    stats = stats.merge(directions, on=keys, how="left")
    stats["model_rank"] = stats["model"].map(MODEL_ORDER).fillna(99)
    stats = stats.sort_values(["asset", "horizon_days", "model_rank"])
    stats = stats.rename(columns={"asset": "Asset", "horizon_days": "Days", "model": "Model",
        "tests": "Evaluated forecasts", "mean_price_error": "Mean price error (%)",
        "direction_pct": "Direction correct (%)"})
    st.dataframe(stats[["Asset", "Days", "Model", "Evaluated forecasts", "Mean price error (%)", "Direction correct (%)"]],
        hide_index=True, use_container_width=True,
        column_config={
            "Mean price error (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Direction correct (%)": st.column_config.NumberColumn(format="%.1f%%"),
        })
    st.caption("Only on-time records count. Lower mean price error is better. Compare models for the same asset and horizon; very small samples are unreliable.")
st.divider()
st.caption("Read-only: GitHub Actions writes forecast outcomes; this page never modifies the database. Historical performance does not guarantee future accuracy.")
