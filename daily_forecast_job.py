"""Daily BTC/ETH Phase 3 forecasts. Dry-run by default; no Streamlit dependency.

For manual/trusted backend execution only (never import/run from public Streamlit UI).
Data: Yahoo Finance fully completed UTC daily candles. Predictions are *experimental*.
Usage: python daily_forecast_job.py [--asset BTC|ETH|both] [--save]

Set SUPABASE_URL and SUPABASE_KEY in environment when using --save.
The database's forecast_date records the actual execution time, not a backdated
candle timestamp. target_date is the end of the forecast candle h days ahead.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import math
import sys

import pandas as pd
import yfinance as yf

from forecast_backtest import FEATURES, HORIZONS, _models, feature_frame
from forecast_storage import StorageError, list_forecasts, save_forecast

SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
MODEL_VERSION = "phase3-daily-v1"
MIN_TRAIN = 180


def completed_daily_history(symbol: str, today_utc: date | None = None) -> pd.DataFrame:
    """Fetch history; strictly exclude the current/incomplete UTC candle."""
    if symbol not in SYMBOLS:
        raise ValueError("Unsupported asset")
    today_utc = today_utc or datetime.now(timezone.utc).date()
    # yfinance end is exclusive; always exclude today's candle.
    raw = yf.download(SYMBOLS[symbol], period="5y", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No historical data received for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["Close", "High", "Low", "Volume"]
    if any(column not in raw for column in required):
        raise RuntimeError(f"Required OHLCV columns missing for {symbol}")
    raw = raw.loc[:, required].copy()
    times = pd.to_datetime(raw.index, utc=True)
    raw.index = times.normalize()
    raw = raw.loc[raw.index < pd.Timestamp(today_utc, tz="UTC")]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    # Reject data that hasn't published yesterday's completed candle yet.
    expected = today_utc - timedelta(days=1)
    if raw.empty or raw.index[-1].date() != expected:
        raise RuntimeError(f"{symbol} does not have yesterday's completed UTC candle ({expected}); retry later")
    # Confirm no gaps in recent days; a horizon in *candles* must mean calendar days.
    recent = raw.index[-min(len(raw), 90):]
    if len(recent) > 1 and any((recent[1:] - recent[:-1]) != pd.Timedelta(days=1)):
        raise RuntimeError(f"{symbol} has missing daily candles; retry after data recovers")
    return raw


def predict_all(raw: pd.DataFrame, asset: str, issued_at: datetime) -> list[dict]:
    """Fit the Phase 3 models once per horizon using labels known at issuance."""
    feat = feature_frame(raw)
    if feat.empty:
        raise RuntimeError(f"Not enough usable data for {asset}")
    idx = len(feat) - 1
    latest = feat.iloc[[idx]].dropna(subset=list(FEATURES))
    if latest.empty:
        raise RuntimeError(f"Latest {asset} features contain missing values")
    origin = feat.index[-1].date()
    # Never backdate a forecast; it must follow the fully completed candle.
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("Issuance timestamp must be timezone-aware")
    if issued_at.astimezone(timezone.utc).date() != origin + timedelta(days=1):
        raise RuntimeError("The completed candle is not from yesterday UTC; cannot issue a current forecast")
    current = float(feat["Close"].iloc[-1])
    if not math.isfinite(current) or current <= 0:
        raise RuntimeError("Invalid current price")
    issue_iso = issued_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for h in HORIZONS:
        df = feat.copy()
        df["target"] = df["Close"].shift(-h) / df["Close"] - 1
        # Shift(-h) ensures training rows finish at or before latest completed candle.
        training = df.iloc[: idx - h + 1].dropna(subset=list(FEATURES) + ["target"])
        if len(training) < MIN_TRAIN:
            raise RuntimeError(f"Not enough matured training examples for {asset}, {h}d")
        # Model names/hyperparameters exactly match Phase 3 backtesting.
        predictions = {}
        for name, model in _models().items():
            model.fit(training[list(FEATURES)], training["target"])
            predictions[name] = float(model.predict(latest[list(FEATURES)])[0])
        # Lightweight benchmark; no extra model training.
        trailing_price = float(feat["Close"].iloc[-8])
        momentum = (current / trailing_price) ** (h / 7.0) - 1.0
        predictions["7-day momentum benchmark"] = momentum
        predictions["No-change baseline"] = 0.0
        # Origin is last completed candle's date; target is h complete UTC candles later.
        target = datetime.combine(origin + timedelta(days=h + 1), time.min, tzinfo=timezone.utc)
        for model, ret in predictions.items():
            price = current * (1 + ret)
            if not math.isfinite(price) or price <= 0:
                print(f"WARNING: skipping invalid {asset} {h}d {model} forecast", file=sys.stderr)
                continue
            rows.append({
                "asset": asset, "model": model, "model_version": MODEL_VERSION,
                "forecast_date": issue_iso, "horizon_days": h,
                "target_date": target.isoformat(timespec="seconds"),
                "starting_price": round(current, 8),
                "predicted_price": round(price, 8),
            })
    return rows


def existing_today(rows: list[dict], history: list[dict]) -> set[tuple]:
    """Do not duplicate an already-issued model/asset/horizon forecast on retries.

    This guards serial retries, not simultaneous writers. Add a DB day-level unique
    constraint before enabling unattended schedules if strict concurrent idempotency
    is required.
    """
    issued_day = rows[0]["forecast_date"][:10]
    return {
        (r["asset"], r["model"], r["model_version"], int(r["horizon_days"]))
        for r in history
        if str(r.get("forecast_date", ""))[:10] == issued_day
    }


def run(assets: list[str], save: bool = False) -> None:
    now = datetime.now(timezone.utc)
    if now.hour == 0 and now.minute < 30:
        print("Note: run after 00:30 UTC for better daily candle availability.")
    predictions = []
    for asset in assets:
        history = completed_daily_history(asset, today_utc=now.date())
        batch = predict_all(history, asset, now)
        predictions.extend(batch)
        print(f"{asset}: {len(batch)} forecasts; latest completed candle "
              f"{history.index[-1].date()}, close ${float(history['Close'].iloc[-1]):,.2f}")
    print(f"Generated {len(predictions)} forecasts ({MODEL_VERSION}).")
    for row in predictions:
        print(f" {row['asset']} {row['horizon_days']:>2}d | {row['model']:<25} "
              f"${row['predicted_price']:,.2f} | target {row['target_date'][:10]}")
    if not save:
        print("DRY RUN: nothing was saved. To write forecasts, run with --save in a trusted environment.")
        return
    # The job is never called from Streamlit and requires external environment secrets.
    historical = list_forecasts(limit=500)
    seen = existing_today(predictions, historical)
    created = skipped = 0
    for row in predictions:
        unique = (row["asset"], row["model"], row["model_version"], row["horizon_days"])
        if unique in seen:
            skipped += 1
            continue
        inserted = save_forecast(**row)
        if inserted:
            created += len(inserted)
        else:
            skipped += 1
        seen.add(unique)
    print(f"Database writes: {created} inserted, {skipped} existing/ignored.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate experimental daily BTC/ETH Phase 3 forecasts")
    parser.add_argument("--asset", choices=["BTC", "ETH", "both"], default="both")
    parser.add_argument("--save", action="store_true", help="Explicitly allow database writes")
    args = parser.parse_args()
    assets = list(SYMBOLS) if args.asset == "both" else [args.asset]
    try:
        run(assets, save=args.save)
        return 0
    except (RuntimeError, ValueError, StorageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
