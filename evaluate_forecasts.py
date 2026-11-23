"""Evaluate matured BTC/ETH forecasts against completed Yahoo Finance daily candles.

Run from a trusted backend only. Needs SUPABASE_URL and SUPABASE_KEY environment
variables. Dry-run by default; explicitly pass --save to update Supabase.

A forecast target_date denotes the UTC *end* of its target candle. For example,
2026-11-24T00:00:00+00:00 is evaluated using the 2026-11-23 daily Close.
Never evaluate against a currently forming candle or overwrite an actual_price.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
import sys

import pandas as pd
import yfinance as yf

from forecast_storage import StorageError, _request, record_outcome

SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
BATCH_SIZE = 500


def due_forecasts(now: datetime, limit: int = BATCH_SIZE) -> list[dict]:
    """Fetch oldest unevaluated forecasts whose target instant has passed.

    The query is server-filtered instead of using list_forecasts(limit=500),
    which would silently omit older pending forecasts after months of daily runs.
    """
    return _request("GET", query={
        "select": "id,asset,model,model_version,forecast_date,horizon_days,target_date,starting_price,predicted_price,actual_price",
        "actual_price": "is.null",
        "target_date": "lte." + now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "order": "target_date.asc,id.asc",
        "limit": str(limit),
    })


def candle_dates(rows: list[dict], now: datetime) -> dict[str, set]:
    """Targets at UTC midnight map to the preceding day's completed daily candle."""
    result: dict[str, set] = defaultdict(set)
    today = now.astimezone(timezone.utc).date()
    for row in rows:
        asset = str(row["asset"])
        if asset not in SYMBOLS:
            raise ValueError("Unexpected asset in Supabase forecast records")
        target = datetime.fromisoformat(str(row["target_date"]).replace("Z", "+00:00"))
        if target.tzinfo is None:
            raise ValueError("Target date is not timezone-aware")
        boundary = target.astimezone(timezone.utc)
        if (boundary.hour, boundary.minute, boundary.second, boundary.microsecond) != (0, 0, 0, 0):
            raise ValueError("Expected UTC-midnight target boundary; refusing ambiguous candle mapping")
        candle_day = boundary.date() - timedelta(days=1)
        if candle_day >= today:
            continue  # Incomplete candle, even if the stored target timestamp is malformed
        result[asset].add(candle_day)
    return result


def fetch_actual_closes(asset: str, dates: set, now: datetime) -> dict:
    """Download only needed OHLC daily history for one asset, excluding today."""
    if not dates:
        return {}
    first = min(dates) - timedelta(days=2)
    today = now.astimezone(timezone.utc).date()
    raw = yf.download(
        SYMBOLS[asset], start=first.isoformat(), end=(today + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=True, progress=False, threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No daily market prices available for {asset}; retry later")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw:
        raise RuntimeError(f"Daily Close missing for {asset}")
    index = pd.to_datetime(raw.index, utc=True).normalize()
    closes = {}
    for ts, close in zip(index, raw["Close"]):
        day = ts.date()
        if day in dates and day < today:
            price = float(close)
            if math.isfinite(price) and price > 0:
                closes[day] = price
    return closes


def evaluate(now: datetime | None = None, save: bool = False) -> tuple[int, int, int]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Current time must be timezone-aware")
    # No pagination offset: on a save run, completed records drop out of the pending
    # set. A subsequent execution picks up any additional batches beyond the first 500.
    rows = due_forecasts(now)
    if not rows:
        print("No matured, unevaluated forecasts found.")
        return 0, 0, 0
    targets = candle_dates(rows, now)
    prices = {asset: fetch_actual_closes(asset, days, now) for asset, days in targets.items()}
    evaluated = missing = skipped = 0
    for row in rows:
        target = datetime.fromisoformat(str(row["target_date"]).replace("Z", "+00:00"))
        day = target.astimezone(timezone.utc).date() - timedelta(days=1)
        asset = str(row["asset"])
        price = prices.get(asset, {}).get(day)
        if price is None:
            missing += 1
            continue
        if save:
            updated = record_outcome(
                forecast_id=int(row["id"]), actual_price=price,
                evaluated_at=now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            )
            if not updated:
                skipped += 1  # Already evaluated by another worker
                continue
        evaluated += 1
        print(f"{asset} | forecast #{row['id']} | {row['horizon_days']}d | "
              f"candle {day} | actual close ${price:,.2f} | "
              f"{'SAVED' if save else 'DRY RUN'}")
    print(f"Evaluation summary: {len(rows)} due; {evaluated} "
          f"{'saved' if save else 'eligible'}; {missing} missing closes; {skipped} already evaluated.")
    if len(rows) == BATCH_SIZE:
        print("Note: batch limit reached. Re-run evaluation to process additional pending rows.")
    if not save:
        print("DRY RUN: no database updates. Run --save in a trusted job to record actual prices.")
    return evaluated, missing, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate completed BTC/ETH forecasts")
    parser.add_argument("--save", action="store_true", help="Persist actual prices to Supabase")
    args = parser.parse_args()
    try:
        evaluate(save=args.save)
        return 0
    except (RuntimeError, ValueError, StorageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
