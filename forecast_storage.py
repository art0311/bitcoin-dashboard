"""Server-side Supabase storage for crypto forecasts.

Secrets are kept in Streamlit/GitHub Actions settings; never in source files.
Do not call write functions from public Streamlit widgets.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TABLE = "forecast_predictions"
VALID_ASSETS = {"BTC", "ETH"}
VALID_HORIZONS = {1, 7, 30, 60}

class StorageError(RuntimeError):
    """A forecast database operation failed."""


def _credentials(secrets: Mapping[str, str] | None = None) -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL") or (secrets.get("SUPABASE_URL") if secrets is not None else None)
    key = os.environ.get("SUPABASE_KEY") or (secrets.get("SUPABASE_KEY") if secrets is not None else None)
    if not url or not key:
        raise StorageError("Supabase URL or secret key is not configured.")
    return str(url).rstrip("/"), str(key)


def _request(method: str, secrets: Mapping[str, str] | None = None, *,
             query: Mapping[str, str] | None = None,
             body: dict[str, Any] | None = None,
             prefer: str | None = None) -> Any:
    url, key = _credentials(secrets)
    endpoint = f"{url}/rest/v1/{TABLE}"
    if query:
        endpoint += "?" + urlencode(query)
    headers = {"apikey": key, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if prefer:
        headers["Prefer"] = prefer
    request = Request(endpoint, data=json.dumps(body).encode("utf-8") if body is not None else None,
                      headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            data = response.read()
            return json.loads(data) if data else []
    except HTTPError as exc:
        raise StorageError(f"Supabase returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError):
        raise StorageError("Supabase could not be reached.") from None


def list_forecasts(secrets: Mapping[str, str] | None = None, *,
                   limit: int = 20, offset: int = 0,
                   model: str | None = None) -> list[dict[str, Any]]:
    """Read recent records in pages; order is stable across tied timestamps."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    query = {
        "select": "id,asset,model,model_version,forecast_date,horizon_days,target_date,starting_price,predicted_price,actual_price,evaluated_at",
        "order": "forecast_date.desc,id.desc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if model is not None:
        query["model"] = "eq." + model
    return _request("GET", secrets, query=query)


def save_forecast(secrets: Mapping[str, str] | None = None, *,
                  asset: str, model: str, model_version: str,
                  forecast_date: str, horizon_days: int, target_date: str,
                  starting_price: float, predicted_price: float) -> list[dict[str, Any]]:
    """Record an issued forecast; on_conflict ignores duplicate records."""
    if asset not in VALID_ASSETS or horizon_days not in VALID_HORIZONS:
        raise ValueError("Unsupported asset or forecast horizon")
    if not model.strip() or not model_version.strip():
        raise ValueError("Model and model_version are required")
    for value in (forecast_date, target_date):
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None or date.utcoffset() is None:
            raise ValueError("Forecast dates must include a timezone offset")
    if float(starting_price) <= 0 or float(predicted_price) <= 0:
        raise ValueError("Prices must be positive")
    row = dict(asset=asset, model=model, model_version=model_version,
               forecast_date=forecast_date, horizon_days=horizon_days, target_date=target_date,
               starting_price=starting_price, predicted_price=predicted_price)
    return _request("POST", secrets,
        query={"on_conflict": "asset,model,model_version,forecast_date,horizon_days"},
        body=row, prefer="resolution=ignore-duplicates,return=representation")


def record_outcome(secrets: Mapping[str, str] | None = None, *,
                   forecast_id: int, actual_price: float, evaluated_at: str) -> list[dict[str, Any]]:
    """Set a matured result, without overwriting an existing actual price."""
    if forecast_id <= 0 or actual_price <= 0:
        raise ValueError("Invalid forecast ID or price")
    dt = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone offset")
    return _request("PATCH", secrets,
        query={"id": f"eq.{forecast_id}", "actual_price": "is.null",
               "select": "id,actual_price,evaluated_at"},
        body={"actual_price": actual_price, "evaluated_at": evaluated_at},
        prefer="return=representation")
