"""Phase 2.1: leakage-aware BTC/ETH daily-return forecasting and walk-forward evaluation.

This module is independent of Streamlit. It makes no API calls and writes no files.
Training at origin i includes only labels whose target date is <= origin i.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 7, 30, 60)
FEATURES = (
    'ret_1', 'ret_3', 'ret_7', 'ret_14', 'ret_30',
    'vol_7', 'vol_30', 'sma_7_gap', 'sma_30_gap',
    'rsi_14', 'volume_ratio', 'macd_gap', 'macd_hist',
    'range_1', 'volume_change_7',
)


def feature_frame(raw):
    """Create features known at each daily candle close; no future values or global scaling."""
    required = ('Close', 'High', 'Low', 'Volume')
    if raw is None or raw.empty or any(c not in raw.columns for c in required):
        return pd.DataFrame(columns=list(FEATURES) + ['Close'])
    df = raw.loc[:, list(required)].copy().sort_index()
    df = df[~df.index.duplicated(keep='last')]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close', 'High', 'Low'])
    df = df[df['Close'] > 0]
    close = df['Close']
    volume = df['Volume'].fillna(0).clip(lower=0)
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=df.index)
    for n in (1, 3, 7, 14, 30):
        out[f'ret_{n}'] = close.pct_change(n, fill_method=None)
    out['vol_7'] = ret.rolling(7, min_periods=7).std()
    out['vol_30'] = ret.rolling(30, min_periods=30).std()
    out['sma_7_gap'] = close / close.rolling(7, min_periods=7).mean() - 1
    out['sma_30_gap'] = close / close.rolling(30, min_periods=30).mean() - 1
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rs_denom = up + down
    out['rsi_14'] = (100 * up / rs_denom.where(rs_denom > 0)).fillna(50) / 100
    avg_vol = volume.rolling(30, min_periods=30).mean()
    out['volume_ratio'] = volume / avg_vol.where(avg_vol > 0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out['macd_gap'] = macd / close
    out['macd_hist'] = (macd - signal) / close
    out['range_1'] = (df['High'] - df['Low']) / close
    out['volume_change_7'] = volume / volume.shift(7).replace(0, np.nan) - 1
    out['Close'] = close
    return out.replace([np.inf, -np.inf], np.nan)


def _models():
    return {
        'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Random Forest': RandomForestRegressor(
            n_estimators=80, max_depth=5, min_samples_leaf=8,
            max_features=0.8, random_state=42, n_jobs=-1,
        ),
    }


def _test_origins(first, last, horizon, max_tests):
    """Non-overlapping target periods; recent tests preferred when capped."""
    if last < first:
        return np.array([], dtype=int)
    origins = np.arange(first, last + 1, horizon, dtype=int)
    return origins[-max_tests:] if max_tests and len(origins) > max_tests else origins


def run_backtest(raw, horizons=HORIZONS, min_train=180, max_tests=60):
    """Return (summary, individual predictions), using strict time-series testing.

    Every observation predicts an h-candle *return*, converted into a future price.
    At test origin i, training rows j are allowed only when j+h <= i.
    All model preprocessing is fitted on training rows only.
    """
    feat = feature_frame(raw)
    if feat.empty:
        return pd.DataFrame(), pd.DataFrame()
    detail = []
    for h in horizons:
        h = int(h)
        if h < 1:
            continue
        data = feat.copy()
        data['target'] = data['Close'].shift(-h) / data['Close'] - 1
        first_origin = min_train + h + 35
        last_origin = len(data) - h - 1
        for i in _test_origins(first_origin, last_origin, h, max_tests):
            # Training is purged by horizon; the latest training label matures at i.
            train = data.iloc[:i-h+1].dropna(subset=list(FEATURES) + ['target'])
            test = data.iloc[[i]].dropna(subset=list(FEATURES))
            if len(train) < min_train or test.empty:
                continue
            current = float(data['Close'].iloc[i])
            actual = float(data['Close'].iloc[i+h])
            actual_return = 100.0 * (actual / current - 1)
            base = {
                'Horizon': h,
                'Forecast date': str(data.index[i].date()),
                'Target date': str(data.index[i+h].date()),
                'Current price': current,
                'Actual price': actual,
                'Actual return %': actual_return,
            }
            for name, model in _models().items():
                model.fit(train[list(FEATURES)], train['target'])
                forecast_return = float(model.predict(test[list(FEATURES)])[0])
                # Don't silently clip model outputs: backtest should expose failures.
                predicted_price = current * (1 + forecast_return)
                if not np.isfinite(predicted_price):
                    continue
                detail.append({**base, 'Model': name,
                               'Predicted price': predicted_price,
                               'Predicted return %': forecast_return * 100.0})
            # Lightweight directional benchmark: predict continuation of the last 7-day return.
            # This is available at the forecast origin and requires no model fitting.
            momentum_7d = float(data['Close'].iloc[i] / data['Close'].iloc[i-7] - 1)
            detail.append({**base, 'Model': '7-day momentum benchmark',
                           'Predicted price': current * (1 + momentum_7d),
                           'Predicted return %': momentum_7d * 100.0})
            detail.append({**base, 'Model': 'No-change baseline',
                           'Predicted price': current, 'Predicted return %': 0.0})
    details = pd.DataFrame(detail)
    if details.empty:
        return pd.DataFrame(), details
    details['Absolute error %'] = (100 * (details['Predicted price'] - details['Actual price']).abs()
                                   / details['Actual price'])
    details['Return error (pp)'] = (details['Predicted return %'] - details['Actual return %']).abs()
    details['Direction correct'] = (np.sign(details['Predicted return %']) ==
                                    np.sign(details['Actual return %']))
    details['Direction correct'] = details['Direction correct'].astype(object)
    details.loc[details['Model'] == 'No-change baseline', 'Direction correct'] = np.nan
    baseline_by_horizon = (
        details[details['Model'] == 'No-change baseline']
        .groupby('Horizon')['Absolute error %'].mean().to_dict()
    )
    summary = []
    for (h, model), group in details.groupby(['Horizon', 'Model'], sort=True):
        direction = group['Direction correct'].dropna()
        baseline_error = baseline_by_horizon.get(h, np.nan)
        model_error = group['Absolute error %'].mean()
        # Positive skill means lower price error than the no-change baseline.
        skill = (100.0 * (baseline_error - model_error) / baseline_error
                 if np.isfinite(baseline_error) and baseline_error > 0 else np.nan)
        summary.append({
            'Horizon': f'{h}d', 'Model': model, 'Tests': len(group),
            'Mean abs. price error': f"{group['Absolute error %'].mean():.2f}%",
            'Mean abs. return error': f"{group['Return error (pp)'].mean():.2f} pp",
            'Error vs baseline': (f'{model_error - baseline_error:+.2f} pp'
                                  if model != 'No-change baseline' else '—'),
            'Model Skill': (f'{skill:+.1f}%' if model != 'No-change baseline'
                            and np.isfinite(skill) else ('0.0%' if model == 'No-change baseline' else 'N/A')),
            'Direction accuracy': (f'{100 * direction.astype(float).mean():.1f}%'
                                   if len(direction) else 'N/A'),
        })
    return pd.DataFrame(summary), details
