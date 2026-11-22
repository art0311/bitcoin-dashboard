"""Leakage-aware, expanding-window evaluation for daily crypto forecasts."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 7, 30, 60)
FEATURES = ('ret_1', 'ret_3', 'ret_7', 'ret_14', 'vol_7', 'vol_30',
            'sma_7_gap', 'sma_30_gap', 'rsi_14', 'volume_ratio', 'macd_gap')


def feature_frame(raw):
    """Indicators calculated using only data available at the forecast origin."""
    df = raw[['Close', 'Volume']].copy().sort_index()
    df = df[~df.index.duplicated(keep='last')]
    close = pd.to_numeric(df['Close'], errors='coerce')
    volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=df.index)
    for n in (1, 3, 7, 14):
        out[f'ret_{n}'] = close.pct_change(n, fill_method=None)
    out['vol_7'] = ret.rolling(7).std()
    out['vol_30'] = ret.rolling(30).std()
    out['sma_7_gap'] = close / close.rolling(7).mean() - 1
    out['sma_30_gap'] = close / close.rolling(30).mean() - 1
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    out['rsi_14'] = (100 * up / (up + down).replace(0, np.nan)).fillna(50) / 100
    avg_volume = volume.rolling(30).mean().replace(0, np.nan)
    out['volume_ratio'] = (volume / avg_volume).replace([np.inf, -np.inf], np.nan).fillna(1)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd_gap'] = (ema12 - ema26) / close
    out['Close'] = close
    return out.replace([np.inf, -np.inf], np.nan)


def run_backtest(raw, horizons=HORIZONS, min_train=180, max_tests=24):
    """Evaluate non-overlapping test origins with purged expanding training windows.

    For horizon h, the latest training label at origin i is from i-h,
    so no training target includes future prices unknown at origin i.
    """
    feat = feature_frame(raw)
    results = []
    detail = []
    for h in horizons:
        data = feat.copy()
        data['target'] = data['Close'].shift(-h) / data['Close'] - 1
        # Select test origins after enough labeled training examples exist.
        first_origin = min_train + h + 30
        last_origin = len(data) - h - 1
        if last_origin < first_origin:
            continue
        origins = np.arange(first_origin, last_origin + 1, h, dtype=int)
        if len(origins) > max_tests:
            origins = origins[-max_tests:]
        for i in origins:
            # Train only on fully matured outcomes, no look-ahead.
            train = data.iloc[:i-h+1].dropna(subset=list(FEATURES) + ['target'])
            test = data.iloc[[i]].dropna(subset=list(FEATURES))
            if len(train) < min_train or test.empty:
                continue
            x_train = train[list(FEATURES)]
            y_train = train['target']
            x_test = test[list(FEATURES)]
            models = {
                'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                'Random Forest': RandomForestRegressor(n_estimators=80, max_depth=5,
                                                      min_samples_leaf=8, random_state=42,
                                                      n_jobs=-1),
            }
            actual_price = float(data['Close'].iloc[i+h])
            current_price = float(data['Close'].iloc[i])
            for name, model in models.items():
                model.fit(x_train, y_train)
                pred_ret = float(model.predict(x_test)[0])
                detail.append({'Horizon': h, 'Model': name,
                               'Forecast date': str(data.index[i].date()),
                               'Target date': str(data.index[i+h].date()),
                               'Current price': current_price,
                               'Predicted price': current_price * (1 + pred_ret),
                               'Actual price': actual_price})
            detail.append({'Horizon': h, 'Model': 'No-change baseline',
                           'Forecast date': str(data.index[i].date()),
                           'Target date': str(data.index[i+h].date()),
                           'Current price': current_price,
                           'Predicted price': current_price,
                           'Actual price': actual_price})
    details = pd.DataFrame(detail)
    if details.empty:
        return pd.DataFrame(), details
    details['Absolute error %'] = (100 * (details['Predicted price'] - details['Actual price']).abs()
                                   / details['Actual price'])
    details['Direction correct'] = (np.sign(details['Predicted price'] - details['Current price']) ==
                                    np.sign(details['Actual price'] - details['Current price']))
    # No-change baseline is a price-only comparator: no directional call.
    details['Direction correct'] = details['Direction correct'].astype(object)
    details.loc[details['Model'] == 'No-change baseline', 'Direction correct'] = np.nan
    for (h, model), group in details.groupby(['Horizon', 'Model'], sort=True):
        valid_direction = group['Direction correct'].dropna()
        results.append({'Horizon': f'{h}d', 'Model': model, 'Tests': len(group),
                        'Mean abs. price error': f"{group['Absolute error %'].mean():.2f}%",
                        'Direction accuracy': (f'{100 * valid_direction.astype(float).mean():.1f}%'
                                               if len(valid_direction) else 'N/A')})
    return pd.DataFrame(results), details
