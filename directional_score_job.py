"""Generate and evaluate 24h directional scores; independent of Phase 3.

Execute without --save for a read-only preview. Scheduled workflow uses --save.
Scores are rule-based and not calibrated probabilities. No trading operations.
"""
from __future__ import annotations
import argparse,json,math,os,sys
from datetime import datetime,timezone,timedelta

import pandas as pd
import yfinance as yf

from directional_score_core import compute_directional_outlook
from directional_sources import fetch_coinalyze_context,fetch_modeled_liquidation_zones,fetch_bitcoin_etf_history
from directional_score_storage import insert_snapshot,due_snapshots,record_actual,StoreError

ASSET='BTC'
MODEL_VERSION='directional-heuristic-v1'

def prices(interval,period):
    raw=yf.download('BTC-USD',period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if raw is None or raw.empty: raise RuntimeError('Yahoo Finance BTC candles unavailable')
    if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
    if 'Close' not in raw:raise RuntimeError('Missing Close column')
    close=raw['Close'].dropna().astype(float)
    close.index=pd.to_datetime(close.index,utc=True)
    close=close[~close.index.duplicated(keep='last')].sort_index()
    return close

def validated_live_price():
    close=prices('5m','2d')
    ts=close.index[-1].to_pydatetime()
    now=datetime.now(timezone.utc)
    age=(now-ts).total_seconds()
    if not (0<=age<=1800):
        raise RuntimeError('Latest 5-minute BTC price is stale/future; no snapshot stored')
    price=float(close.iloc[-1]);
    if not math.isfinite(price) or price<=0:raise RuntimeError('Invalid BTC price')
    return price,ts

def capture(save=False):
    key=os.getenv('COINALYZE_API_KEY','').strip()
    if not key:raise RuntimeError('COINALYZE_API_KEY missing in GitHub Actions secrets')
    price,price_time=validated_live_price()
    daily=prices('1d','1mo')
    context=fetch_coinalyze_context(ASSET,key)
    zones=fetch_modeled_liquidation_zones(ASSET)
    etf=fetch_bitcoin_etf_history()
    # Explicit freshness check: a stale derivatives feed must not create a seemingly current signal.
    for stamp_name,limit in [('ratio_asof',6),('oi_asof',6)]:
        value=context.get(stamp_name)
        if value:
            timestamp=datetime.strptime(value,'%Y-%m-%d %H:%M UTC').replace(tzinfo=timezone.utc)
            if abs((datetime.now(timezone.utc)-timestamp).total_seconds())>limit*3600:
                if stamp_name=='ratio_asof':context['long_pct']=context['short_pct']=None
                if stamp_name=='oi_asof':context['oi_usd']=context['oi_change_pct']=None
    outlook=compute_directional_outlook(context,zones,daily,etf)
    issued_at=datetime.now(timezone.utc)
    print(f'BTC current ${price:,.2f}; data coverage {outlook["coverage"]:.0%}; ready={outlook["ready"]}; sources: coinalyze={context.get("available",0)}, zones={zones.get("valid")}, etf_sessions={len(etf)}')
    if not outlook['ready']:
        raise RuntimeError('Directional score not ready: '+outlook['reason'])
    # Fixed half-hour bucket for retries/deduplication. Actual issuance is never backdated.
    bucket=issued_at.replace(minute=(issued_at.minute//30)*30,second=0,microsecond=0)
    def v(x):
        return float(x) if x is not None and math.isfinite(float(x)) else None
    row={
      'asset':ASSET,'model_version':MODEL_VERSION,
      'sample_bucket':bucket.isoformat(),'issued_at':issued_at.isoformat(),
      'target_at':(issued_at+timedelta(hours=24)).isoformat(),
      'starting_price':price,'starting_price_at':price_time.isoformat(),
      'score_up':int(outlook['up_score']),'score_down':int(outlook['down_score']),
      'direction':outlook['direction'],'coverage':round(float(outlook['coverage']),4),
      'components':outlook['components'],
      'inputs':{
        'funding_pct':v(context.get('funding_pct')),'open_interest_usd':v(context.get('oi_usd')),
        'open_interest_change_pct':v(context.get('oi_change_pct')),
        'long_accounts_pct':v(context.get('long_pct')),
        'short_accounts_pct':v(context.get('short_pct')),
        'contract':context.get('contract'),'exchange':context.get('exchange'),
        'funding_asof':context.get('asof'),'oi_asof':context.get('oi_asof'),
        'ratio_asof':context.get('ratio_asof'),
        'liquidation_model_asof':zones.get('asof'),
        'liquidation_model_version':zones.get('version'),
        'liquidation_reference_price':v(zones.get('price')),
        'liquidation_long_below':zones.get('long'),
        'liquidation_short_above':zones.get('short'),
        'etf_latest_session':etf[-1][0] if etf else None,
        'etf_last_7_usdm':sum(float(x[1]) for x in etf[-7:]) if len(etf)>=7 else None,
        'source':'Yahoo BTC-USD, Coinalyze, Swolecharts, TFTC',
      }
    }
    print(f'{issued_at.isoformat()} | {outlook["direction"]} | upside {outlook["up_score"]}/100 | target {row["target_at"]}')
    if not save:
        print('DRY RUN: no Supabase write.');return
    inserted=insert_snapshot(row)
    print('Database writes: '+('1 inserted' if inserted else '0 inserted; bucket already exists'))

def evaluate(save=False):
    rows=due_snapshots()
    print(f'Due directional records: {len(rows)}')
    if not rows:return
    # 5-minute close series: target is exactly 24h after issue. Use first completed
    # 5-minute candle at/after target, with at most a 15-minute tolerance.
    hist=prices('5m','30d')
    now=datetime.now(timezone.utc)
    saved=skipped=0
    for row in rows:
        target=pd.Timestamp(row['target_at']).tz_convert('UTC')
        # Candles indexed at bar OPEN; bar closes five minutes later.
        candle_close_times=hist.index+pd.Timedelta(minutes=5)
        eligible=(candle_close_times >= target)&(candle_close_times <= target+pd.Timedelta(minutes=15))&(candle_close_times <=pd.Timestamp(now))
        if not eligible.any():
            print(f"No time-matched BTC candle within 15m of target for ID {row['id']}");skipped+=1;continue
        candidates=hist[eligible]
        actual=float(candidates.iloc[0]);at=candidates.index[0]+pd.Timedelta(minutes=5)
        if not math.isfinite(actual) or actual<=0:skipped+=1;continue
        start=float(row['starting_price'])
        outcome='up' if actual>start else ('down' if actual<start else 'flat')
        direction=row['direction']
        correct=(direction=='Bullish lean' and outcome=='up') or (direction=='Bearish lean' and outcome=='down')
        print(f"ID {row['id']}: ${start:,.2f} -> ${actual:,.2f}, actual={outcome}, predicted={direction}, {'correct' if correct else 'not correct'}")
        if save:
            updated=record_actual(row['id'],actual,at.isoformat(),now.isoformat())
            saved+=len(updated)
    print(f'Evaluations: {saved} saved; {skipped} with no matching candle. {"DRY RUN" if not save else "SAVE"}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    try:
        errors=[]
        if not args.evaluate_only:
            try:capture(args.save)
            except (RuntimeError,ValueError,StoreError) as e:
                print('CAPTURE ERROR:',e,file=sys.stderr);errors.append(str(e))
        if not args.capture_only:
            try:evaluate(args.save)
            except (RuntimeError,ValueError,StoreError) as e:
                print('EVALUATION ERROR:',e,file=sys.stderr);errors.append(str(e))
        return 1 if errors else 0
    except Exception as e:
        print('UNEXPECTED ERROR:',type(e).__name__,file=sys.stderr)
        return 1
if __name__=='__main__':raise SystemExit(main())
