"""Server-side access for 24h directional snapshots, segregated from Phase 3.
Use only with trusted GitHub Actions environment secrets. No writes from Streamlit.
"""
from __future__ import annotations
import json,os
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

TABLE='directional_signal_history'
class StoreError(RuntimeError): pass

def request(method, *, filters=None, payload=None, prefer=None):
    base=os.environ.get('SUPABASE_URL','').rstrip('/')
    key=os.environ.get('SUPABASE_KEY','')
    if not base or not key: raise StoreError('SUPABASE_URL / SUPABASE_KEY not configured')
    url=f'{base}/rest/v1/{TABLE}'
    if filters: url += '?' + urlencode(filters)
    headers={'apikey':key,'Authorization':'Bearer '+key,'Accept':'application/json'}
    if payload is not None: headers['Content-Type']='application/json'
    if prefer: headers['Prefer']=prefer
    data=json.dumps(payload,allow_nan=False).encode() if payload is not None else None
    try:
        with urlopen(Request(url,data=data,headers=headers,method=method),timeout=22) as response:
            text=response.read().decode()
            return json.loads(text) if text else []
    except HTTPError as e:
        raise StoreError(f'Supabase {TABLE}: HTTP {e.code} (check table/schema/credentials)') from None
    except (URLError,TimeoutError):
        raise StoreError('Supabase network request failed') from None

def insert_snapshot(row):
    return request('POST',filters={'on_conflict':'asset,model_version,sample_bucket'},payload=row,
                   prefer='resolution=ignore-duplicates,return=representation')

def due_snapshots(limit=250):
    from datetime import datetime,timezone
    now=datetime.now(timezone.utc).isoformat()
    return request('GET',filters={'select':'id,asset,issued_at,target_at,starting_price,score_up,direction',
                       'status':'eq.pending','target_at':'lte.'+now,'order':'target_at.asc','limit':str(limit)})

def record_actual(row_id,actual,price_at,eval_at):
    return request('PATCH',filters={'id':'eq.'+str(row_id),'status':'eq.pending','select':'id'},
                   payload={'actual_price':actual,'actual_price_at':price_at,'evaluated_at':eval_at,'status':'evaluated'},
                   prefer='return=representation')
