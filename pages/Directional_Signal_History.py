"""Read-only view of the 24h rule-based directional-signal experiment."""
import json
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,urlopen

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Directional Signal History',layout='wide')
st.title('🧭 Directional Signal History')
st.caption('Archived BTC 24-hour rule-based readings, recorded before outcomes. The score is not a calibrated probability.')

@st.cache_data(ttl=120,show_spinner=False)
def load_history(url,key):
    query=urlencode({'select':'id,asset,model_version,issued_at,target_at,starting_price,score_up,score_down,direction,coverage,actual_price,actual_price_at,status,inputs',
                     'order':'issued_at.desc','limit':'1000'})
    req=Request(url.rstrip('/')+'/rest/v1/directional_signal_history?'+query,
                headers={'apikey':key,'Authorization':'Bearer '+key,'Accept':'application/json'})
    try:
        with urlopen(req,timeout=20) as res:return json.loads(res.read())
    except HTTPError as e:raise RuntimeError(f'Supabase returned HTTP {e.code}. Check table setup and secrets.') from None
    except (URLError,TimeoutError):raise RuntimeError('Cannot reach Supabase') from None

try:
    db_url=str(st.secrets['SUPABASE_URL'])
    db_key=str(st.secrets['SUPABASE_KEY'])
    records=load_history(db_url,db_key)
except KeyError:
    st.error('Streamlit Secrets must contain SUPABASE_URL and SUPABASE_KEY.')
    st.stop()
except Exception as e:
    st.error(f'History unavailable: {e}')
    st.stop()

if not records:
    st.info('No directional snapshots yet. The new workflow will start recording after the SQL table and GitHub Actions secret are configured.')
    st.stop()

df=pd.DataFrame(records)
for col in ('issued_at','target_at','actual_price_at'):
    df[col]=pd.to_datetime(df[col],utc=True,errors='coerce')
for col in ('score_up','score_down','starting_price','actual_price','coverage'):
    df[col]=pd.to_numeric(df[col],errors='coerce')

def outcome(r):
    if pd.isna(r['actual_price']):return None
    if r['actual_price']>r['starting_price']:return 'Up'
    if r['actual_price']<r['starting_price']:return 'Down'
    return 'Flat'

df['Actual move']=df.apply(outcome,axis=1)
def correctness(r):
    if pd.isna(r['actual_price']) or r['direction']=='Mixed / neutral' or r['Actual move']=='Flat':return None
    return bool((r['direction']=='Bullish lean' and r['Actual move']=='Up') or (r['direction']=='Bearish lean' and r['Actual move']=='Down'))
df['Correct']=df.apply(correctness,axis=1)

c1,c2,c3,c4=st.columns(4)
c1.metric('Recorded',len(df))
c2.metric('Evaluated',int((df['status']=='evaluated').sum()))
c3.metric('Awaiting outcomes',int((df['status']=='pending').sum()))
count=df['Correct'].notna().sum()
c4.metric('Directional accuracy',f"{100*df['Correct'].dropna().mean():.1f}%" if count else 'N/A')
st.caption(f'Accuracy uses {count} evaluated non-neutral calls. Overlapping snapshots are correlated; a small sample cannot establish a predictive edge.')

st.subheader('Upside score over time')
limit=st.selectbox('Display recent', [48,168,336,720,1000],index=1,format_func=lambda n:f'Last {n} snapshots')
chart=df.sort_values('issued_at').tail(limit).set_index('issued_at')[['score_up']].rename(columns={'score_up':'Upside score (0–100)'})
st.line_chart(chart)
st.caption('50 is neutral. Higher values represent a stronger rule-based upside lean, not a percentage probability.')

st.subheader('Recorded signals')
status=st.selectbox('Status',['All','Pending','Evaluated'])
view=df if status=='All' else df[df['status']==status.lower()]
cols={'issued_at':'Issued (UTC)','score_up':'Upside score','score_down':'Downside score',
      'direction':'Direction','coverage':'Coverage','starting_price':'BTC at issue',
      'target_at':'24h target (UTC)','actual_price':'BTC at target','Actual move':'Actual move',
      'Correct':'Correct','status':'Status'}
shown=view[list(cols)].rename(columns=cols)
st.dataframe(shown,use_container_width=True,hide_index=True)
st.download_button('Download signal history (CSV)',shown.to_csv(index=False).encode(),
                   file_name='directional_signal_history.csv',mime='text/csv')
with st.expander('What this experiment measures'):
    st.markdown('The 24-hour signal combines price momentum, liquidation estimates, funding, long/short accounts, and ETF flows. Each reading is archived with its inputs, and later evaluated using a BTC price candle ending near the target instant. Scores are heuristic and are **not** probabilities. Evaluations of closely spaced signals are not independent; results need a substantial track record and comparison with simple baselines before trading use.')
