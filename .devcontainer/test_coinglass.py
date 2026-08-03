import streamlit as st

API_KEY = st.secrets["COINGLASS_API_KEY"]

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history"

headers = {
    "accept": "application/json",
    "CG-API-KEY": API_KEY,
}

r = requests.get(url, headers=headers, timeout=15)

print("Status:", r.status_code)
print(r.text)
