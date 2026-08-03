import requests
import streamlit as st

st.title("CoinGlass API Test")

API_KEY = st.secrets["COINGLASS_API_KEY"]

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history"

headers = {
    "accept": "application/json",
    "CG-API-KEY": API_KEY,
}

response = requests.get(url, headers=headers, timeout=15)

st.write("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()
    st.success("API connection successful!")
    st.json(data)
else:
    st.error("API request failed")
    st.code(response.text)
