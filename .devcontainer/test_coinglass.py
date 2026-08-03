import requests
import streamlit as st

st.title("CoinGlass API Test")

API_KEY = st.secrets["COINGLASS_API_KEY"]

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history"

headers = {
    "accept": "application/json",
    "CG-API-KEY": API_KEY,
}

try:
    response = requests.get(
        url,
        headers=headers,
        timeout=15,
    )

    st.write("Status:", response.status_code)
    st.code(response.text[:3000])

except Exception as error:
    st.error(f"Request failed: {error}")
