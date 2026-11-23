
import streamlit as st
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

st.title("Database Connection Test")

st.write("Checking the connection to Supabase...")

try:
    url = st.secrets["SUPABASE_URL"].rstrip("/")
    key = st.secrets["SUPABASE_KEY"]

    request = Request(
        f"{url}/rest/v1/forecast_predictions?select=id&limit=1",
        headers={
            "apikey": key,
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=15) as response:
        if response.status == 200:
            st.success("Supabase connection successful!")
            st.info("Forecast history table is accessible.")
        else:
            st.error(f"Unexpected response: {response.status}")

except KeyError:
    st.error("Supabase credentials are missing from Streamlit Secrets.")
except HTTPError as error:
    st.error(f"Database request failed: HTTP {error.code}")
except URLError:
    st.error("Could not reach the Supabase server.")
except Exception:
    st.error("Database connection test failed.")
