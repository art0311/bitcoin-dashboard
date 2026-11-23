"""Read-only database connection test page for the Streamlit dev app."""
import streamlit as st

from forecast_storage import StorageError, list_forecasts

st.title("Database Connection Test")
st.caption("Read-only check. This page cannot save, change, or delete forecasts.")

try:
    records = list_forecasts(st.secrets, limit=1)
    st.success("Supabase connection successful!")
    st.info("Forecast history table is accessible.")
    if records:
        st.success("At least one forecast record is stored and readable.")
    else:
        st.warning("The connection works, but the forecast table is currently empty.")
except StorageError as exc:
    st.error(str(exc))
except Exception:
    st.error("Connection test failed. Check app logs for configuration issues; do not share secret keys.")
