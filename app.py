import streamlit as st

st.set_page_config(
    page_title="TNRVision",
    layout="wide",
)

from app.ui import run_app

run_app()