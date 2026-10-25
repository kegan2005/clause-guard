"""ClauseGuard entry point: ``streamlit run app.py``.

This file sets the page config once, injects the shared CSS, builds the sidebar
navigation with ``st.navigation`` and draws the status bar.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ClauseGuard - Contract Clause Classifier", layout="wide", initial_sidebar_state="expanded")

import ui  # noqa: E402  (must come after set_page_config)

ui.inject_css()

PAGES = [
    st.Page("pages/1_Analyze.py", title="Analyze", url_path="analyze", default=True),
    st.Page("pages/2_Model_Performance.py", title="Model Performance", url_path="performance"),
    st.Page("pages/3_Compare_Models.py", title="Compare Models", url_path="compare"),
    st.Page("pages/4_About.py", title="About", url_path="about"),
]

navigation = st.navigation(PAGES)
ui.sidebar_status()
navigation.run()
ui.status_bar(f"ClauseGuard | {navigation.title}")
