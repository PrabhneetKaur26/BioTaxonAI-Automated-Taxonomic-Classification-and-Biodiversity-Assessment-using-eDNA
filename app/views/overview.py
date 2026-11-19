"""BioTaxonAI: entry point.  Run with:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

# Make the project root importable so pages can use `src` and `app`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import nav, ui  # noqa: E402
from src.edna import store  # noqa: E402

st.set_page_config(page_title="BioTaxonAI", page_icon="🧬", layout="wide")
ui.apply_style()

if not store.results_available():
    st.title("BioTaxonAI")
    st.warning(store.NOT_READY_MESSAGE)
    st.stop()

pages = nav.build_pages()
navigation = st.navigation(list(pages.values()), position="hidden")
nav.render_sidebar(pages)
navigation.run()