"""Page registry and the custom sidebar navigation.

All eight pages are listed here. A page becomes a live link automatically as soon as its
file exists in app/views/, so we can build the app one page at a time.
"""
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from app import ui

APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PageSpec:
    slug: str
    title: str
    icon: str
    file: str


PAGES = [
    PageSpec("overview", "Overview", ":material/dashboard:", "views/overview.py"),
    PageSpec("sequence-space", "Sequence Space", ":material/scatter_plot:", "views/sequence_space.py"),
    PageSpec("clusters", "Clusters", ":material/layers:", "views/clusters.py"),
    PageSpec("method-validation", "Method Validation", ":material/bar_chart:", "views/method_validation.py"),
    PageSpec("biodiversity", "Biodiversity", ":material/public:", "views/biodiversity.py"),
    PageSpec("novelty", "Novelty Candidates", ":material/warning:", "views/novelty.py"),
    PageSpec("analyse", "Analyse Sequences", ":material/cloud_upload:", "views/analyse.py"),
    PageSpec("methodology", "Methodology", ":material/menu_book:", "views/methodology.py"),
]

_registry = {}  # slug -> st.Page for pages that exist; refreshed on every run


def build_pages():
    """Create st.Page objects for the pages whose files exist. The first page is the default."""
    pages = {}
    for position, spec in enumerate(PAGES):
        if (APP_DIR / spec.file).exists():
            if position == 0:
                pages[spec.slug] = st.Page(spec.file, title=spec.title, icon=spec.icon, default=True)
            else:
                pages[spec.slug] = st.Page(spec.file, title=spec.title, icon=spec.icon, url_path=spec.slug)
    _registry.clear()
    _registry.update(pages)
    return pages


def render_sidebar(pages):
    """Brand block, navigation links (unbuilt pages shown as 'soon'), and the footer note."""
    with st.sidebar:
        ui.html_block(ui.brand_html())
        for spec in PAGES:
            page = pages.get(spec.slug)
            if page is not None:
                st.page_link(page, label=spec.title, icon=spec.icon)
            else:
                ui.html_block(f'<div class="bt-nav-soon"><i></i>{spec.title}<em>soon</em></div>')
        ui.html_block(ui.sidebar_footer_html())


def link_to(slug, label):
    """A link to another page, or plain greyed text while that page does not exist yet."""
    page = _registry.get(slug)
    if page is not None:
        st.page_link(page, label=label)
    else:
        ui.html_block(f'<div class="bt-soon-link">{label}</div>')