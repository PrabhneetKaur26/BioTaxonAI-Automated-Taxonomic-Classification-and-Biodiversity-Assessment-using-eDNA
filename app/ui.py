"""BioTaxonAI design system: palette, CSS, HTML components and chart styling.

Every page imports from here, so the look stays consistent and changes in one place.
"""
import html
import math
import re

import streamlit as st

REPO_URL = ("https://github.com/PrabhneetKaur26/"
            "BioTaxonAI-Automated-Taxonomic-Classification-and-Biodiversity-Assessment-using-eDNA")

TEAL, SKY, AMBER, PURPLE, GREEN = "#2DD4BF", "#38BDF8", "#FBBF24", "#A78BFA", "#34D399"
TEXT, MUTED, FAINT = "#E7EEF8", "#9FB0C8", "#6B7C99"

# One colour per K-Means cluster id and per reference group, reused on every page.
CLUSTER_COLORS = ["#38BDF8", "#34D399", "#F43F5E", "#A855F7", "#F59E0B",
                  "#06B6D4", "#EC4899", "#84CC16", "#6366F1", "#F97316", "#14B8A6", "#E879F9"]
GROUP_COLORS = {"Metazoa": "#06B6D4", "Radiolaria": "#EC4899", "Dinoflagellata": "#84CC16",
                "Ciliophora": "#6366F1", "Unassigned": "#F97316", "Haptophyta": "#22D3EE"}


def cluster_color(cluster_id):
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


# ----------------------------------------------------------------------------- CSS ------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0B1224; --sidebar:#090F1E; --card:#101A33; --border:#1C2A4A; --soft:rgba(148,163,184,.14);
  --text:#E7EEF8; --muted:#9FB0C8; --faint:#6B7C99; --teal:#2DD4BF; --sky:#38BDF8;
  --mono:'JetBrains Mono','Cascadia Mono',Consolas,monospace;
}
html, body, .stApp, .stMarkdown, button, input, textarea, select, [class*="css"]{
  font-family:'Inter','Segoe UI',system-ui,sans-serif;
}
.stApp{ background:radial-gradient(1100px 520px at 72% -8%, rgba(45,212,191,.07), transparent 62%), var(--bg); }

/* chrome */
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer{ display:none !important; }
[data-testid="stMainBlockContainer"], .block-container{
  padding:5.4rem 2.4rem 4rem 2.4rem; max-width:1480px; animation:btFade .5s ease both;
}
*::-webkit-scrollbar{ width:10px; height:10px; }
*::-webkit-scrollbar-thumb{ background:#22304F; border-radius:8px; }
*::-webkit-scrollbar-track{ background:transparent; }

/* sidebar */
section[data-testid="stSidebar"]{ background:var(--sidebar); border-right:1px solid var(--soft); width:300px !important; min-width:300px; }
[data-testid="stSidebarContent"]{ padding:1.5rem 1rem 1rem 1rem; }
.bt-brand{ display:flex; align-items:center; gap:.85rem; padding:.2rem .6rem 1.6rem .6rem; }
.bt-logo{ width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background:linear-gradient(135deg,#2DD4BF,#38BDF8); box-shadow:0 8px 24px rgba(45,212,191,.28);
  transition:transform .35s ease, box-shadow .35s ease; }
.bt-logo:hover{ transform:rotate(-6deg) scale(1.05); box-shadow:0 10px 30px rgba(56,189,248,.4); }
.bt-brand-name{ font-weight:700; font-size:1.18rem; letter-spacing:-.02em; color:var(--text); line-height:1.15; }
.bt-brand-sub{ font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:var(--faint); margin-top:.15rem; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]{
  display:flex; align-items:center; gap:.75rem; padding:.7rem .95rem; margin:.12rem 0; border-radius:12px;
  border:1px solid transparent; color:var(--muted); font-weight:500; text-decoration:none;
  transition:background .2s ease, color .2s ease, border-color .2s ease, transform .2s ease;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *{ color:inherit !important; font-size:.96rem; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover{ background:rgba(56,189,248,.07); color:var(--text); transform:translateX(3px); }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]{
  background:linear-gradient(90deg,rgba(45,212,191,.16),rgba(56,189,248,.07));
  border-color:rgba(56,189,248,.5); color:var(--sky); box-shadow:0 0 0 1px rgba(56,189,248,.08) inset;
}
.bt-nav-soon{ display:flex; align-items:center; gap:.75rem; padding:.7rem .95rem; margin:.12rem 0; color:#4B5D7B; font-size:.96rem; font-weight:500; }
.bt-nav-soon i{ width:8px; height:8px; border-radius:50%; background:#2A3857; margin:0 .45rem 0 .3rem; }
.bt-nav-soon em{ font-style:normal; margin-left:auto; font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; border:1px solid #26365A; border-radius:999px; padding:.1rem .5rem; color:#4B5D7B; }
.bt-side-foot{ position:fixed; bottom:14px; left:0; width:300px; padding:0 1.6rem; color:var(--faint); font-size:.82rem; line-height:1.5; }
.bt-side-foot a{ display:inline-flex; align-items:center; gap:.55rem; margin-top:.7rem; color:var(--muted); text-decoration:none; font-size:.88rem; }
.bt-side-foot a:hover{ color:var(--sky); }

/* top bar */
.bt-topbar{ position:fixed; top:0; left:300px; right:0; height:64px; display:flex; align-items:center; justify-content:space-between;
  padding:0 2.4rem; background:rgba(11,18,36,.84); backdrop-filter:blur(12px); border-bottom:1px solid var(--soft); z-index:999980; }
.bt-topbar-title{ font-weight:600; font-size:1rem; color:var(--text); }
.bt-pills{ display:flex; gap:.7rem; align-items:center; }
.bt-pill{ font-size:.78rem; font-weight:600; padding:.42rem .95rem; border-radius:999px; border:1px solid var(--soft); color:var(--muted); white-space:nowrap; }
.bt-pill.teal{ color:#5EEAD4; background:rgba(45,212,191,.1); border-color:rgba(45,212,191,.45); }
.bt-pill.mono{ font-family:var(--mono); font-weight:400; font-size:.74rem; }

/* hero */
.bt-hero{ position:relative; margin:.6rem 0 2rem 0; animation:btUp .6s ease both; }
.bt-eyebrow{ font-size:.74rem; letter-spacing:.2em; text-transform:uppercase; font-weight:600; color:#A9B8CE; margin-bottom:.9rem; }
.bt-h1{ font-size:clamp(2rem,3.6vw,3.15rem); font-weight:800; letter-spacing:-.035em; line-height:1.06; margin:0 0 1.1rem 0; color:#fff; max-width:62rem; }
.bt-lead{ color:var(--muted); font-size:1.1rem; line-height:1.6; max-width:56rem; margin:0; }
.bt-helix{ position:absolute; right:0; top:-.4rem; width:240px; height:110px; opacity:.55; pointer-events:none; }
.bt-helix .rung{ stroke:#38BDF8; stroke-width:1.4; opacity:.35; animation:btPulse 2.8s ease-in-out infinite; }
.bt-helix .strand-a{ stroke:#2DD4BF; fill:none; stroke-width:2.4; stroke-linecap:round; }
.bt-helix .strand-b{ stroke:#38BDF8; fill:none; stroke-width:2.4; stroke-linecap:round; }

/* KPI cards */
.bt-kpis{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:1rem; margin-bottom:1.4rem; }
.bt-kpi{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:1.25rem 1.3rem 1.2rem; min-height:158px;
  animation:btUp .55s ease both; animation-delay:calc(var(--i) * 70ms); transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.bt-kpi:hover{ transform:translateY(-3px); border-color:rgba(56,189,248,.45); box-shadow:0 14px 32px rgba(0,0,0,.35); }
.bt-kpi-label{ display:flex; justify-content:space-between; align-items:flex-start; gap:.5rem; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-weight:600; color:#B6C3D8; }
.bt-info{ flex:none; width:17px; height:17px; border-radius:50%; border:1.5px solid #6B7C99; color:#6B7C99; font-size:.62rem; font-weight:700;
  display:grid; place-items:center; cursor:help; letter-spacing:0; text-transform:none; transition:color .2s, border-color .2s; }
.bt-info:hover{ color:var(--sky); border-color:var(--sky); }
.bt-kpi-value{ font-size:2.45rem; font-weight:700; letter-spacing:-.03em; color:var(--c); margin:.55rem 0 .25rem; line-height:1.1; }
.bt-kpi-sub{ color:var(--muted); font-size:.88rem; line-height:1.4; }

/* cards (keyed containers: st.container(key="card-...")) */
[class*="st-key-card-"]{ background:var(--card); border:1px solid var(--border); border-radius:20px; padding:1.6rem 1.7rem 1.5rem;
  animation:btUp .6s ease both; transition:border-color .25s ease, box-shadow .25s ease; }
[class*="st-key-card-"]:hover{ border-color:rgba(56,189,248,.28); box-shadow:0 16px 36px rgba(0,0,0,.28); }
.bt-card-title{ font-size:1.42rem; font-weight:650; letter-spacing:-.025em; color:#fff; margin:0 0 .35rem 0; }
.bt-card-sub{ color:var(--muted); font-size:.92rem; line-height:1.5; margin:0 0 .9rem 0; }
.bt-badge{ display:inline-block; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; color:#A5B4FC;
  background:rgba(99,102,241,.14); border:1px solid rgba(129,140,248,.38); border-radius:999px; padding:.3rem .8rem; margin-bottom:1rem; }
.bt-stack{ display:flex; gap:3px; height:14px; border-radius:999px; overflow:hidden; margin:.2rem 0 1.4rem; }
.bt-stack span{ display:block; height:100%; transition:filter .2s ease, flex-grow .6s ease; }
.bt-stack span:hover{ filter:brightness(1.25); }
.bt-legend{ display:grid; grid-template-columns:1fr 1fr; column-gap:2.4rem; }
.bt-leg{ display:flex; align-items:center; gap:.7rem; padding:.78rem 0; border-bottom:1px solid var(--soft); font-size:1rem; }
.bt-leg i{ width:10px; height:10px; border-radius:50%; flex:none; }
.bt-leg b{ margin-left:auto; font-weight:500; color:#C9D5E6; font-variant-numeric:tabular-nums; }
.bt-kv{ display:flex; justify-content:space-between; align-items:baseline; gap:1.2rem; padding:.82rem 0; border-bottom:1px solid var(--soft); font-size:.98rem; }
.bt-kv:last-child{ border-bottom:none; }
.bt-kv span{ color:var(--muted); }
.bt-kv b{ font-weight:500; color:#E7EEF8; text-align:right; font-variant-numeric:tabular-nums; }
.bt-kv b.mono{ font-family:var(--mono); font-size:.76rem; font-weight:400; word-break:break-all; max-width:62%; line-height:1.5; }
.bt-kv b.good{ color:#34D399; font-weight:600; }
.bt-note{ display:flex; gap:.8rem; align-items:flex-start; background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.25);
  border-radius:16px; padding:1rem 1.2rem; color:#C9D5E6; font-size:.95rem; line-height:1.55; margin-top:1.4rem; }
.bt-note b{ color:#fff; }

/* links, buttons, tooltips */
[data-testid="stMain"] [data-testid="stPageLink-NavLink"]{ background:transparent; border:none; color:var(--sky); font-weight:500; justify-content:flex-end; padding:0; }
[data-testid="stMain"] [data-testid="stPageLink-NavLink"]:hover{ color:#7DD3FC; }
[data-testid="stMain"] [data-testid="stPageLink-NavLink"] *{ color:inherit !important; }
.bt-soon-link{ text-align:right; color:#4B5D7B; font-size:.95rem; }
[data-testid="stDownloadButton"] button, [class*="st-key-card-"] button{
  border-radius:999px; background:rgba(56,189,248,.07); border:1px solid rgba(56,189,248,.38); color:#7DD3FC;
  font-weight:600; font-size:.84rem; padding:.4rem 1.05rem; transition:transform .2s ease, background .2s ease, box-shadow .2s ease; }
[data-testid="stDownloadButton"] button:hover, [class*="st-key-card-"] button:hover{
  background:rgba(56,189,248,.16); border-color:rgba(56,189,248,.7); transform:translateY(-2px); box-shadow:0 8px 20px rgba(56,189,248,.18); color:#BAE6FD; }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:16px; background:var(--card); }
[data-testid="stSpinner"] *{ color:var(--teal) !important; }

/* animation */
@keyframes btUp{ from{ opacity:0; transform:translateY(10px); } to{ opacity:1; transform:none; } }
@keyframes btFade{ from{ opacity:0; } to{ opacity:1; } }
@keyframes btPulse{ 0%,100%{ opacity:.18; } 50%{ opacity:.75; } }

/* responsive */
@media (max-width:1250px){ .bt-kpis{ grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width:992px){ .bt-topbar{ left:0; padding-left:4rem; } .bt-helix{ display:none; } .bt-legend{ grid-template-columns:1fr; } }
@media (max-width:640px){ .bt-kpis{ grid-template-columns:repeat(2,minmax(0,1fr)); } .bt-pill.mono{ display:none; } }
@media (prefers-reduced-motion:reduce){ *{ animation:none !important; transition:none !important; } }
</style>
"""

_LOGO_SVG = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#06202B" stroke-width="2" stroke-linecap="round">'
             '<path d="M7 3c0 4 10 4 10 9s-10 5-10 9"/><path d="M17 3c0 4-10 4-10 9s10 5 10 9"/>'
             '<path d="M8.6 7.5h6.8M8.6 16.5h6.8" stroke-width="1.6"/></svg>')
_GITHUB_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
               'stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 '
               '2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 '
               '5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/></svg>')


def _compact(markup):
    """Collapse HTML to one line: Markdown treats blank lines and 4-space indents as code."""
    return re.sub(r">\s+<", "><", " ".join(markup.split()))


def html_block(markup):
    """Render an HTML snippet safely through Streamlit's Markdown renderer."""
    st.markdown(_compact(markup), unsafe_allow_html=True)


def apply_style():
    """Inject the global CSS. Call once per page run (the entry file does it)."""
    st.markdown(_compact(_CSS), unsafe_allow_html=True)


# ------------------------------------------------------------------- components --------
def helix_svg():
    """A small decorative DNA helix: two sine strands with pulsing rungs (pure SVG + CSS)."""
    width, height, mid, amp, points = 240, 110, 55, 34, 48
    xs = [i * width / (points - 1) for i in range(points)]
    a = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{mid + amp * math.sin(x / 22):.1f}" for i, x in enumerate(xs))
    b = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{mid - amp * math.sin(x / 22):.1f}" for i, x in enumerate(xs))
    rungs = "".join(
        f'<line class="rung" x1="{x:.1f}" y1="{mid + amp * math.sin(x / 22):.1f}" x2="{x:.1f}" '
        f'y2="{mid - amp * math.sin(x / 22):.1f}" style="animation-delay:{i * 0.22:.2f}s"/>'
        for i, x in enumerate(range(6, width, 12)))
    return (f'<svg class="bt-helix" viewBox="0 0 {width} {height}" aria-hidden="true">{rungs}'
            f'<path class="strand-a" d="{a}"/><path class="strand-b" d="{b}"/></svg>')


def brand_html():
    return (f'<div class="bt-brand"><div class="bt-logo">{_LOGO_SVG}</div><div>'
            '<div class="bt-brand-name">BioTaxonAI</div><div class="bt-brand-sub">eDNA · 18S rRNA</div></div></div>')


def sidebar_footer_html():
    return ('<div class="bt-side-foot">Unsupervised clustering — reference taxonomy is used only after clustering, '
            f'never as input.<br><a href="{REPO_URL}" target="_blank" rel="noopener">{_GITHUB_SVG}Source repository</a></div>')


def topbar(title, dataset_label, fingerprint):
    """Fixed top bar: page title on the left, dataset and fingerprint pills on the right."""
    html_block(f'<div class="bt-topbar"><div class="bt-topbar-title">{html.escape(title)}</div><div class="bt-pills">'
               f'<span class="bt-pill teal">{html.escape(dataset_label)}</span>'
               f'<span class="bt-pill mono" title="SHA-256 over the hashes of both raw data files">'
               f'sha256 {fingerprint[:10]}…</span></div></div>')


def hero(eyebrow, title, lead):
    html_block(f'<div class="bt-hero">{helix_svg()}<div class="bt-eyebrow">{html.escape(eyebrow)}</div>'
               f'<h1 class="bt-h1">{html.escape(title)}</h1><p class="bt-lead">{html.escape(lead)}</p></div>')


def kpi_row(items):
    """items: list of (label, value, subtext, colour, tooltip-or-None)."""
    cards = []
    for i, (label, value, sub, color, tip) in enumerate(items):
        info = f'<span class="bt-info" title="{html.escape(tip, quote=True)}">i</span>' if tip else ""
        cards.append(f'<div class="bt-kpi" style="--i:{i};--c:{color}"><div class="bt-kpi-label"><span>{html.escape(label)}</span>{info}</div>'
                     f'<div class="bt-kpi-value">{html.escape(str(value))}</div><div class="bt-kpi-sub">{html.escape(sub)}</div></div>')
    html_block(f'<div class="bt-kpis">{"".join(cards)}</div>')


def card_header(title, subtitle=None, badge=None):
    parts = [f'<div class="bt-card-title">{html.escape(title)}</div>']
    if subtitle:
        parts.append(f'<div class="bt-card-sub">{html.escape(subtitle)}</div>')
    if badge:
        parts.append(f'<span class="bt-badge">{html.escape(badge)}</span>')
    html_block("".join(parts))


def group_stack(counts):
    """Stacked proportion bar plus a two-column legend. counts: {group: n_asvs}."""
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    bar = "".join(f'<span title="{html.escape(g)}: {n}" style="flex:{n};background:{GROUP_COLORS.get(g, "#94A3B8")}"></span>'
                  for g, n in ordered)
    legend = "".join(f'<div class="bt-leg"><i style="background:{GROUP_COLORS.get(g, "#94A3B8")}"></i>'
                     f'<span>{html.escape(g)}</span><b>{n:,}</b></div>' for g, n in ordered)
    html_block(f'<div class="bt-stack" aria-label="{total} ASVs by group">{bar}</div><div class="bt-legend">{legend}</div>')


def key_values(rows):
    """rows: list of (label, value, kind) where kind is '', 'mono' or 'good'."""
    html_block("".join(f'<div class="bt-kv"><span>{html.escape(str(k))}</span>'
                       f'<b class="{kind}">{html.escape(str(v))}</b></div>' for k, v, kind in rows))


def note(markup_text):
    html_block(f'<div class="bt-note"><span>ℹ️</span><div>{markup_text}</div></div>')


def style_figure(fig, height=340):
    """Dark theme for Plotly figures: transparent background, dotted grid, muted ticks."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=MUTED, size=12), height=height,
        margin=dict(l=8, r=8, t=12, b=8), showlegend=False,
        hoverlabel=dict(bgcolor="#0B1224", bordercolor="#2A3857", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False, linecolor="rgba(148,163,184,.25)", tickfont=dict(color=FAINT))
    fig.update_yaxes(gridcolor="rgba(148,163,184,.14)", griddash="dot", zeroline=False, tickfont=dict(color=FAINT))
    return fig