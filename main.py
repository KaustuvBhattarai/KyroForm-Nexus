import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import pandas as pd
import requests
import networkx as nx
import plotly.graph_objects as go
import base64
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import io
import matplotlib

try:
    import streamlit_agraph as sag

    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False
    sag = None

try:
    import py3Dmol

    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    py3Dmol = None

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    FPDF = None

from src.kyroform import (
    get_inference_engine,
    get_state_manager,
    fetch_uniprot,
    fetch_string_neighbors,
    fetch_alphafold_structure,
    parse_uniprot_annotations,
    format_protein_info,
    highlight_sequence,
    create_sequence_map_html,
    show_toast,
    show_loading_spinner,
    format_confidence_gauge,
    get_all_diseases,
    get_disease_genes,
    fetch_open_targets,
    fetch_pubmed_count,
    check_pathway_involvement,
    calculate_kyro_score,
    generate_contact_map_heatmap,
    compute_saliency,
    AUTOIMMUNE_PATHWAYS,
    save_session_to_kyro,
    load_session_from_kyro,
)

BASE_DIR = Path(__file__).resolve().parent
logo_src = BASE_DIR / "assets" / "logo.png"
logo_circle = BASE_DIR / "assets" / "logo_circle.png"
page_icon = None

if logo_src.exists():
    try:
        im = Image.open(logo_src).convert("RGBA")
        size = min(im.size)
        im = ImageOps.fit(im, (size, size), centering=(0.5, 0.5))
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        im.putalpha(mask)
        im.save(logo_circle)
        page_icon = str(logo_circle)
    except Exception:
        page_icon = None


st.set_page_config(
    page_title="Kyroform AI | Professional Research Suite",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


LINEAR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --bg-deep: #F7F8FA;
    --bg-base: #FFFFFF;
    --bg-elevated: #FFFFFF;
    --bg-surface: rgba(0,0,0,0.04);
    --bg-surface-hover: rgba(0,0,0,0.06);

    --text-primary: #0F172A;
    --text-muted: #6B7280;
    --text-subtle: rgba(0,0,0,0.6);

    --accent: #5E6AD2;
    --accent-bright: #4F5BD5;
    --accent-glow: rgba(94,106,210,0.2);

    --border-default: rgba(0,0,0,0.08);
    --border-hover: rgba(0,0,0,0.12);
    --border-accent: rgba(94,106,210,0.35);

    --success: #059669;
    --warning: #D97706;
    --danger: #DC2626;
}

* {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.disease-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(0,0,0,0.04);
    color: var(--text-muted);
    border: 1px solid var(--border-default);
    margin: 3px;
}
.disease-pill.active {
    background: rgba(94,106,210,0.2);
    color: var(--accent-bright);
    border-color: var(--border-accent);
}

.stApp {
    background: #FFFFFF;
    color: var(--text-primary);
}

/* Animated ambient blobs - disabled for light mode */
.stApp::before, .stApp::after {
    display: none;
}

@keyframes float {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(1deg); }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8F9FA 0%, #F0F1F2 100%);
    border-right: 1px solid var(--border-default);
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

/* Glass card component */
.glass-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    border: 1px solid var(--border-default);
    border-radius: 16px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 16px rgba(0,0,0,0.1), 0 0 0 1px rgba(94,106,210,0.1);
    transform: translateY(-2px);
}

/* Header styling */
.hero-header {
    text-align: center;
    padding: 48px 0 32px 0;
    position: relative;
    z-index: 1;
}

.hero-header h1 {
    font-size: 56px;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(180deg, #1A1A1A 0%, #4B5563 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.hero-header .accent-text {
    background: linear-gradient(90deg, #5E6AD2, #818cf8, #5E6AD2);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-header .subtitle {
    color: var(--text-muted);
    font-size: 18px;
    margin-top: 12px;
    font-weight: 400;
}

/* Navigation buttons */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 14px 16px;
    border-radius: 12px;
    color: var(--text-muted);
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
}

.nav-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
}

.nav-btn.active {
    background: linear-gradient(135deg, rgba(94,106,210,0.2) 0%, rgba(94,106,210,0.1) 100%);
    border-color: var(--border-accent);
    color: var(--text-primary);
    box-shadow: 0 0 20px rgba(94,106,210,0.15);
}

.nav-btn .icon {
    font-size: 18px;
}

/* Sidebar navigation buttons - light mode */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(0,0,0,0.02) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #6B7280 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(94,106,210,0.1) !important;
    border: 1px solid rgba(94,106,210,0.2) !important;
    color: #1A1A1A !important;
    transform: translateX(4px) !important;
}

[data-testid="stSidebar"] .stButton > button:focus {
    background: rgba(94,106,210,0.15) !important;
    border: 1px solid rgba(94,106,210,0.3) !important;
    box-shadow: 0 0 0 2px rgba(94,106,210,0.15) !important;
}

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #5E6AD2 0%, #6872D9 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.01em;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 0 1px rgba(94,106,210,0.5), 0 4px 12px rgba(94,106,210,0.3), inset 0 1px 0 0 rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 0 1px rgba(94,106,210,0.6), 0 8px 24px rgba(94,106,210,0.4), inset 0 1px 0 0 rgba(0,0,0,0.25);
    background: linear-gradient(135deg, #6872D9 0%, #7c83e0 100%);
}

.stButton > button:active {
    transform: scale(0.98);
}

/* Secondary buttons */
.secondary-btn {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}

.secondary-btn:hover {
    background: var(--bg-surface-hover) !important;
    border-color: var(--border-hover) !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    border: 1px solid var(--border-default);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.metric-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
}

.metric-card .value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
}

.metric-card .label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-weight: 500;
}

/* Confidence gauge */
.confidence-gauge {
    height: 6px;
    background: rgba(0,0,0,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 12px 0;
}

.confidence-gauge .fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 10px currentColor;
}

/* Select boxes and inputs */
.stSelectbox > div > div > div,
.stTextInput > div > div > input,
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
    transition: all 0.2s ease;
    min-height: 48px !important;
}

.stSelectbox > div > div > div:hover,
.stTextInput > div > div > input:hover {
    border-color: rgba(0,0,0,0.15) !important;
}

.stSelectbox > div > div > div:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(94,106,210,0.15) !important;
}

/* Selectbox dropdown menu */
[data-baseweb="popover"] ul,
div[role="listbox"],
[class*="dropdown"],
[class*="select"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    padding: 8px !important;
    max-height: 400px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
}

[data-baseweb="popover"] li,
div[role="listbox"] > div,
[class*="dropdown"] > div,
[class*="select"] > div {
    padding: 12px 16px !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    min-height: 40px !important;
    background: #FFFFFF !important;
}

[data-baseweb="popover"] li:hover,
div[role="listbox"] > div:hover {
    background: rgba(94,106,210,0.1) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px 10px 0 0;
    padding: 12px 20px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 14px;
    border: none;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,0,0,0.05) !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent);
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #5E6AD2, #818cf8) !important;
    border-radius: 4px;
    box-shadow: 0 0 10px rgba(94,106,210,0.5);
}

/* Section headers */
.section-title {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 24px;
}

/* Subsection headers */
.subsection-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
}

/* Info panels */
.info-panel {
    background: rgba(0,0,0,0.03);
    border: 1px solid var(--border-default);
    border-radius: 14px;
    padding: 20px;
}

.info-panel .title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Badges and pills */
.data-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(0,0,0,0.06);
    color: var(--text-muted);
    border: 1px solid var(--border-default);
    margin: 3px;
}

.data-pill.active {
    background: rgba(94,106,210,0.2);
    color: var(--accent-bright);
    border-color: var(--border-accent);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(0,0,0,0.15);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0,0,0,0.25);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.02);
    border: 1px solid var(--border-default);
    border-radius: 14px;
    overflow: hidden;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(0,0,0,0.03) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 16px !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(0,0,0,0.05) !important;
}

/* Error/Warning/Success messages */
.stAlert {
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent);
    margin: 32px 0;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.05);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--accent-bright);
}

/* Hide Streamlit top menu bar */
[data-testid="stHeader"] {
    display: none !important;
}

#MainMenu {
    display: none !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}

/* Hide the hamburger menu */
.stApp > header > div:first-child {
    display: none !important;
}

/* Also hide the footer */
footer {
    display: none !important;
}

[data-testid="stFooter"] {
    display: none !important;
}
</style>
"""

st.markdown(LINEAR_CSS, unsafe_allow_html=True)

_engine = get_inference_engine()
_state = get_state_manager()


def render_header():
    if page_icon and Path(page_icon).exists():
        try:
            with open(page_icon, "rb") as _f:
                img_b64 = base64.b64encode(_f.read()).decode("ascii")
            header_html = f"""
            <div class="hero-header">
                <img src='data:image/png;base64,{img_b64}' style='width:64px;border-radius:16px;margin-bottom:16px;box-shadow: 0 0 30px rgba(94,106,210,0.3);'/>
                <h1><span class="accent-text">Kyroform</span> AI</h1>
                <div class="subtitle">Professional Research Suite — Gut-Host Interactome Discovery Engine</div>
            </div>
            """
        except Exception:
            header_html = """
            <div class="hero-header">
                <h1><span class="accent-text">Kyroform</span> AI</h1>
                <div class="subtitle">Professional Research Suite — Gut-Host Interactome Discovery Engine</div>
            </div>
            """
    else:
        header_html = """
        <div class="hero-header">
            <h1><span class="accent-text">Kyroform</span> AI</h1>
            <div class="subtitle">Professional Research Suite — Gut-Host Interactome Discovery Engine</div>
        </div>
        """
    st.markdown(header_html, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
        <div style="padding: 8px 0 24px 0;">
            <div style="font-size: 11px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;">Navigation</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        pages = [
            "Global Explorer",
            "Single Predictor",
            "Batch Analysis",
            "Comparative Analysis",
            "3D Structure Viewer",
            "Publication Pack",
        ]

        current_view = _state.get_view()

        for page_name in pages:
            is_active = page_name == current_view

            if st.button(page_name, use_container_width=True, key=f"nav_{page_name}"):
                _state.set_view(page_name)
                st.rerun()

            if is_active:
                st.markdown(
                    f"""
                <div style="font-size: 11px; color: #5E6AD2; padding-left: 16px; margin-top: -8px; margin-bottom: 12px;">● Active</div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """<hr style="margin: 24px 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent);">""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### Settings")
        _state.update_settings(
            "calibration_samples",
            st.slider(
                "Calibration samples",
                100,
                2000,
                _state.get_setting("calibration_samples", 400),
                100,
            ),
        )
        _state.update_settings(
            "neg_controls",
            st.slider(
                "Negative controls", 10, 200, _state.get_setting("neg_controls", 40), 10
            ),
        )
        _state.update_settings(
            "edge_threshold",
            st.slider(
                "Edge threshold",
                0.0,
                1.0,
                _state.get_setting("edge_threshold", 0.3),
                0.01,
            ),
        )

        st.markdown(
            """<hr style="margin: 24px 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent);">""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style="font-size: 11px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;">Session</div>
        """,
            unsafe_allow_html=True,
        )

        stats = _state.export_state()
        st.markdown(
            f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div class="glass-card" style="padding: 16px; text-align: center;">
                <div style="font-size: 20px; font-weight: 600; color: #1A1A1A;">{stats["prediction_count"]}</div>
                <div style="font-size: 10px; color: #6B7280; text-transform: uppercase;">Predictions</div>
            </div>
            <div class="glass-card" style="padding: 16px; text-align: center;">
                <div style="font-size: 20px; font-weight: 600; color: #1A1A1A;">{stats["batch_count"]}</div>
                <div style="font-size: 10px; color: #6B7280; text-transform: uppercase;">Batch Runs</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Session", use_container_width=True):
            _state.clear_session()
            st.rerun()


def render_confidence_indicator(probability: float):
    conf = format_confidence_gauge(probability)

    color_map = {
        "High": "#10b981",
        "Moderate": "#f59e0b",
        "Low": "#3b82f6",
        "Very Low": "#6b7280",
    }
    color = color_map.get(conf["label"], "#5E6AD2")

    st.markdown(
        f"""
    <div class="glass-card" style="padding: 24px; margin-bottom: 20px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <span style="font-size:13px;color:#6B7280; text-transform: uppercase; letter-spacing: 0.05em;">Prediction Confidence</span>
            <span style="font-size:32px;font-weight:700;color:{color}; text-shadow: 0 0 20px {color}40;">{probability:.3f}</span>
        </div>
        <div class="confidence-gauge">
            <div class="fill" style="width:{probability * 100}%;background:{color}"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:12px;">
            <span style="color:#6B7280;font-size:13px;">{conf["description"]}</span>
            <span style="color:#1A1A1A;font-size:13px;font-weight: 500;">{conf["percentile"]}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict):
    cols = st.columns(len(metrics))
    for idx, (label, value) in enumerate(metrics.items()):
        with cols[idx]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_protein_panel(protein_id: str, protein_type: str):
    with st.expander(
        f"{protein_type.title()} Protein Details",
        expanded=True,
    ):
        info = fetch_uniprot(protein_id)
        if info:
            details = format_protein_info(info)
            ann = parse_uniprot_annotations(info)

            st.markdown(f"**{details.get('name', 'Unknown')}**")
            st.write(f"**Gene:** {details.get('gene', 'N/A')}")
            st.write(f"**Organism:** {details.get('organism', 'N/A')}")
            st.write(f"**Length:** {details.get('length', 0)} aa")

            if ann["go"]:
                st.markdown("**GO Terms:**")
                pills = " ".join(
                    [f"<span class='disease-pill'>{g}</span>" for g in ann["go"][:5]]
                )
                st.markdown(pills, unsafe_allow_html=True)

            if ann["diseases"]:
                st.markdown("**Associated Diseases:**")
                pills = " ".join(
                    [
                        f"<span class='disease-pill active'>{d}</span>"
                        for d in ann["diseases"][:3]
                    ]
                )
                st.markdown(pills, unsafe_allow_html=True)
        else:
            st.warning("No UniProt data available")


def render_sequence_viewer(protein_id: str, use_expander: bool = True):
    info = fetch_uniprot(protein_id)
    if info and info.get("sequence", {}).get("value"):
        seq = info["sequence"]["value"]
        st.markdown("### Sequence Map")
        st.markdown(highlight_sequence(seq), unsafe_allow_html=True)

        if use_expander:
            with st.expander("Full Sequence"):
                st.code(seq)
        else:
            st.markdown("**Full Sequence:**")
            st.code(seq)
    else:
        st.info("Sequence not available")


def build_interaction_graph(human_id: str, bacterial_id: str, probability: float):
    threshold = _state.get_setting("edge_threshold", 0.3)

    human_neighbors = fetch_string_neighbors(human_id, species=9606, limit=8) or []
    bact_neighbors = fetch_string_neighbors(bacterial_id, species=511145, limit=8) or []

    G = nx.Graph()
    G.add_node(human_id, label=human_id, color="#ff6b6b", type="human")
    G.add_node(bacterial_id, label=bacterial_id, color="#4ecdc4", type="bacterial")

    for n in human_neighbors:
        partner = n.get("preferredName") or n.get("stringId_B")
        score = float(n.get("score", 0))
        if partner:
            G.add_node(partner, label=partner, color="#ffcccc", type="human_neighbor")
            if score >= threshold:
                G.add_edge(human_id, partner, weight=score)

    for n in bact_neighbors:
        partner = n.get("preferredName") or n.get("stringId_B")
        score = float(n.get("score", 0))
        if partner:
            G.add_node(partner, label=partner, color="#ccffdd", type="bact_neighbor")
            if score >= threshold:
                G.add_edge(bacterial_id, partner, weight=score)

    if probability >= threshold:
        G.add_edge(human_id, bacterial_id, weight=probability, is_prediction=True)

    return G, human_neighbors, bact_neighbors


def render_interactive_graph(
    G: nx.Graph, human_id: str, bacterial_id: str, probability: float
):
    if AGRAPH_AVAILABLE and sag is not None:
        try:
            nodes = []
            edges = []

            for node, data in G.nodes(data=True):
                size = 25 + G.degree(node) * 8
                color = data.get("color", "#888")

                nodes.append(
                    sag.Node(id=node, label=node, size=size, color=color, shape="dot")
                )

            for a, b, data in G.edges(data=True):
                width = 1 + data.get("weight", 0) * 4
                color = "#3b82f6" if data.get("is_prediction", False) else "#64748b"

                edges.append(
                    sag.Edge(
                        source=a, target=b, width=width, color=color, type="straight"
                    )
                )

            config = sag.Config(
                width=800,
                height=500,
                directed=False,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightAlpha=0.5,
            )

            return sag.Graph(nodes=nodes, edges=edges, config=config)

        except Exception:
            pass

    pos = nx.spring_layout(G, seed=42)

    edge_traces = []
    for a, b, d in G.edges(data=True):
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        score = float(d.get("weight", 0))

        if score >= 0.7:
            color = "#ef4444"
            width = 3 + score * 3
        elif score >= 0.4:
            color = "#f59e0b"
            width = 2 + score * 2
        else:
            color = "#64748b"
            width = 1 + score * 2

        is_pred = d.get("is_prediction", False)
        if is_pred:
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(width=width + 6, color="rgba(59,130,246,0.2)"),
                    hoverinfo="skip",
                    mode="lines",
                    showlegend=False,
                )
            )

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=f"{a} ↔ {b}: {score:.3f}",
                mode="lines",
            )
        )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{d.get('label', n)}<br>Degree: {G.degree(n)}")
        node_color.append(d.get("color", "#888"))
        node_size.append(15 + G.degree(n) * 6)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(color=node_color, size=node_size, line_width=1),
        text=[n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8, color="#f1f5f9"),
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Interaction Network: {human_id} ↔ {bacterial_id}",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#f1f5f9"),
            margin=dict(b=20, l=5, r=5, t=40),
            height=500,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="glass-card" style="padding: 16px; margin-top: 16px;">
        <div style="font-size: 12px; color: #6B7280; font-weight: 500; margin-bottom: 8px;">GRAPH INTERPRETATION</div>
        <ul style="font-size: 12px; color: #1A1A1A; margin: 0; padding-left: 16px; line-height: 1.8;">
            <li><span style="color: #ff6b6b;">● Red nodes</span> = Human protein and its STRING neighbors</li>
            <li><span style="color: #4ecdc4;">● Teal nodes</span> = Bacterial protein and its neighbors</li>
            <li><span style="color: #f39d0d;">● Orange edge</span> = Predicted interaction (thickness = confidence)</li>
            <li><span style="color: #ef4444;">● Red edges</span> = STRING database interactions</li>
            <li>Node size reflects degree (connections) in the network</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_3d_viewer(protein_id: str):
    if not protein_id:
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"""
        <div class="glass-card" style="padding: 24px; border-radius: 16px; background: #FFFFFF; border: 1px solid rgba(0,0,0,0.06);">
            <h3 style="margin: 0 0 16px 0; color: #1A1A1A; font-size: 18px; font-weight: 600;">3D Structure: {protein_id}</h3>
        """,
            unsafe_allow_html=True,
        )

        pdb_url = fetch_alphafold_structure(protein_id)

        if pdb_url:
            st.markdown(
                f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></span>
                <span style="color: #10b981; font-size: 14px;">Structure available</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if py3Dmol is not None:
                try:
                    import requests as _req

                    pdb_content = _req.get(pdb_url, timeout=10).text
                    view = py3Dmol.view(width=450, height=400)
                    view.addModel(pdb_content, "pdb")
                    view.setStyle({}, {"cartoon": {"color": "spectrum"}})
                    view.zoomTo()
                    html_content = view._make_html()
                    st.components.v1.html(html_content, height=420)
                except Exception as e:
                    st.warning(f"3D render failed: {e}")
                    st.markdown(
                        f"""
                    <a href="{pdb_url}" target="_blank"
                       style="display:inline-block;padding:10px 18px;background:rgba(94,106,210,0.2);
                              border:1px solid rgba(94,106,210,0.4);border-radius:8px;
                              color:#818cf8;text-decoration:none;font-size:13px;">
                        View on AlphaFold DB
                    </a>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                af_page = f"https://alphafold.ebi.ac.uk/entry/{protein_id}"
                st.markdown(
                    f"""
                <div style="background:#0F0F12;padding:16px;border-radius:8px;margin-top:12px;">
                    <p style="color:#6B7280;font-size:13px;margin:0 0 12px 0;">
                        Interactive 3D viewer requires <code>py3Dmol</code>.
                    </p>
                    <a href="{af_page}" target="_blank"
                       style="display:inline-block;padding:10px 18px;
                              background:rgba(94,106,210,0.2);
                              border:1px solid rgba(94,106,210,0.4);
                              border-radius:8px;color:#818cf8;
                              text-decoration:none;font-size:13px;">
                        View {protein_id} on AlphaFold DB
                    </a>
                    <p style="color:#6b7280;font-size:11px;margin-top:12px;">
                        Install: <code>pip install py3Dmol</code>
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:8px;height:8px;background:#f59e0b;border-radius:50%;"></span>
                <span style="color:#f59e0b;font-size:14px;">No AlphaFold structure found for this ID</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
        <div class="glass-card" style="padding:20px;">
            <h4 style="margin:0 0 12px 0;color:#6B7280;font-size:12px;font-weight:500;
                       text-transform:uppercase;letter-spacing:0.05em;">Structure Info</h4>
            <p style="color:#1A1A1A;font-size:14px;margin:0;">
                AlphaFold DB provides AI-predicted protein structures with per-residue confidence (pLDDT) scores.
            </p>
            <p style="color:#6B7280;font-size:12px;margin-top:12px;">
                Blue = high confidence · Orange = low confidence
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_comparative_analysis():
    st.markdown("## Comparative Analysis Mode")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configuration")
        all_diseases = get_all_diseases()
        current = _state.get_disease_context()
        idx = all_diseases.index(current) if current in all_diseases else 0
        disease = st.selectbox(
            "Disease Context", all_diseases, index=idx, key="comp_disease"
        )
        if disease != _state.get_disease_context():
            _state.set_disease_context(disease)

        disease_genes = get_disease_genes(disease) if disease else []
        available_human = _engine.human_proteins
        available_bact = _engine.bacterial_proteins

        selected_bact = st.selectbox("Bacterial Protein", [""] + sorted(available_bact))

        st.markdown(f"**Disease-associated genes ({len(disease_genes)}):**")
        st.write(
            ", ".join(disease_genes[:10]) + ("..." if len(disease_genes) > 10 else "")
        )

        if st.button("Run Comparison", type="primary"):
            if not selected_bact:
                st.error("Please select a bacterial protein")
            else:
                results = []
                progress_bar = st.progress(0)

                available_genes = [g for g in disease_genes if g in available_human]

                if not available_genes:
                    st.warning(
                        "No disease genes found in available proteins. Using all human proteins."
                    )
                    available_genes = available_human[:20]

                for idx, human_gene in enumerate(available_genes):
                    try:
                        prob, z_h, z_b, _, _ = _engine.predict_interaction(
                            human_gene, selected_bact
                        )
                        results.append(
                            {
                                "human_gene": human_gene,
                                "bacterial_id": selected_bact,
                                "probability": prob,
                                "z_score": (prob - 0.5) / 0.2,
                            }
                        )
                    except:
                        results.append(
                            {
                                "human_gene": human_gene,
                                "bacterial_id": selected_bact,
                                "probability": 0,
                                "z_score": 0,
                            }
                        )
                    progress_bar.progress((idx + 1) / len(available_genes))

                results_df = pd.DataFrame(results).sort_values(
                    "probability", ascending=False
                )
                _state.store_batch_results(results_df.to_dict("records"))

                st.success(f"Analysis complete! {len(results)} interactions evaluated.")
                show_toast("Comparative analysis completed")

    with col2:
        batch_results = _state.get_batch_results()

        if batch_results:
            results_df = pd.DataFrame(batch_results)

            # Normalize column name — batch analysis uses 'human_id', comparative uses 'human_gene'
            if (
                "human_gene" not in results_df.columns
                and "human_id" in results_df.columns
            ):
                results_df = results_df.rename(columns={"human_id": "human_gene"})

            if "human_gene" not in results_df.columns:
                st.warning("Result format mismatch. Please re-run the comparison.")
            else:
                st.markdown("### Results")

                top_results = results_df.head(10)

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=top_results["human_gene"].tolist(),
                        y=top_results["probability"].tolist(),
                        marker=dict(
                            color=top_results["probability"].tolist(),
                            colorscale="RdYlGn",
                        ),
                        text=top_results["probability"].round(3).tolist(),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Interaction Probabilities by Disease Gene",
                    xaxis_title="Human Gene",
                    yaxis_title="Probability",
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#f1f5f9"),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    results_df.style.background_gradient(
                        subset=["probability"], cmap="RdYlGn"
                    ),
                    use_container_width=True,
                )

                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results", csv, "comparative_analysis.csv", "text/csv"
                )
        else:
            st.info("Run a comparison to see results here")


def render_global_explorer():
    st.markdown("## Global Explorer")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Interactome Overview")

        human_id = st.selectbox(
            "Select Human Protein",
            [""] + sorted(_engine.human_proteins),
            key="explorer_human",
        )

        bact_id = st.selectbox(
            "Select Bacterial Protein",
            [""] + sorted(_engine.bacterial_proteins),
            key="explorer_bact",
        )

        if human_id and bact_id:
            with st.spinner("Predicting interaction..."):
                try:
                    prob, z_h, z_b, emb_h, emb_b = _engine.predict_interaction(
                        human_id, bact_id
                    )

                    render_confidence_indicator(prob)

                    G, human_neighbors, bact_neighbors = build_interaction_graph(
                        human_id, bact_id, prob
                    )
                    render_interactive_graph(G, human_id, bact_id, prob)

                    render_metric_cards(
                        {
                            "Degree": len(G.nodes()),
                            "Edges": len(G.edges()),
                            "Human Neighbors": len(human_neighbors),
                            "Bact Neighbors": len(bact_neighbors),
                        }
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.info("Select proteins to explore their interaction")

    with col2:
        st.markdown("### Info Panel")

        if human_id:
            render_protein_panel(human_id, "human")

        if bact_id:
            render_protein_panel(bact_id, "bacterial")

        if human_id and bact_id:
            with st.expander("Sequence Viewer"):
                tab1, tab2 = st.tabs(["Human", "Bacterial"])
                with tab1:
                    render_sequence_viewer(human_id, use_expander=False)
                with tab2:
                    render_sequence_viewer(bact_id, use_expander=False)


def render_single_predictor():
    st.markdown("## Single Protein Predictor")
    col_main, col_side = st.columns([3, 1])

    with col_main:
        human_id = st.selectbox(
            "Human Protein (UniProt ID)",
            [""] + sorted(_engine.human_proteins),
            key="predictor_human",
        )
        bact_id = st.selectbox(
            "Bacterial Protein (UniProt ID)",
            [""] + sorted(_engine.bacterial_proteins),
            key="predictor_bact",
        )

        if st.button("Predict Interaction", type="primary", use_container_width=True):
            if not human_id or not bact_id:
                st.error("Please select both proteins")
            else:
                with st.spinner("Computing prediction..."):
                    try:
                        prob, z_h, z_b, emb_h, emb_b = _engine.predict_interaction(
                            human_id, bact_id
                        )
                        # Store everything needed for downstream tabs
                        _state.store_prediction(
                            {
                                "human_id": human_id,
                                "bacterial_id": bact_id,
                                "probability": prob,
                            }
                        )
                        _state.update_settings("last_emb_h", emb_h.tolist())
                        _state.update_settings("last_emb_b", emb_b.tolist())
                        _state.update_settings("last_z_h", z_h.tolist())
                        _state.update_settings("last_z_b", z_b.tolist())
                        _state.update_settings("last_human_id", human_id)
                        _state.update_settings("last_bact_id", bact_id)
                        show_toast(f"Prediction complete: {prob:.3f}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- Persistent results block (survives reruns) ---
        predictions = _state.get_predictions()
        last_human = _state.get_setting("last_human_id")
        last_bact = _state.get_setting("last_bact_id")
        raw_z_h = _state.get_setting("last_z_h")
        raw_z_b = _state.get_setting("last_z_b")
        raw_emb_h = _state.get_setting("last_emb_h")
        raw_emb_b = _state.get_setting("last_emb_b")

        if predictions and raw_z_h and last_human == human_id and last_bact == bact_id:
            last_pred = predictions[-1]
            prob = last_pred["probability"]
            z_h = np.array(raw_z_h)
            z_b = np.array(raw_z_b)
            emb_h = np.array(raw_emb_h)
            emb_b = np.array(raw_emb_b)

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "General Info",
                    "XAI / Interpretability",
                    "Structural Analysis",
                    "Literature & Pathways",
                ]
            )

            with tab1:
                render_confidence_indicator(prob)
                from sklearn.metrics.pairwise import cosine_similarity

                cos_orig = float(
                    cosine_similarity(emb_h.reshape(1, -1), emb_b.reshape(1, -1))[0, 0]
                )
                cos_z = float(
                    cosine_similarity(z_h.reshape(1, -1), z_b.reshape(1, -1))[0, 0]
                )
                kyro_score = calculate_kyro_score(prob, cos_orig, centrality=0.6)
                render_metric_cards(
                    {
                        "Kyro-Score": f"{kyro_score:.3f}",
                        "ESM Cosine": f"{cos_orig:.4f}",
                        "Latent Cosine": f"{cos_z:.4f}",
                        "Human Norm": f"{np.linalg.norm(emb_h):.1f}",
                        "Bact Norm": f"{np.linalg.norm(emb_b):.1f}",
                    }
                )

            with tab2:
                st.markdown("### Attention Heatmap (Latent Space)")
                contribs = _engine.compute_contributions(z_h, z_b)
                contrib_fig = go.Figure(
                    go.Bar(
                        x=[c["value"] for c in contribs],
                        y=[f"dim {c['feature']}" for c in contribs],
                        orientation="h",
                        marker=dict(color="#5E6AD2"),
                    )
                )
                contrib_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1A1A1A"),
                    height=300,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(contrib_fig, use_container_width=True)

                st.markdown("### Saliency Mapping (In-Silico Mutagenesis)")
                position = st.number_input(
                    "Amino Acid Position",
                    min_value=0,
                    max_value=1279,
                    value=0,
                    key="sal_pos",
                )
                mutation = st.selectbox(
                    "Mutation To",
                    [
                        "A",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "K",
                        "L",
                        "M",
                        "N",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "V",
                        "W",
                        "Y",
                    ],
                    key="sal_mut",
                )
                if st.button("Compute Saliency", key="btn_saliency"):
                    saliency = compute_saliency(emb_h, emb_b, position, mutation)
                    st.metric(
                        "Saliency Score",
                        f"{saliency:.4f}",
                        help="Effect of mutating this position on prediction",
                    )

            with tab3:
                st.markdown("### 3D Structure Viewer")
                render_3d_viewer(human_id)
                render_3d_viewer(bact_id)

                st.markdown("### Contact Map")
                contact_fig = generate_contact_map_heatmap(z_h, z_b)
                st.plotly_chart(contact_fig, use_container_width=True)

            with tab4:
                st.markdown("### Clinical Significance")
                gene = "N/A"
                pubmed_count = 0
                human_info = fetch_uniprot(human_id)
                if human_info:
                    gene = (
                        human_info.get("genes", [{}])[0]
                        .get("geneName", {})
                        .get("value", "N/A")
                    )
                    pubmed_count = fetch_pubmed_count(human_id)
                    st.markdown(
                        f"""
                    <div class="glass-card" style="padding: 20px; margin: 12px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <div style="font-size: 12px; color: #6B7280;">Gene Symbol</div>
                                <div style="font-size: 18px; font-weight: 600; color: #1A1A1A;">{gene}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 12px; color: #6B7280;">PubMed Evidence</div>
                                <div style="font-size: 18px; font-weight: 600; color: #5E6AD2;">{pubmed_count:,} papers</div>
                            </div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                pathways = check_pathway_involvement(gene) if gene != "N/A" else []
                if pathways:
                    st.success(f"Found involvement in: {', '.join(pathways)}")
                    for pw in pathways:
                        genes_in_pathway = AUTOIMMUNE_PATHWAYS.get(pw, [])
                        st.markdown(f"**{pw}:** {', '.join(genes_in_pathway[:5])}")
                else:
                    st.info("No major autoimmune pathway involvement detected")

                st.markdown("### Network Analysis")
                G, human_neighbors, bact_neighbors = build_interaction_graph(
                    human_id, bact_id, prob
                )
                render_interactive_graph(G, human_id, bact_id, prob)

                if human_neighbors:
                    st.markdown("**Top Human Neighbors (STRING)**")
                    st.table(
                        pd.DataFrame(
                            [
                                {
                                    "Partner": n.get("preferredName", "N/A"),
                                    "Score": float(n.get("score", 0)),
                                }
                                for n in human_neighbors[:5]
                            ]
                        )
                    )
                else:
                    st.markdown("**Top Human Neighbors (STRING)**")
                    st.table(pd.DataFrame([{"Partner": "N/A", "Score": "N/A"}]))

                if bact_neighbors:
                    st.markdown("**Top Bacterial Neighbors (STRING)**")
                    st.table(
                        pd.DataFrame(
                            [
                                {
                                    "Partner": n.get("preferredName", "N/A"),
                                    "Score": float(n.get("score", 0)),
                                }
                                for n in bact_neighbors[:5]
                            ]
                        )
                    )
                else:
                    st.markdown("**Top Bacterial Neighbors (STRING)**")
                    st.table(pd.DataFrame([{"Partner": "N/A", "Score": "N/A"}]))

    with col_side:
        st.markdown("### Quick Stats")
        st.metric("Human Proteins", len(_engine.human_proteins))
        st.metric("Bacterial Proteins", len(_engine.bacterial_proteins))
        st.metric("Total Predictions", len(_state.get_predictions()))
        predictions = _state.get_predictions()
        if predictions:
            probs = [p["probability"] for p in predictions]
            st.metric("Avg Probability", f"{np.mean(probs):.3f}")


def render_batch_analysis():
    st.markdown("## Batch Analysis")

    st.markdown(
        """
    <div class="glass-card" style="padding: 16px; margin-bottom: 20px;">
        Upload a CSV file with two columns: <code>human_id</code> and <code>bact_id</code>. 
        Each row represents one protein pair to predict.
    </div>
    """,
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame(
        {"human_id": ["P12345", "Q9Y6X5"], "bact_id": ["A0A001", "A0A002"]}
    )

    st.download_button(
        "Download Template CSV",
        template_df.to_csv(index=False).encode("utf-8"),
        "batch_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)

            if "human_id" not in batch_df.columns or "bact_id" not in batch_df.columns:
                st.error("CSV must have columns: human_id, bact_id")
            else:
                valid_mask = batch_df["human_id"].isin(
                    list(_engine.embeddings.keys())
                ) & batch_df["bact_id"].isin(list(_engine.embeddings.keys()))
                valid_df = batch_df[valid_mask].reset_index(drop=True)

                st.info(f"Processing {len(valid_df)} valid pairs...")
                progress_bar = st.progress(0)

                results = []
                pairs = list(zip(valid_df["human_id"], valid_df["bact_id"]))

                for idx, result in enumerate(_engine.batch_predict(pairs)):
                    prob_val = result.get("probability")
                    if prob_val is None:
                        prob_val = result.get("prob")
                    if prob_val is None:
                        prob_val = result.get("score")
                    if prob_val is None:
                        prob_val = 0.0

                    prob_float = float(prob_val)

                    conf_val = result.get("confidence")
                    if conf_val is None or str(conf_val).startswith("Error"):
                        conf_val = (
                            "High"
                            if prob_float > 0.7
                            else "Moderate"
                            if prob_float >= 0.5
                            else "Low"
                            if prob_float >= 0.3
                            else "Very Low"
                        )

                    results.append(
                        {
                            "human_id": result.get("human_id", pairs[idx][0]),
                            "bact_id": result.get("bact_id", pairs[idx][1]),
                            "probability": prob_float,
                            "confidence": conf_val,
                        }
                    )
                    progress_bar.progress((idx + 1) / len(pairs))

                if not results:
                    st.error(
                        "No valid predictions could be generated. Check your input data."
                    )
                    return

                results_df = pd.DataFrame(results)
                _state.store_batch_results(results_df.to_dict("records"))

                show_toast(f"Batch processing complete: {len(results)} predictions")

                st.markdown("### Summary")

                valid_probs = results_df["probability"].dropna()

                scols = st.columns(4)
                scols[0].metric("Total Pairs", len(results_df))
                scols[1].metric("High (>0.7)", int((valid_probs > 0.7).sum()))
                scols[2].metric(
                    "Moderate (0.5-0.7)",
                    int(((valid_probs >= 0.5) & (valid_probs <= 0.7)).sum()),
                )
                scols[3].metric("Avg Probability", f"{valid_probs.mean():.3f}")

                st.markdown("### Distribution")

                chart_cols = st.columns(2)

                with chart_cols[0]:
                    dist_fig = go.Figure()
                    dist_fig.add_trace(
                        go.Histogram(
                            x=valid_probs.tolist(),
                            nbinsx=20,
                            marker=dict(color="#3b82f6", opacity=0.8),
                        )
                    )
                    dist_fig.update_layout(
                        title="Score Distribution",
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font=dict(color="#f1f5f9"),
                        height=300,
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)

                with chart_cols[1]:
                    conf_counts = results_df["confidence"].value_counts()
                    pie_fig = go.Figure(
                        go.Pie(
                            labels=conf_counts.index,
                            values=conf_counts.values,
                            hole=0.4,
                            marker=dict(
                                colors=["#10b981", "#f59e0b", "#3b82f6", "#6b7280"]
                            ),
                        )
                    )
                    pie_fig.update_layout(
                        title="Confidence Breakdown",
                        paper_bgcolor="#0f172a",
                        font=dict(color="#f1f5f9"),
                        height=300,
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                st.markdown("### Results")

                st.dataframe(
                    results_df.sort_values("probability", ascending=False),
                    use_container_width=True,
                )

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV", csv_out, "batch_results.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_publication_pack():
    st.markdown("## Publication Pack Export")

    st.markdown(
        """
    <div class="glass-card" style="padding: 16px; margin-bottom: 20px;">
        Generate a comprehensive research report with high-resolution visualizations, 
        top predictions, and model metadata.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Report Contents")

        include_graph = st.checkbox("Interaction Network Graph", value=True)
        include_sequences = st.checkbox("Sequence Maps", value=True)
        include_metadata = st.checkbox("Model Metadata", value=True)
        top_n = st.slider("Top Predictions", 5, 20, 10)

    with col2:
        st.markdown("### Model Information")

        st.markdown("""
        **Kyroform AI**
        - Architecture: HeteroSAGE (Graph Neural Network)
        - Embeddings: ESM-2 (650M parameters)
        - Input Dimension: 1280
        - Hidden Dimension: 256
        - Training Data: 16M+ gut-host PPIs
        - Validation AUC: ~0.897
        """)

    predictions = _state.get_predictions()
    batch_results = _state.get_batch_results()

    if not predictions and not batch_results:
        st.warning("No predictions available. Run some predictions first.")
        return

    if st.button("Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating publication pack..."):
            lines = []
            lines.append("# Kyroform 2.0 Research Report\n")
            lines.append("## Model Metadata")
            lines.append("- Model: Kyroform 2.0 - HeteroSAGE")
            lines.append("- Input Dimension: 1280 (ESM-2 embeddings)")
            lines.append("- Hidden Dimension: 256")
            lines.append("- Validation AUC: ~0.92")
            lines.append("- Training Data: 16M+ gut-host protein interactions\n")

            if predictions:
                lines.append(f"## Top {top_n} Predictions")
                top_preds = sorted(
                    predictions, key=lambda x: x.get("probability", 0), reverse=True
                )[:top_n]
                lines.append("| Human ID | Bacterial ID | Probability | Confidence |")
                lines.append("|----------|-------------|-------------|------------|")
                for pred in top_preds:
                    lines.append(
                        f"| {pred.get('human_id', '')} | {pred.get('bacterial_id', '')} "
                        f"| {pred.get('probability', 0):.4f} | {pred.get('confidence', 'N/A')} |"
                    )
                lines.append("")

            if batch_results:
                results_df = pd.DataFrame(batch_results)
                valid_probs = results_df["probability"].dropna()
                lines.append("## Batch Analysis Summary")
                lines.append(f"- Total pairs analyzed: {len(results_df)}")
                lines.append(
                    f"- High confidence (>0.7): {int((valid_probs > 0.7).sum())}"
                )
                lines.append(f"- Mean probability: {valid_probs.mean():.4f}")

            report_md = "\n".join(lines)

            st.download_button(
                "Download Markdown Report",
                report_md.encode("utf-8"),
                "kyroform_research_report.md",
                "text/markdown",
            )

            if predictions:
                df = pd.DataFrame(predictions)
                st.download_button(
                    "Download Predictions CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "kyroform_predictions.csv",
                    "text/csv",
                )

            show_toast("Report generated successfully!")


def render_3d_structure_viewer():
    st.markdown("## 3D Structure Viewer")

    human_id = st.selectbox(
        "Human Protein", [""] + sorted(_engine.human_proteins), key="struct_human"
    )
    bact_id = st.selectbox(
        "Bacterial Protein",
        [""] + sorted(_engine.bacterial_proteins),
        key="struct_bact",
    )

    if human_id:
        st.markdown("### Human Protein")
        render_3d_viewer(human_id)

    if bact_id:
        st.markdown("### Bacteria Protein")
        render_3d_viewer(bact_id)


def main():
    render_header()
    render_sidebar()

    current_view = _state.get_view()

    if current_view == "Global Explorer":
        render_global_explorer()
    elif current_view == "Single Predictor":
        render_single_predictor()
    elif current_view == "Batch Analysis":
        render_batch_analysis()
    elif current_view == "Comparative Analysis":
        render_comparative_analysis()
    elif current_view == "3D Structure Viewer":
        render_3d_structure_viewer()
    elif current_view == "Publication Pack":
        render_publication_pack()
    else:
        render_global_explorer()


if __name__ == "__main__":
    main()
