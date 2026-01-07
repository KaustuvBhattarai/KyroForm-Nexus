import streamlit as st
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pickle
import random
import requests
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageOps, ImageDraw
import io
import base64
import time

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR /"esm2_embeddings_1143_proteins.pkl"
MODEL_PATH = BASE_DIR / "kyroform_ek.pth"

@st.cache_resource
def load_kyroform():
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeds = pickle.load(f)
    
    class HeteroSAGE(torch.nn.Module):
        def __init__(self, input_dim=1280, hidden=256):
            super().__init__()
            # Layer 1: Standard 1280 -> 256
            self.h_conv1 = SAGEConv(input_dim, hidden)
            self.b_conv1 = SAGEConv(input_dim, hidden)
            
            # Layer 2: Hybrid dimensions
            # (1280, 256) means: 
            # neighbors are expected to be 1280-dim, 
            # but the target node itself is 256-dim.
            self.h_conv2 = SAGEConv((input_dim, hidden), hidden)
            self.b_conv2 = SAGEConv((input_dim, hidden), hidden)

        def forward(self, x_dict, edge_index_dict):
            edge = edge_index_dict[('human', 'interacts', 'bacterial')]
            rev = edge.flip(0)
            
            # Layer 1
            h1 = F.relu(self.h_conv1(x_dict['human'], rev))
            b1 = F.relu(self.b_conv1(x_dict['bacterial'], edge))
            
            # Layer 2: Passing a tuple (Source features comma Target features)
            # Neighbors (Source) = Original 1280 features
            # Root (Target) = Layer 1 output (256)
            h2 = F.relu(self.h_conv2((x_dict['human'], h1), rev))
            b2 = F.relu(self.b_conv2((x_dict['bacterial'], b1), edge))
            
            return {'human': h2, 'bacterial': b2}
    
    model = HeteroSAGE(input_dim=1280, hidden=256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    return model, embeds

model, embeds = load_kyroform()

all_human = [p for p in embeds.keys() if not p.startswith('A0A')]
all_bact = [p for p in embeds.keys() if p.startswith('A0A')]

logo_src = BASE_DIR / 'assets' / 'logo.png'
logo_circle = BASE_DIR / 'assets' / 'logo_circle.png'
page_icon = None
if logo_src.exists():
    try:
        im = Image.open(logo_src).convert('RGBA')
        size = min(im.size)
        im = ImageOps.fit(im, (size, size), centering=(0.5, 0.5))
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        im.putalpha(mask)
        im.save(logo_circle)
        page_icon = str(logo_circle)
    except Exception:
        page_icon = None

st.set_page_config(page_title="Kyroform AI", layout="wide", page_icon=page_icon)

css = """
:root{--bg:#000;--panel:#071024;--muted:#9fb7e8;--accent:#1f6feb;--card:#07131f}
body{background:var(--bg);color:#e6eef8;font-family:'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;margin:0;font-size:15px}
.header{display:flex;align-items:center;justify-content:center;flex-direction:column;padding:32px 0 10px 0;margin-bottom:12px}
.logo{width:96px;border-radius:50%;margin-bottom:12px;display:block}
.title{font-size:60px !important;font-weight:700;margin:0;color:#f5fbff;letter-spacing:-0.6px;font-family:'Segoe UI', Roboto, sans-serif}
.subtitle{color:var(--muted);margin-top:8px;margin-bottom:8px;font-size:18px;max-width:980px;text-align:center;line-height:1.25}
.container-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), transparent);border:1px solid rgba(255,255,255,0.04);padding:16px;border-radius:12px;margin-bottom:14px}
.panel-title{font-weight:600;color:#e9f3ff;margin-bottom:8px}
.controls-row{display:flex;gap:10px;align-items:center}
.stButton>button{background:var(--accent);color:white;border-radius:8px;padding:8px 12px}
.stSelectbox>div>div>div>select, .stTextInput>div>div>input{background:#07121a;color:#e6eef8;border-radius:8px;padding:10px;font-size:15px}
.small-pill{background:#0b335f;color:#e6eef8;padding:6px 10px;border-radius:999px;font-size:12px}
.plotly-graph-div{background:transparent}
"""

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
if page_icon and Path(page_icon).exists():
    try:
        with open(page_icon, 'rb') as _f:
            img_b64 = base64.b64encode(_f.read()).decode('ascii')
        header_html = (
            f"<div class='header'><img src='data:image/png;base64,{img_b64}' class='logo'/>"
            "<div class='title'>Kyroform AI</div>"
            "<div class='subtitle'>Gut-Host Interactome Explorer — Predicting novel gut microbiome–host protein interactions</div>"
            "</div>"
        )
    except Exception:
        header_html = "<div class='header'><div class='title'>Kyroform AI</div><div class='subtitle'>Gut-Host Interactome Explorer — Predicting novel gut microbiome–host protein interactions</div></div>"
else:
    header_html = "<div class='header'><div class='title'>Kyroform AI</div><div class='subtitle'>Gut-Host Interactome Explorer — Predicting novel gut microbiome–host protein interactions</div></div>"

st.markdown(header_html, unsafe_allow_html=True)

# Layout: left spacer, main center, right sidebar
cols = st.columns([0.06, 1])
left_spacer, main_col = cols

with main_col:
    st.subheader("Manual Prediction")
    st.markdown("<div style='max-width:1000px;margin:0 auto;'>", unsafe_allow_html=True)
    if 'human_select' not in st.session_state:
        st.session_state['human_select'] = ""
    if 'bact_select' not in st.session_state:
        st.session_state['bact_select'] = ""

    human_id = st.selectbox("Select Human Protein (UniProt ID)", options=[""] + sorted(all_human), index=0, key='human_select')
    bact_id = st.selectbox("Select Bacterial Protein (UniProt ID)", options=[""] + sorted(all_bact), index=0, key='bact_select')

    def predict_pair(human_id, bact_id):
        with torch.no_grad():
            h_emb = torch.tensor(embeds[human_id]).unsqueeze(0)
            b_emb = torch.tensor(embeds[bact_id]).unsqueeze(0)
            dummy_data = torch.empty((2, 0), dtype=torch.long)
            z = model({'human': h_emb, 'bacterial': b_emb}, 
                      {('human', 'interacts', 'bacterial'): dummy_data})
            z_h = z['human'][0].cpu().numpy()
            z_b = z['bacterial'][0].cpu().numpy()
            score = (z['human'][0] * z['bacterial'][0]).sum().item()
            prob = torch.sigmoid(torch.tensor(score)).item()
        return float(prob), z_h, z_b, np.array(embeds[human_id]), np.array(embeds[bact_id])

    # Fetch helpers (moved earlier so variables like human_info are available)
    @st.cache_data
    def fetch_uniprot(uniprot_id):
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None

    @st.cache_data
    def fetch_string_neighbors(uniprot_id, species=9606, limit=10):
        base = "https://string-db.org/api/json/network"
        params = {"identifiers": uniprot_id, "species": species}
        try:
            r = requests.get(base, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                data_sorted = sorted(data, key=lambda x: float(x.get('score', 0)), reverse=True)
                return data_sorted[:limit]
        except Exception:
            return []

    if st.button("Predict Interaction") and human_id and bact_id:
        prob, z_h, z_b, emb_h, emb_b = predict_pair(human_id, bact_id)

        # Fetch UniProt before cause later cause eror
        human_info = fetch_uniprot(human_id)
        bact_info = fetch_uniprot(bact_id)

        with st.expander("Advanced settings", expanded=False):
            calib_samples = st.slider("Calibration sample size", min_value=100, max_value=2000, value=400, step=100)
            neg_ctrl_count = st.slider("Negative control sample count", min_value=10, max_value=200, value=40, step=10)
            show_edge_threshold = st.slider("Edge display threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

        tab1, tab2, tab3 = st.tabs(["Prediction", "Network", "Details"])

        with tab1:
            def create_indicator(prob):
                return go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#1f6feb"},
                        "steps": [
                            {"range": [0, 0.25], "color": "#2b2b2b"},
                            {"range": [0.25, 0.5], "color": "#3b82f6"},
                            {"range": [0.5, 0.7], "color": "#ffa500"},
                            {"range": [0.7, 1], "color": "#ff4d4d"},
                        ],
                    },
                    number={"font": {"size": 36}}
                ))
                

            ind_fig = create_indicator(prob)
            ind_fig.update_layout(paper_bgcolor='#000000', font=dict(color='#e6eef8'))
            st.plotly_chart(ind_fig, use_container_width=True, key=f"indicator_{human_id}_{bact_id}")

            if prob > 0.8:
                st.success("🔥 Strong predicted interaction!")
            elif prob > 0.6:
                st.warning("⚡ Moderate interaction")
            else:
                st.info("➖ Weak/no interaction")

            # Quick embedding similarity (original embeddings + latent z)
            cos_orig = float(cosine_similarity(emb_h.reshape(1, -1), emb_b.reshape(1, -1))[0, 0])
            cos_z = float(cosine_similarity(z_h.reshape(1, -1), z_b.reshape(1, -1))[0, 0])
            st.markdown(f"**Embedding cosine (ESM):** {cos_orig:.4f} — **Latent cosine (model):** {cos_z:.4f}")
            st.markdown(f"**Norms:** ESM_h={np.linalg.norm(emb_h):.2f}, ESM_b={np.linalg.norm(emb_b):.2f}, Z_h={np.linalg.norm(z_h):.2f}, Z_b={np.linalg.norm(z_b):.2f}")

            # Color index / legend for indicator (horizontal compact row)
            st.markdown(
                "<div style='display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-top:8px'>"
                "<div style='background:#2b2b2b;padding:8px 12px;border-radius:8px;color:#e6eef8;text-align:center;font-size:13px'>0-0.25<br/><small style=\'opacity:0.8;font-size:11px\'>Very low</small></div>"
                "<div style='background:#3b82f6;padding:8px 12px;border-radius:8px;color:#e6eef8;text-align:center;font-size:13px'>0.25-0.5<br/><small style=\'opacity:0.8;font-size:11px\'>Low</small></div>"
                "<div style='background:#ffa500;padding:8px 12px;border-radius:8px;color:#000;text-align:center;font-size:13px'>0.5-0.7<br/><small style=\'opacity:0.85;font-size:11px\'>Moderate</small></div>"
                "<div style='background:#ff4d4d;padding:8px 12px;border-radius:8px;color:#000;text-align:center;font-size:13px'>0.7-1.0<br/><small style=\'opacity:0.85;font-size:11px\'>High</small></div>"
                "</div>", unsafe_allow_html=True)

            st.markdown("**Protein Details**")
            pcols = st.columns(2)
            with pcols[0]:
                if human_info:
                    name = human_info.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value')
                    gene = None
                    genes = human_info.get('genes', [])
                    if genes:
                        gene = genes[0].get('geneName', {}).get('value')
                    organism = human_info.get('organism', {}).get('scientificName')
                    length = human_info.get('sequence', {}).get('length')
                    st.markdown(f"**Human: {human_id}**")
                    st.write(f"Name: {name}")
                    st.write(f"Gene: {gene}")
                    st.write(f"Organism: {organism}")
                    st.write(f"Length: {length}")
                    if human_info.get('sequence', {}).get('value'):
                        seq = human_info['sequence']['value']
                        st.text_area("Sequence (truncated)", value=seq[:1000], height=120)
                else:
                    st.write("No UniProt data for human protein")

            with pcols[1]:
                if bact_info:
                    name = bact_info.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value')
                    organism = bact_info.get('organism', {}).get('scientificName')
                    length = bact_info.get('sequence', {}).get('length')
                    st.markdown(f"**Bacterial: {bact_id}**")
                    st.write(f"Name: {name}")
                    st.write(f"Organism: {organism}")
                    st.write(f"Length: {length}")
                    if bact_info.get('sequence', {}).get('value'):
                        seq = bact_info['sequence']['value']
                        st.text_area("Sequence (truncated)", value=seq[:1000], height=120)
                else:
                    st.write("No UniProt data for bacterial protein")

            # Sim proteins
            def compute_similar_proteins(query_id, embeds_dict, topk=5):
                if query_id not in embeds_dict:
                    return []
                q = np.array(embeds_dict[query_id]).reshape(1, -1)
                keys = [k for k in embeds_dict.keys() if not k.startswith('A0A') and k != query_id]
                mat = np.vstack([embeds_dict[k] for k in keys])
                sims = cosine_similarity(q, mat)[0]
                idx = np.argsort(-sims)[:topk]
                return [{'id': keys[i], 'score': float(sims[i])} for i in idx]

            similar = compute_similar_proteins(human_id, embeds, topk=5)
            if similar:
                st.markdown("**Explore Similar Proteins**")
                sim_cols = st.columns([2,1,3])
                with sim_cols[0]:
                    for s in similar:
                        btn_key = f"sim_{s['id']}"
                        if st.button(f"{s['id']} ({s['score']:.3f})", key=btn_key):
                            st.session_state['human_select'] = s['id']
                            st.experimental_rerun()
                # fetch annotations for top similar proteins and show as pill tags
                def parse_uniprot_annotations(ujson):
                    if not ujson:
                        return {'go':[], 'pathways':[], 'diseases':[]}
                    go_terms = []
                    pathways = []
                    diseases = []
                    for db in ujson.get('dbReferences', []):
                        if db.get('type') == 'GO':
                            go_terms.append(db.get('id'))
                        if db.get('type') in ('Reactome', 'KEGG'):
                            pathways.append(db.get('id'))
                    for c in ujson.get('comments', []):
                        if c.get('type') == 'disease':
                            dis = c.get('disease', {}).get('diseaseId') or c.get('text', '')
                            diseases.append(dis)
                    return {'go': go_terms, 'pathways': pathways, 'diseases': diseases}

                with sim_cols[2]:
                    for s in similar:
                        info = fetch_uniprot(s['id'])
                        ann = parse_uniprot_annotations(info)
                        st.markdown(f"**{s['id']}** — {s['score']:.3f}")
                        # render pills
                        pills = []
                        for g in ann['go'][:6]:
                            pills.append(f"<span style='background:#0b3d91;color:#fff;padding:4px 8px;border-radius:12px;margin-right:6px;display:inline-block'>{g}</span>")
                        for p in ann['pathways'][:6]:
                            pills.append(f"<span style='background:#1f6feb;color:#fff;padding:4px 8px;border-radius:12px;margin-right:6px;display:inline-block'>{p}</span>")
                        for d in ann['diseases'][:6]:
                            pills.append(f"<span style='background:#ff4d4d;color:#fff;padding:4px 8px;border-radius:12px;margin-right:6px;display:inline-block'>{d}</span>")
                        st.markdown(''.join(pills), unsafe_allow_html=True)

        # Build interaction network and render in Network tab
        # Fetch and display protein info and interaction network (human_info already fetched)
        # Build interaction graph
        human_neighbors = fetch_string_neighbors(human_id, species=9606, limit=8) or []
        bact_neighbors = fetch_string_neighbors(bact_id, species=511145, limit=8) or []

        G = nx.Graph()
        G.add_node(human_id, label=human_id, color='#ff6666')
        G.add_node(bact_id, label=bact_id, color='#66ff99')

        for n in human_neighbors:
            partner = n.get('preferredName') or n.get('stringId_B')
            score = float(n.get('score', 0))
            G.add_node(partner, label=partner, color='#ffcccc')
            G.add_edge(human_id, partner, weight=score)

        for n in bact_neighbors:
            partner = n.get('preferredName') or n.get('stringId_B')
            score = float(n.get('score', 0))
            G.add_node(partner, label=partner, color='#ccffdd')
            G.add_edge(bact_id, partner, weight=score)

        if prob > show_edge_threshold:
            G.add_edge(human_id, bact_id, weight=prob)

        pos = nx.spring_layout(G, seed=42)

        def edge_hover_text(a, b, score):
            return f"{a} ↔ {b}<br>Score: {score:.3f}" if score is not None else f"{a} ↔ {b}"

        # Build per-edge traces so each edge can have hover text and its own style
        edge_traces = []
        for a, b, d in G.edges(data=True):
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            score = float(d.get('weight', 0))
            if score >= 0.7:
                color = '#ff4d4d'
                width = 3 + score * 3
            elif score >= 0.4:
                color = '#ffa500'
                width = 2 + score * 2
            else:
                color = '#888888'
                width = 1 + score * 2

            # If this is the predicted human-bacterial edge, draw an underlay glow line
            is_predicted_pair = (set([a, b]) == set([human_id, bact_id]))
            if is_predicted_pair:
                # thick, semi-transparent under-edge in blue-ish/purple
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=width + 8, color='rgba(31,111,235,0.12)'),
                    hoverinfo='skip',
                    mode='lines',
                    showlegend=False,
                ))

            edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=width, color=color),
                hoverinfo='text',
                hovertemplate=edge_hover_text(a, b, score),
                text=[edge_hover_text(a, b, score)],
                mode='lines'))

        # Node trace with labels and hover info; also create glow traces for important nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_label = []
        glow_traces = []
        for n, d in G.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            label = d.get('label', n)
            deg = G.degree(n)
            node_text.append(f"{label}<br>Degree: {deg}")
            node_color.append(d.get('color', '#888'))
            node_size.append(12 + deg * 5)
            node_label.append(label if (n == human_id or n == bact_id or deg > 1) else '')

            # Glow for query proteins and high-degree nodes
            if n in (human_id, bact_id) or deg >= 4:
                glow_traces.append(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    hoverinfo='skip',
                    marker=dict(size=(12 + deg * 5) * 2.2, color=d.get('color', '#888'), opacity=0.14, line_width=0),
                    showlegend=False))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_label,
            textposition='top center',
            textfont=dict(size=10, color='#e6eef8'),
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=1))

        # Build figure: place glow traces under normal edges + nodes so glow peeks through
        fig = go.Figure(data=edge_traces + glow_traces + [node_trace],
                        layout=go.Layout(
                            title=f"Interaction network: {human_id} ↔ {bact_id}",
                            title_x=0.5,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            paper_bgcolor='#000000',
                            plot_bgcolor='#000000',
                            font=dict(color='#e6eef8')
                        ))

        # Optional: create a simple pulsate animation for the predicted edge width
        try:
            if (human_id and bact_id) and G.has_edge(human_id, bact_id):
                base_w = next((float(d.get('weight', 0)) for a, b, d in G.edges(data=True) if set([a,b])==set([human_id,bact_id])), 0.5)
                # create 8 frames that are no-op frames (best-effort placeholder for client-side animation)
                frames = []
                for i, t in enumerate(np.linspace(0, 2 * np.pi, 8)):
                    frames.append(go.Frame(name=f'f{i}'))
                if frames:
                    fig.frames = frames
                    fig.update_layout(updatemenus=[dict(type='dropdown', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 450, 'redraw': True}, 'fromcurrent': True}])])])
        except Exception:
            pass

        with tab2:
            # Full-width graph for better use of space
            st.plotly_chart(fig, use_container_width=True, key=f"network_{human_id}_{bact_id}")

            # Centered middle area with guide + top neighbors for better layout
            mid = st.columns([1, 2, 1])[1]
            with mid:
                st.markdown("**Graph guide**")
                st.markdown("<div style='background:#0b0b0b;padding:12px;border-radius:8px'>", unsafe_allow_html=True)
                st.markdown(f"**Predicted pair:** {human_id} ↔ {bact_id}")
                st.markdown(f"**Model probability:** **{prob:.3f}**")
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("**Legend**")
                st.markdown("<div style='display:flex;gap:10px;flex-wrap:wrap'>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ff6666;margin-right:8px;border-radius:2px'></span>Query human protein</div>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#66ff99;margin-right:8px;border-radius:2px'></span>Query bacterial protein</div>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ffcccc;margin-right:8px;border-radius:2px'></span>Human neighbors (STRING)</div>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ccffdd;margin-right:8px;border-radius:2px'></span>Bacterial neighbors (STRING)</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("**How to read the graph**")
                st.markdown("- Nodes represent proteins. Hover to see name and degree (connections).")
                st.markdown("- Edge thickness and color reflect interaction score (STRING) or model probability for the query pair.")
                st.markdown("- Larger nodes have more connections; they may be hubs or well-studied proteins.")
                st.markdown("</div>", unsafe_allow_html=True)

                # Top neighbors beneath the guide
                if human_neighbors:
                    df_h = pd.DataFrame([{'partner': (n.get('preferredName') or n.get('stringId_B')), 'score': float(n.get('score',0))} for n in human_neighbors])
                    st.markdown("**Top human neighbors (STRING)**")
                    st.table(df_h)
                if bact_neighbors:
                    df_b = pd.DataFrame([{'partner': (n.get('preferredName') or n.get('stringId_B')), 'score': float(n.get('score',0))} for n in bact_neighbors])
                    st.markdown("**Top bacterial neighbors (STRING)**")
                    st.table(df_b)

        st.markdown("</div>", unsafe_allow_html=True)

        # --- Details tab: Why this prediction? embeddings, contributions, calibration, neg controls, sequence highlighting
        def compute_contributions(z_h, z_b, topk=10):
            prod = np.abs(z_h * z_b)
            idx = np.argsort(-prod)[:topk]
            return [{'feature': int(i), 'value': float(prod[i])} for i in idx]

        @st.cache_data
        def sample_calibration(n=400):
            # Sample random human-bacterial pairs to approximate score distribution
            pairs = [(random.choice(all_human), random.choice(all_bact)) for _ in range(n)]
            probs = []
            for h, b in pairs:
                with torch.no_grad():
                    h_emb = torch.tensor(embeds[h]).unsqueeze(0)
                    b_emb = torch.tensor(embeds[b]).unsqueeze(0)
                    dummy = torch.empty((2, 0), dtype=torch.long)
                    z = model({'human': h_emb, 'bacterial': b_emb}, {('human', 'interacts', 'bacterial'): dummy})
                    score = (z['human'][0] * z['bacterial'][0]).sum().item()
                    probs.append(float(torch.sigmoid(torch.tensor(score)).item()))
            return probs

        @st.cache_data
        def sample_negative_controls(n=40):
            # Sample human-human and bact-bact unrelated pairs
            hh = [(random.choice(all_human), random.choice(all_human)) for _ in range(n)]
            bb = [(random.choice(all_bact), random.choice(all_bact)) for _ in range(n)]
            scores_hh = []
            scores_bb = []
            for a, b in hh:
                with torch.no_grad():
                    h_emb = torch.tensor(embeds[a]).unsqueeze(0)
                    b_emb = torch.tensor(embeds[b]).unsqueeze(0)
                    dummy = torch.empty((2, 0), dtype=torch.long)
                    z = model({'human': h_emb, 'bacterial': b_emb}, {('human', 'interacts', 'bacterial'): dummy})
                    score = (z['human'][0] * z['bacterial'][0]).sum().item()
                    scores_hh.append(float(torch.sigmoid(torch.tensor(score)).item()))
            for a, b in bb:
                with torch.no_grad():
                    h_emb = torch.tensor(embeds[a]).unsqueeze(0)
                    b_emb = torch.tensor(embeds[b]).unsqueeze(0)
                    dummy = torch.empty((2, 0), dtype=torch.long)
                    z = model({'human': h_emb, 'bacterial': b_emb}, {('human', 'interacts', 'bacterial'): dummy})
                    score = (z['human'][0] * z['bacterial'][0]).sum().item()
                    scores_bb.append(float(torch.sigmoid(torch.tensor(score)).item()))
            return scores_hh, scores_bb

        def highlight_sequence_html(seq):
            # simple heuristics for signal peptide, TM regions, low-complexity
            hydrophobic = set(list('AILMFVWY'))
            seq = seq or ''
            n = len(seq)
            flags = ['none'] * n

            # signal peptide heuristic: first 30 aa have many hydrophobic residues
            if n >= 20:
                window = seq[:30]
                frac = sum(1 for c in window if c in hydrophobic) / max(1, len(window))
                if frac > 0.55:
                    for i in range(min(30, n)):
                        flags[i] = 'signal'

            # transmembrane heuristic: sliding window 18
            for i in range(0, max(1, n - 17)):
                w = seq[i:i+18]
                frac = sum(1 for c in w if c in hydrophobic) / 18
                if frac > 0.75:
                    for j in range(i, min(i+18, n)):
                        flags[j] = 'tm'

            # low-complexity: window 30, high single-aa fraction
            for i in range(0, max(1, n - 29)):
                w = seq[i:i+30]
                from collections import Counter
                c = Counter(w)
                top_frac = c.most_common(1)[0][1] / 30
                if top_frac > 0.6:
                    for j in range(i, min(i+30, n)):
                        if flags[j] == 'none':
                            flags[j] = 'lowcomp'

            # Build HTML
            out = []
            color_map = {'signal':'#ffd27f','tm':'#ffa5d0','lowcomp':'#6b7280','none':None}
            i = 0
            while i < n:
                f = flags[i]
                j = i
                while j < n and flags[j] == f:
                    j += 1
                segment = seq[i:j]
                if f == 'none':
                    out.append(segment)
                else:
                    color = color_map.get(f, '#ffffff')
                    out.append(f"<span style='background:{color};padding:1px 2px;border-radius:3px;margin-right:1px;color:#000'>{segment}</span>")
                i = j
            return '<div style="font-family:monospace;white-space:pre-wrap">' + ''.join(out) + '</div>'

        with tab3:
            st.header("Why this prediction?")
            st.markdown("**Embedding similarity & norms**")
            st.write(f"ESM cosine: {cos_orig:.4f}, latent cosine: {cos_z:.4f}")
            st.write(f"Norms — ESM_h: {np.linalg.norm(emb_h):.2f}, ESM_b: {np.linalg.norm(emb_b):.2f}, Z_h: {np.linalg.norm(z_h):.2f}, Z_b: {np.linalg.norm(z_b):.2f}")

            st.markdown("**Top contributing latent dimensions (abs(z_h * z_b))**")
            contribs = compute_contributions(z_h, z_b, topk=10)
            dfc = pd.DataFrame(contribs)
            st.table(dfc)

            st.markdown("**Confidence calibration**")
            calib_probs = sample_calibration(calib_samples)
            hist = go.Figure()
            hist.add_trace(go.Histogram(x=calib_probs, nbinsx=30, marker=dict(color='#1f6feb'), opacity=0.8, name='All pairs'))
            hist.add_trace(go.Scatter(x=[prob, prob], y=[0, max(1, max(np.histogram(calib_probs, bins=30)[0]))], mode='lines', line=dict(color='#ff4d4d', width=3), name='This prediction'))
            hist.update_layout(paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#e6eef8'), showlegend=True)
            st.plotly_chart(hist, use_container_width=True, key=f"calib_{human_id}_{bact_id}")

            st.markdown("**Negative controls**")
            neg_hh, neg_bb = sample_negative_controls(neg_ctrl_count)
            fig_nc = go.Figure()
            fig_nc.add_trace(go.Histogram(x=neg_hh, nbinsx=20, name='Human-Human', marker=dict(color='#ffa500'), opacity=0.6))
            fig_nc.add_trace(go.Histogram(x=neg_bb, nbinsx=20, name='Bact-Bact', marker=dict(color='#66ff99'), opacity=0.6))
            fig_nc.add_trace(go.Scatter(x=[prob, prob], y=[0, max(1, max(np.histogram(neg_hh+neg_bb, bins=20)[0]))], mode='lines', line=dict(color='#1f6feb', width=3), name='This prediction'))
            fig_nc.update_layout(barmode='overlay', paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#e6eef8'))
            st.plotly_chart(fig_nc, use_container_width=True, key=f"neg_{human_id}_{bact_id}")

            st.markdown("**Sequence highlights**")
            if human_info and human_info.get('sequence', {}).get('value'):
                seqh = human_info['sequence']['value']
                st.markdown("**Human sequence**")
                st.markdown(highlight_sequence_html(seqh), unsafe_allow_html=True)
            if bact_info and bact_info.get('sequence', {}).get('value'):
                seqb = bact_info['sequence']['value']
                st.markdown("**Bacterial sequence**")
                st.markdown(highlight_sequence_html(seqb), unsafe_allow_html=True)


st.markdown("---")
st.caption("Kyroform AI v13.1.1 | Last update on 26-01-07 12:39 by Kaustuv | Trained on 2025 predicted gut-host PPIs | Current Config : SLE | ESM-2 + Heterogeneous GraphSAGE")