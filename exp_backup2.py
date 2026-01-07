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
            
            # Layer 2: Passing a tuple (Source features, Target features)
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

st.set_page_config(page_title="Kyroform AI", layout="wide")
st.markdown("<style>\n@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');\nbody { background-color: #000000; color: #e6eef8; font-family: Inter, Roboto, 'Helvetica Neue', Arial, sans-serif; }\nh1 { font-size: 36px; font-weight:700; margin-bottom:6px; }\n.h-sub { color:#bcd7ff; margin-top:0px; margin-bottom:12px; }\n.stButton>button { background-color: #1f6feb; color: white; border-radius:6px; }\n.stTextInput>div>div>input { background-color: #111; color: #e6eef8; }\n.metric-label, .stMetric { color: #e6eef8; }\n</style>", unsafe_allow_html=True)
st.markdown("<h1>Kyroform AI: Gut-Host Interactome Explorer</h1>", unsafe_allow_html=True)
st.markdown("<div class='h-sub'>Predicting novel gut microbiome–host protein interactions in autoimmune diseases</div>", unsafe_allow_html=True)

# Layout: left spacer, main center, right sidebar
cols = st.columns([0.5, 3, 1])
left_spacer, main_col, side_col = cols

with main_col:
    st.subheader("Manual Prediction")
    st.markdown("<div style='max-width:1000px;margin:0 auto;'>", unsafe_allow_html=True)
    human_id = st.selectbox("Select Human Protein (UniProt ID)", options=[""] + sorted(all_human), index=0)
    bact_id = st.selectbox("Select Bacterial Protein (UniProt ID)", options=[""] + sorted(all_bact), index=0)

    def predict_pair(human_id, bact_id):
        with torch.no_grad():
            h_emb = torch.tensor(embeds[human_id]).unsqueeze(0)
            b_emb = torch.tensor(embeds[bact_id]).unsqueeze(0)
            dummy_data = torch.empty((2, 0), dtype=torch.long)
            z = model({'human': h_emb, 'bacterial': b_emb}, 
                      {('human', 'interacts', 'bacterial'): dummy_data})
            score = (z['human'][0] * z['bacterial'][0]).sum().item()
            prob = torch.sigmoid(torch.tensor(score)).item()
        return prob

    if st.button("Predict Interaction") and human_id and bact_id:
        prob = predict_pair(human_id, bact_id)
        # Show a richer gauge + number indicator
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
        st.plotly_chart(ind_fig, use_container_width=True)
        if prob > 0.8:
            st.success("🔥 Strong predicted interaction!")
        elif prob > 0.6:
            st.warning("⚡ Moderate interaction")
        else:
            st.info("➖ Weak/no interaction")

        # Fetch and display protein info and interaction network
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

        human_info = fetch_uniprot(human_id)
        bact_info = fetch_uniprot(bact_id)

        st.markdown("**Protein Details**")
        cols = st.columns(2)
        with cols[0]:
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

        with cols[1]:
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

        if prob > 0.3:
            G.add_edge(human_id, bact_id, weight=prob)

        pos = nx.spring_layout(G, seed=42)

        # Helper: create hovertemplate and styling for edges
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

        cols_graph = st.columns([3, 1])
        with cols_graph[0]:
            st.plotly_chart(fig, use_container_width=True)

        with cols_graph[1]:
            st.markdown("**Graph guide**")
            st.markdown("<div style='background:#0b0b0b;padding:8px;border-radius:6px'>", unsafe_allow_html=True)
            st.markdown(f"**Predicted pair:** {human_id} ↔ {bact_id}")
            st.markdown(f"**Model probability:** **{prob:.3f}**")
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("**Legend**")
            st.markdown("<span style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ff6666;margin-right:8px;border-radius:2px'></span>Query human protein</span>", unsafe_allow_html=True)
            st.markdown("<span style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#66ff99;margin-right:8px;border-radius:2px'></span>Query bacterial protein</span>", unsafe_allow_html=True)
            st.markdown("<span style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ffcccc;margin-right:8px;border-radius:2px'></span>Human neighbors (STRING)</span>", unsafe_allow_html=True)
            st.markdown("<span style='display:flex;align-items:center'><span style='display:inline-block;width:12px;height:12px;background:#ccffdd;margin-right:8px;border-radius:2px'></span>Bacterial neighbors (STRING)</span>", unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("**How to read the graph**")
            st.markdown("- Nodes represent proteins. Hover to see name and degree (connections).")
            st.markdown("- Edge thickness and color reflect interaction score (STRING) or model probability for the query pair.")
            st.markdown("- Larger nodes have more connections; they may be hubs or well-studied proteins.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Show top neighbor tables below the guide
        with cols_graph[1]:
            if human_neighbors:
                df_h = pd.DataFrame([{'partner': (n.get('preferredName') or n.get('stringId_B')), 'score': float(n.get('score',0))} for n in human_neighbors])
                st.markdown("**Top human neighbors (STRING)**")
                st.table(df_h)
            if bact_neighbors:
                df_b = pd.DataFrame([{'partner': (n.get('preferredName') or n.get('stringId_B')), 'score': float(n.get('score',0))} for n in bact_neighbors])
                st.markdown("**Top bacterial neighbors (STRING)**")
                st.table(df_b)

        # Compute and show similar human proteins (cosine similarity on loaded embeddings)
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
                        # autofill the human selection and re-run (Streamlit state)
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

        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.subheader("Random Exploration")
    if st.button("Generate 5 Random Predictions"):
        st.write("### Random Gut-Host Candidates")
        random.seed()
        pairs = random.sample(list(zip(random.choices(all_human, k=5), random.choices(all_bact, k=5))), 5)
        for h, b in pairs:
            with torch.no_grad():
                h_emb = torch.tensor(embeds[h]).unsqueeze(0)
                b_emb = torch.tensor(embeds[b]).unsqueeze(0)
                dummy_data = torch.empty((2, 0), dtype=torch.long)
                z = model({'human': h_emb, 'bacterial': b_emb}, 
                          {('human', 'interacts', 'bacterial'): dummy_data})
                score = (z['human'][0] * z['bacterial'][0]).sum().item()
                prob = torch.sigmoid(torch.tensor(score)).item()
            
            status = "Positive" if prob > 0.7 else "Weak"
            st.write(f"**{h}** ↔ **{b}** → **{prob:.4f}** ({status})")

st.markdown("---")
st.caption("Kyroform AI v13.1.1 | Trained on 2025 predicted gut-host PPIs | ESM-2 + Heterogeneous GraphSAGE")