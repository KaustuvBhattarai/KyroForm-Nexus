import requests
import streamlit as st
from typing import Optional, Dict, List
import logging
import time
import numpy as np
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_uniprot(uniprot_id: str) -> Optional[Dict]:
    @st.cache_data(ttl=3600)
    def _fetch(uniprot_id: str) -> Optional[Dict]:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                logger.warning(f"UniProt ID not found: {uniprot_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching UniProt data: {e}")
        return None

    return _fetch(uniprot_id)


def fetch_string_neighbors(
    uniprot_id: str, species: int = 9606, limit: int = 10
) -> list:
    @st.cache_data(ttl=1800)
    def _fetch(uniprot_id: str, species: int, limit: int) -> list:
        base = "https://string-db.org/api/json/network"
        params = {"identifiers": uniprot_id, "species": species}
        try:
            r = requests.get(base, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                data_sorted = sorted(
                    data, key=lambda x: float(x.get("score", 0)), reverse=True
                )
                return data_sorted[:limit]
        except Exception as e:
            logger.error(f"Error fetching STRING neighbors: {e}")
        return []

    return _fetch(uniprot_id, species, limit)


def fetch_alphafold_structure(uniprot_id: str) -> Optional[str]:
    @st.cache_data(ttl=86400)
    def _fetch(uniprot_id: str) -> Optional[str]:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0:
                    return data[0].get("pdbUrl")
        except Exception as e:
            logger.warning(f"AlphaFold structure not available: {e}")
        return None

    return _fetch(uniprot_id)


def parse_uniprot_annotations(ujson: Optional[Dict]) -> Dict:
    if not ujson:
        return {"go": [], "pathways": [], "diseases": [], "functions": []}

    go_terms = []
    pathways = []
    diseases = []
    functions = []

    for db in ujson.get("dbReferences", []):
        if db.get("type") == "GO":
            go_terms.append(db.get("id"))
        if db.get("type") in ("Reactome", "KEGG"):
            pathways.append(db.get("id"))

    for c in ujson.get("comments", []):
        if c.get("type") == "disease":
            dis = c.get("disease", {}).get("diseaseId") or c.get("text", "")
            if dis:
                diseases.append(dis)
        if c.get("type") == "function":
            func = c.get("text", "")
            if func:
                functions.append(func[:200])

    return {
        "go": go_terms[:10],
        "pathways": pathways[:10],
        "diseases": diseases[:5],
        "functions": functions[:3],
    }


def format_protein_info(ujson: Optional[Dict]) -> Dict:
    if not ujson:
        return {}

    name = (
        ujson.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "Unknown")
    )

    gene = None
    genes = ujson.get("genes", [])
    if genes:
        gene = genes[0].get("geneName", {}).get("value")

    organism = ujson.get("organism", {}).get("scientificName", "Unknown")
    length = ujson.get("sequence", {}).get("length", 0)
    sequence = ujson.get("sequence", {}).get("value", "")

    return {
        "name": name,
        "gene": gene,
        "organism": organism,
        "length": length,
        "sequence": sequence,
        "uniprot_id": ujson.get("primaryAccession", ""),
    }


def highlight_sequence(seq: str) -> str:
    hydrophobic = set(list("AILMFVWY"))
    seq = seq or ""
    n = len(seq)
    flags = ["none"] * n

    if n >= 20:
        window = seq[:30]
        frac = sum(1 for c in window if c in hydrophobic) / max(1, len(window))
        if frac > 0.55:
            for i in range(min(30, n)):
                flags[i] = "signal"

    for i in range(0, max(1, n - 17)):
        w = seq[i : i + 18]
        frac = sum(1 for c in w if c in hydrophobic) / 18
        if frac > 0.75:
            for j in range(i, min(i + 18, n)):
                flags[j] = "tm"

    from collections import Counter

    for i in range(0, max(1, n - 29)):
        w = seq[i : i + 30]
        c = Counter(w)
        top_frac = c.most_common(1)[0][1] / 30
        if top_frac > 0.6:
            for j in range(i, min(i + 30, n)):
                if flags[j] == "none":
                    flags[j] = "lowcomp"

    color_map = {
        "signal": "#ffd27f",
        "tm": "#ffa5d0",
        "lowcomp": "#6b7280",
        "none": None,
    }

    out = []
    i = 0
    while i < n:
        f = flags[i]
        j = i
        while j < n and flags[j] == f:
            j += 1
        segment = seq[i:j]
        if f == "none":
            out.append(segment)
        else:
            color = color_map.get(f, "#ffffff")
            out.append(
                f"<span style='background:{color};padding:1px 2px;border-radius:3px;margin-right:1px;color:#000'>{segment}</span>"
            )
        i = j

    return (
        '<div style="font-family:monospace;white-space:pre-wrap;font-size:12px;line-height:1.4;">'
        + "".join(out)
        + "</div>"
    )


def create_sequence_map_html(
    seq: str, interaction_sites: list = None, domains: list = None
) -> str:
    if not seq:
        return "<p>No sequence available</p>"

    n = len(seq)
    chunk_size = 10
    blocks = []

    for i in range(0, n, chunk_size):
        chunk = seq[i : i + chunk_size]
        blocks.append(f"<span style='color:#60a5fa'>{i + 1:4d}</span> {chunk}")

    lines = []
    for i in range(0, len(blocks), 6):
        lines.append(" ".join(blocks[i : i + 6]))

    seq_display = "<br>".join(lines)

    html = f"""
    <div style="background:#0f172a;padding:16px;border-radius:8px;font-family:monospace;font-size:11px;overflow-x:auto;">
        <div style="color:#94a3b8;margin-bottom:8px;">
            <span style="color:#f59e0b">●</span> Signal peptide (positions 1-30)
            <span style="color:#ec4899;margin-left:12px;">●</span> Transmembrane (positions 31-50)
            <span style="color:#6b7280;margin-left:12px;">●</span> Low complexity
        </div>
        <div style="color:#e2e8f0;line-height:1.6;">{seq_display}</div>
    </div>
    """
    return html


def show_toast(message: str, icon: str = "✅") -> None:
    st.toast(f"{icon} {message}")


def show_loading_spinner(message: str = "Loading...") -> None:
    with st.spinner(message):
        time.sleep(0.5)


def format_confidence_gauge(probability: float) -> Dict:
    if probability >= 0.7:
        label = "High"
        color = "#10b981"
        description = "Strong predicted interaction"
    elif probability >= 0.5:
        label = "Moderate"
        color = "#f59e0b"
        description = "Moderate interaction potential"
    elif probability >= 0.25:
        label = "Low"
        color = "#3b82f6"
        description = "Weak interaction signal"
    else:
        label = "Very Low"
        color = "#6b7280"
        description = "Minimal interaction likelihood"

    return {
        "label": label,
        "color": color,
        "description": description,
        "percentile": f"top {int((1 - probability) * 100)}%",
    }


def format_large_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fetch_open_targets(uniprot_id: str) -> Optional[Dict]:
    @st.cache_data(ttl=3600)
    def _fetch(uniprot_id: str) -> Optional[Dict]:
        url = f"https://api.opentargets.org/v3/target/{uniprot_id}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.warning(f"Open Targets API error: {e}")
        return None

    return _fetch(uniprot_id)


def fetch_pubmed_count(uniprot_id: str) -> int:
    @st.cache_data(ttl=3600)
    def _fetch(uniprot_id: str) -> int:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=protein:{uniprot_id}&resulttype=lite&format=json"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                return data.get("hitCount", 0)
        except Exception:
            pass
        return 0

    return _fetch(uniprot_id)


AUTOIMMUNE_PATHWAYS = {
    "IL-23 Signaling": [
        "IL23A",
        "IL23R",
        "IL12B",
        "STAT3",
        "JAK2",
        "TYK2",
        "RORC",
        "IL17A",
        "IL17F",
    ],
    "TNF-alpha Signaling": [
        "TNF",
        "TNFR1",
        "TNFR2",
        "NFKB1",
        "NFKBIA",
        "IKBKB",
        "RIPK1",
        "TRADD",
        "FADD",
    ],
    "Type I Interferon": [
        "IFNA1",
        "IFNB1",
        "IFNAR1",
        "IFNAR2",
        "STAT1",
        "STAT2",
        "IRF9",
        "MX1",
        "OAS1",
    ],
    "IL-10 Signaling": ["IL10", "IL10RA", "IL10RB", "STAT3", "SOCS3", "TGFB1"],
    "NF-kB Pathway": [
        "NFKB1",
        "NFKB2",
        "RELA",
        "REL",
        "CREB",
        "IKBKA",
        "IKBKB",
        "NEMO",
    ],
    "TLR Signaling": [
        "TLR1",
        "TLR2",
        "TLR3",
        "TLR4",
        "TLR5",
        "TLR6",
        "MYD88",
        "TRAF6",
        "IRAK4",
    ],
    "JAK-STAT": [
        "JAK1",
        "JAK2",
        "JAK3",
        "TYK2",
        "STAT1",
        "STAT2",
        "STAT3",
        "STAT4",
        "STAT5A",
        "STAT5B",
    ],
}


def check_pathway_involvement(gene_symbol: str) -> list:
    """Check if a gene is involved in major autoimmune pathways"""
    involved = []
    for pathway, genes in AUTOIMMUNE_PATHWAYS.items():
        if gene_symbol.upper() in [g.upper() for g in genes]:
            involved.append(pathway)
    return involved


def calculate_kyro_score(
    probability: float, esm_cosine: float = 0.5, centrality: float = 0.5
) -> float:
    """
    Calculate Kyro-Score: weighted combination of model probability,
    sequence similarity, and topological importance
    """
    weights = {"probability": 0.5, "sequence": 0.3, "topology": 0.2}
    score = (
        weights["probability"] * probability
        + weights["sequence"] * (esm_cosine + 1) / 2
        + weights["topology"] * centrality
    )
    return min(1.0, max(0.0, score))


def generate_contact_map_heatmap(pred_h: np.ndarray, pred_b: np.ndarray) -> go.Figure:
    """Generate a contact map heatmap from latent representations"""
    contact_map = np.outer(pred_h, pred_b)

    fig = go.Figure(
        data=go.Heatmap(
            z=contact_map,
            colorscale="RdBu",
            zmid=0,
            showscale=True,
            colorbar=dict(title="Interaction Strength"),
        )
    )

    fig.update_layout(
        title="Predicted Contact Map",
        xaxis_title="Human Protein Dimensions",
        yaxis_title="Bacterial Protein Dimensions",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1A1A1A"),
        height=400,
    )
    return fig


def compute_saliency(
    h_embedding: np.ndarray, b_embedding: np.ndarray, position: int, mutation: str
) -> float:
    """Simulate saliency for in-silico mutagenesis"""
    mutation_effect = abs(ord(mutation.upper()) - ord("A")) / 25.0
    base_saliency = abs(h_embedding[position] * b_embedding[position])
    return base_saliency * (1 + mutation_effect)
