import streamlit as st
from typing import Any, Dict, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionStateManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_state()
            SessionStateManager._initialized = True

    def _initialize_state(self) -> None:
        if "kyroform_initialized" not in st.session_state:
            st.session_state["kyroform_initialized"] = True
            st.session_state["current_view"] = "Global Explorer"
            st.session_state["selected_human"] = ""
            st.session_state["selected_bacterial"] = ""
            st.session_state["prediction_results"] = []
            st.session_state["batch_results"] = []
            st.session_state["graph_data"] = {}
            st.session_state["selected_protein"] = None
            st.session_state["disease_context"] = "All"
            st.session_state["comparison_battery"] = []
            st.session_state["show_3d_viewer"] = False
            st.session_state["calibration_samples"] = 400
            st.session_state["neg_controls"] = 40
            st.session_state["edge_threshold"] = 0.3
            st.session_state["graph_layout"] = "spring"
            st.session_state["last_update"] = None

    def reset_prediction(self) -> None:
        st.session_state["selected_human"] = ""
        st.session_state["selected_bacterial"] = ""
        st.session_state["prediction_results"] = []
        st.session_state["graph_data"] = {}
        st.session_state["selected_protein"] = None

    def set_view(self, view_name: str) -> None:
        st.session_state["current_view"] = view_name

    def get_view(self) -> str:
        return st.session_state.get("current_view", "Global Explorer")

    def store_prediction(self, result: Dict) -> None:
        if "prediction_results" not in st.session_state:
            st.session_state["prediction_results"] = []
        st.session_state["prediction_results"].append(
            {**result, "timestamp": self._get_timestamp()}
        )

    def get_predictions(self) -> list:
        return st.session_state.get("prediction_results", [])

    def store_batch_results(self, results: list) -> None:
        st.session_state["batch_results"] = results

    def get_batch_results(self) -> list:
        return st.session_state.get("batch_results", [])

    def set_selected_protein(self, protein_id: str, protein_type: str) -> None:
        st.session_state["selected_protein"] = {"id": protein_id, "type": protein_type}

    def get_selected_protein(self) -> Optional[Dict]:
        return st.session_state.get("selected_protein")

    def set_disease_context(self, disease: str) -> None:
        st.session_state["disease_context"] = disease

    def get_disease_context(self) -> str:
        return st.session_state.get("disease_context", "All")

    def set_comparison_battery(self, proteins: list) -> None:
        st.session_state["comparison_battery"] = proteins

    def get_comparison_battery(self) -> list:
        return st.session_state.get("comparison_battery", [])

    def set_3d_viewer_state(self, show: bool) -> None:
        st.session_state["show_3d_viewer"] = show

    def get_3d_viewer_state(self) -> bool:
        return st.session_state.get("show_3d_viewer", False)

    def update_settings(self, key: str, value: Any) -> None:
        st.session_state[key] = value

    def get_setting(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def _get_timestamp(self) -> str:
        from datetime import datetime

        return datetime.now().isoformat()

    def clear_session(self) -> None:
        keys_to_keep = ["kyroform_initialized"]
        keys_to_clear = [k for k in st.session_state.keys() if k not in keys_to_keep]

        for key in keys_to_clear:
            del st.session_state[key]

        self._initialize_state()
        logger.info("Session state cleared and reinitialized")

    def export_state(self) -> Dict:
        return {
            "current_view": self.get_view(),
            "selected_human": st.session_state.get("selected_human", ""),
            "selected_bacterial": st.session_state.get("selected_bacterial", ""),
            "disease_context": self.get_disease_context(),
            "prediction_count": len(self.get_predictions()),
            "batch_count": len(self.get_batch_results()),
            "last_update": st.session_state.get("last_update"),
        }


_state_manager = None


def get_state_manager() -> SessionStateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = SessionStateManager()
    return _state_manager


DISEASE_ONTOLOGY = {
    "All": {
        "description": "Show all predicted interactions without disease filtering",
        "human_genes": [],
    },
    "Systemic Lupus Erythematosus (SLE)": {
        "description": "Autoimmune disease affecting multiple organ systems",
        "human_genes": [
            "PTPN22",
            "IRF5",
            "STAT4",
            "TNFAIP3",
            "ITGAM",
            "FCGR2A",
            "HLA-DR3",
            "HLA-DR15",
            "TREX1",
            "RNASEL",
        ],
    },
    "Inflammatory Bowel Disease (IBD)": {
        "description": "Chronic inflammatory conditions of the gastrointestinal tract",
        "human_genes": [
            "NOD2",
            "IL23R",
            "ATG16L1",
            "IRGM",
            "PTGER4",
            "LCK",
            "IL10",
            "CARD9",
            "SLCO1A2",
            "SLC22A4",
        ],
    },
    "Rheumatoid Arthritis (RA)": {
        "description": "Chronic autoimmune arthritis affecting joints",
        "human_genes": [
            "HLA-DRB1",
            "PTPN22",
            "PADI4",
            "STAT4",
            "TRAF1",
            "CTLA4",
            "IL2RB",
            "IL6R",
            "FCGR2A",
            "TNFAIP3",
        ],
    },
    "Type 1 Diabetes (T1D)": {
        "description": "Autoimmune destruction of pancreatic beta cells",
        "human_genes": [
            "HLA-DR3",
            "HLA-DR4",
            "INS",
            "PTPN22",
            "CTLA4",
            "IL2RA",
            "CD25",
            "IFIH1",
            "SH2B3",
            "ERBB3",
        ],
    },
    "Multiple Sclerosis (MS)": {
        "description": "Autoimmune demyelination in central nervous system",
        "human_genes": [
            "HLA-DRB1",
            "IL2RA",
            "IL7R",
            "CD6",
            "TYK2",
            "STAT3",
            "IRF8",
            "EOMES",
            "CTLA4",
            "CYP2R1",
        ],
    },
    "Celiac Disease": {
        "description": "Autoimmune reaction to gluten in small intestine",
        "human_genes": [
            "HLA-DQ2",
            "HLA-DQ8",
            "CTLA4",
            "PTPN2",
            "IL2",
            "IL21",
            "CCR1",
            "POMP",
            "IL12A",
            "SH2B3",
        ],
    },
}


def get_disease_genes(disease: str) -> list:
    return DISEASE_ONTOLOGY.get(disease, {}).get("human_genes", [])


def get_all_diseases() -> list:
    return list(DISEASE_ONTOLOGY.keys())


import json
from datetime import datetime


def save_session_to_kyro(state_dict: dict) -> bytes:
    """Save current session state to a .kyro file (JSON)"""
    session_data = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "current_view": state_dict.get("current_view", "Global Explorer"),
        "selected_human": state_dict.get("selected_human", ""),
        "selected_bacterial": state_dict.get("selected_bacterial", ""),
        "disease_context": state_dict.get("disease_context", "All"),
        "prediction_results": state_dict.get("prediction_results", []),
        "batch_results": state_dict.get("batch_results", []),
        "graph_data": state_dict.get("graph_data", {}),
        "settings": {
            "calibration_samples": state_dict.get("calibration_samples", 400),
            "neg_controls": state_dict.get("neg_controls", 40),
            "edge_threshold": state_dict.get("edge_threshold", 0.3),
        },
    }
    return json.dumps(session_data, indent=2).encode("utf-8")


def load_session_from_kyro(file_bytes: bytes) -> dict:
    """Load session state from a .kyro file"""
    try:
        data = json.loads(file_bytes.decode("utf-8"))
        return data
    except Exception as e:
        logger.error(f"Error loading .kyro file: {e}")
        return {}
