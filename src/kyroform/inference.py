import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_PATH = BASE_DIR / "outputs" / "models" / "esm2_embeddings_1143_proteins.pkl"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "kyroform_ek.pth"


class HeteroSAGE(torch.nn.Module):
    def __init__(self, input_dim: int = 1280, hidden: int = 256):
        super().__init__()
        self.h_conv1 = SAGEConv(input_dim, hidden)
        self.b_conv1 = SAGEConv(input_dim, hidden)
        self.h_conv2 = SAGEConv((input_dim, hidden), hidden)
        self.b_conv2 = SAGEConv((input_dim, hidden), hidden)

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        edge = edge_index_dict[("human", "interacts", "bacterial")]
        rev = edge.flip(0)

        h1 = F.relu(self.h_conv1(x_dict["human"], rev))
        b1 = F.relu(self.b_conv1(x_dict["bacterial"], edge))

        h2 = F.relu(self.h_conv2((x_dict["human"], h1), rev))
        b2 = F.relu(self.b_conv2((x_dict["bacterial"], b1), edge))

        return {"human": h2, "bacterial": b2}


class KyroformInference:
    _instance = None
    _model = None
    _embeddings = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def embeddings(self) -> Dict:
        if self._embeddings is None:
            self._load_embeddings()
        return self._embeddings

    @property
    def human_proteins(self) -> list:
        return [p for p in self.embeddings.keys() if not p.startswith("A0A")]

    @property
    def bacterial_proteins(self) -> list:
        return [p for p in self.embeddings.keys() if p.startswith("A0A")]

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading Kyroform model...")
        self._model = HeteroSAGE(input_dim=1280, hidden=256)

        if MODEL_PATH.exists():
            self._model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")

        self._model.eval()

    def _load_embeddings(self) -> None:
        if self._embeddings is not None:
            return

        logger.info("Loading ESM-2 embeddings...")
        if EMBEDDINGS_PATH.exists():
            try:
                with open(EMBEDDINGS_PATH, "rb") as f:
                    self._embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self._embeddings)} protein embeddings")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                self._embeddings = {}
        else:
            logger.error(f"Embeddings file not found at {EMBEDDINGS_PATH}")
            self._embeddings = {}

    def predict_interaction(
        self, human_id: str, bacterial_id: str
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emb = self.embeddings
        if emb is None or len(emb) == 0:
            raise RuntimeError(
                "Embeddings not loaded. Please ensure embedding file exists."
            )

        if human_id not in emb or bacterial_id not in emb:
            raise ValueError(
                f"Protein IDs not found in embeddings. Available: {len(emb)} proteins"
            )

        m = self.model
        if m is None:
            raise RuntimeError("Model not loaded. Please ensure model files exist.")

        with torch.no_grad():
            h_emb = torch.tensor(self.embeddings[human_id]).unsqueeze(0)
            b_emb = torch.tensor(self.embeddings[bacterial_id]).unsqueeze(0)
            dummy_data = torch.empty((2, 0), dtype=torch.long)

            z = m(
                {"human": h_emb, "bacterial": b_emb},
                {("human", "interacts", "bacterial"): dummy_data},
            )

            z_h = z["human"][0].cpu().numpy()
            print(z)
            z_b = z["bacterial"][0].cpu().numpy()
            score = (z["human"][0] * z["bacterial"][0]).sum().item()
            prob = torch.sigmoid(torch.tensor(score)).item()

        return (
            float(prob),
            z_h,
            z_b,
            np.array(self.embeddings[human_id]),
            np.array(self.embeddings[bacterial_id]),
        )

    def batch_predict(
        self, pairs: list, progress_callback: Optional[callable] = None
    ) -> Generator[dict, None, None]:
        total = len(pairs)

        for idx, (human_id, bacterial_id) in enumerate(pairs):
            try:
                prob, z_h, z_b, emb_h, emb_b = self.predict_interaction(
                    human_id, bacterial_id
                )

                from sklearn.metrics.pairwise import cosine_similarity

                cos_orig = float(
                    cosine_similarity(emb_h.reshape(1, -1), emb_b.reshape(1, -1))[0, 0]
                )
                cos_z = float(
                    cosine_similarity(z_h.reshape(1, -1), z_b.reshape(1, -1))[0, 0]
                )

                if prob >= 0.7:
                    confidence = "High"
                elif prob >= 0.5:
                    confidence = "Moderate"
                elif prob >= 0.25:
                    confidence = "Low"
                else:
                    confidence = "Non-interacting"

                yield {
                    "human_id": human_id,
                    "bacterial_id": bacterial_id,
                    "probability": round(prob, 4),
                    "confidence": confidence,
                    "esm_cosine": round(cos_orig, 4),
                    "latent_cosine": round(cos_z, 4),
                    "norm_h": round(float(np.linalg.norm(emb_h)), 2),
                    "norm_b": round(float(np.linalg.norm(emb_b)), 2),
                    "z_h": z_h,
                    "z_b": z_b,
                }

            except Exception as e:
                yield {
                    "human_id": human_id,
                    "bacterial_id": bacterial_id,
                    "probability": None,
                    "confidence": f"Error: {e}",
                    "esm_cosine": None,
                    "latent_cosine": None,
                    "norm_h": None,
                    "norm_b": None,
                    "z_h": None,
                    "z_b": None,
                }

            if progress_callback:
                progress_callback((idx + 1) / total)

    def compute_contributions(
        self, z_h: np.ndarray, z_b: np.ndarray, topk: int = 10
    ) -> list:
        prod = np.abs(z_h * z_b)
        idx = np.argsort(-prod)[:topk]
        return [{"feature": int(i), "value": float(prod[i])} for i in idx]

    def compute_similar_proteins(self, query_id: str, topk: int = 5) -> list:
        from sklearn.metrics.pairwise import cosine_similarity

        if query_id not in self.embeddings:
            return []

        q = np.array(self.embeddings[query_id]).reshape(1, -1)

        if not query_id.startswith("A0A"):
            keys = [
                k
                for k in self.embeddings.keys()
                if not k.startswith("A0A") and k != query_id
            ]
        else:
            keys = [
                k
                for k in self.embeddings.keys()
                if k.startswith("A0A") and k != query_id
            ]

        if not keys:
            return []

        mat = np.vstack([self.embeddings[k] for k in keys])
        sims = cosine_similarity(q, mat)[0]
        idx = np.argsort(-sims)[:topk]

        return [{"id": keys[i], "score": float(sims[i])} for i in idx]


_inference_engine = None


def get_inference_engine() -> KyroformInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = KyroformInference()
    return _inference_engine
