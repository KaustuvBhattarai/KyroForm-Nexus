# KyroForm AI
### Gut-Host Interactome Discovery Engine

**Live Demo**: [KyroForm on Hugging Face Spaces](https://huggingface.co/spaces/kr1nzl3r/KyroForm)

---

## Key Results

| Metric | Value |
|--------|-------|
| Validation AUC | **0.92** |
| Validation AP | **0.88** |
| Proteins Modeled | 1,143 |
| PPI Training Data | 16M+ |

---

## What It Does

KyroForm is an ML system that predicts protein-protein interactions (PPIs) between human host proteins and gut microbiome bacteria — critical for understanding autoimmune diseases like Systemic Lupus Erythematosus (SLE).

Uses a **Heterogeneous Graph Neural Network (HGNN)** with **ESM-2 protein embeddings** to model the gut-host interactome.

---

## Project Structure

```
kyroform-nexus/
├── README.md
├── requirements.txt
├── main.py                  # Main Streamlit application
│
├── src/kyroform/           # Core ML modules
│   ├── inference.py        # Model inference engine
│   ├── state_manager.py    # Session state management
│   └── utils.py           # API utilities (UniProt, STRING, AlphaFold)
│
├── outputs/
│   ├── figures/            # Visualizations
│   └── models/             # Trained model & embeddings
│
├── notebooks/              # Analysis notebooks
│   └── Kyroformv1.ipynb
│
└── data/                   # Raw datasets
    └── raw/
```

---

## Tech Stack

- **Model**: Heterogeneous GraphSAGE (PyTorch Geometric)
- **Embeddings**: ESM-2 (650M params, 1280-dim)
- **Framework**: PyTorch
- **UI**: Streamlit
- **Visualization**: Plotly, NetworkX

---

## Installation

```bash
pip install -r requirements.txt
streamlit run main.py
```

Open http://localhost:8501

---

## Example Predictions

```
Human Protein    Bacterial Protein    Probability    Prediction
---------------------------------------------------------------
Q5TCU3          A0A0J6C625          0.709          Positive
Q5VV89          A0A0J6C367          0.676          Positive
O95670          A0A0J6C2K2          0.804          Positive
```

---

## Key Features

- **Interactive Explorer**: Select human/bacterial proteins → get interaction probability
- **Network Visualization**: STRING neighbors + predicted edges
- **3D Structure Viewer**: AlphaFold DB integration
- **XAI**: Saliency mapping, latent space analysis
- **Disease Context**: SLE, IBD, RA, T1D gene filtering

---

## Author

**Kaustuv Bhattarai**  
Computer Engineering, Tribhuvan University  
Kathmandu Engineering College

---

## License

MIT
