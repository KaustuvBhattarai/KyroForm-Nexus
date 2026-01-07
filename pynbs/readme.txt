# KyroForm Interaction Prediction (Google Colab)

This Colab notebook implements a **protein-protein interaction prediction pipeline** using precomputed ESM-2 embeddings and a Heterogeneous Graph Neural Network (HeteroSAGE). It predicts potential interactions between human and bacterial proteins.  

## Features

- Loads precomputed protein embeddings (`ESM-2`)  
- Defines and loads a **HeteroSAGE model** for human-bacterial interaction prediction  
- Provides a `predict_interaction(human_id, bacterial_id)` function to score protein pairs  
- Performs **sample predictions** to demonstrate the workflow  

## Requirements

This notebook installs and uses the following Python packages:

```bash
!pip install torch torch-geometric biopython transformers
It is recommended to run on GPU for faster computation (T4 or higher).
```

## File Structure
esm2_embeddings_1143_proteins.pkl — Precomputed embeddings for 1143 proteins

kyroform_ek.pth — Trained model checkpoint

training_edges_with_labels.csv — Interaction edges used for training

## How to Use
- Open the notebook in Google Colab.

- Ensure GPU runtime is selected (Runtime > Change runtime type > GPU).

- Run the installation cell to install dependencies.

- Load embeddings and the model by running the setup cell.

- Use the predict_interaction(human_id, bacterial_id) function to predict interactions:

python
```
prob = predict_interaction("C9J9G2", "A0A0J6C408")
print(f"Predicted interaction probability: {prob:.4f}")
Run the sample prediction cell to test random protein pairs.
```

Notess
- Some pretrained model weights may not match exactly with the defined architecture. Check the console output for potential warnings.
- Predictions are based on dot-product similarity of embeddings processed through the HeteroSAGE network.
