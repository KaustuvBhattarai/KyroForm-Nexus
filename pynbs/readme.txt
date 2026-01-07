## ⚠️ Notes
* **Model Weights**: Some pretrained model weights may not match exactly with the defined architecture. Check the console output for potential warnings regarding unexpected or missing keys.
* **Inference Mechanism**: Predictions are calculated based on the dot-product similarity of protein embeddings after they have been processed through the HeteroSAGE network.
* **Data Scale**: The current setup is optimized for the 1,143 proteins included in the precomputed embeddings file.
