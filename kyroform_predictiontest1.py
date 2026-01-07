import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import random

# ================== LOAD EVERYTHING ==================
# Change these paths if needed
EMBEDDINGS_PATH = "esm2_embeddings_1143_proteins.pkl"
MODEL_PATH = "kyroform_ek.pth"  # or your .pth name
EDGES_PATH = "training_edges_with_labels.csv"   # for known positives reference

print("Loading precomputed ESM-2 embeddings...")
with open(EMBEDDINGS_PATH, 'rb') as f:
    embeds = pickle.load(f)
print(f"Loaded embeddings for {len(embeds)} proteins")

print("Loading trained model...")
class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.h_conv1 = SAGEConv((-1, -1), hidden)
        self.h_conv2 = SAGEConv(hidden, hidden)
        self.b_conv1 = SAGEConv((-1, -1), hidden)
        self.b_conv2 = SAGEConv(hidden, hidden)

    def forward(self, x_dict, edge_index_dict):
        edge = edge_index_dict[('human', 'interacts', 'bacterial')]
        rev = edge.flip(0)
        
        h = F.relu(self.h_conv1(x_dict['human'], rev))
        h = F.relu(self.h_conv2(h, rev))
        
        b = F.relu(self.b_conv1(x_dict['bacterial'], edge))
        b = F.relu(self.b_conv2(b, edge))
        
        return {'human': h, 'bacterial': b}

model = HeteroSAGE(hidden_channels=256)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("Model loaded successfully")

# Load edges to get all known proteins
df_edges = pd.read_csv(EDGES_PATH)
all_human = df_edges['human'].unique().tolist()
all_bact = df_edges['bacterial'].unique().tolist()

# Filter to those with embeddings
all_human = [p for p in all_human if p in embeds]
all_bact = [p for p in all_bact if p in embeds]

print(f"Available for prediction: {len(all_human)} human, {len(all_bact)} bacterial proteins")

# ================== PREDICTION FUNCTION ==================
def predict_interaction(human_id, bacterial_id):
    if human_id not in embeds:
        return f"Human protein {human_id} not in embeddings"
    if bacterial_id not in embeds:
        return f"Bacterial protein {bacterial_id} not in embeddings"
    
    # Build mini graph with just these two nodes
    data = HeteroData()
    data['human'].x = torch.tensor(embeds[human_id]).unsqueeze(0)
    data['bacterial'].x = torch.tensor(embeds[bacterial_id]).unsqueeze(0)
    
    # Dummy edge_index (not used for single pair prediction)
    data['human', 'interacts', 'bacterial'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    with torch.no_grad():
        z = model(data.x_dict, data.edge_index_dict)
        score = (z['human'][0] * z['bacterial'][0]).sum().item()
        prob = torch.sigmoid(torch.tensor(score)).item()
    
    return prob

# ================== TEST RANDOM PAIRS ==================
print("\n" + "="*60)
print("TESTING 10 RANDOM PAIRS (human vs bacterial)")
print("="*60)

random.seed(42)
test_pairs = []
for _ in range(10):
    h = random.choice(all_human)
    b = random.choice(all_bact)
    prob = predict_interaction(h, b)
    status = "Likely interacts" if prob > 0.7 else "Unlikely"
    print(f"{h}  —  {b}")
    print(f"   → Predicted probability: {prob:.4f} ({status})\n")

# ================== TEST SPECIFIC PAIRS ==================
print("EXAMPLE: Test known positive (should be high)")
# Pick one known positive from your data
known_pos = df_edges[df_edges['label'] == 1].sample(1)
h_pos = known_pos['human'].values[0]
b_pos = known_pos['bacterial'].values[0]
print(f"Known positive: {h_pos} — {b_pos} → {predict_interaction(h_pos, b_pos):.4f}")

print("\nKyroform AI prediction ready!")
print("Use predict_interaction('P12345', 'A0A0J6XXXX') for any pair.")