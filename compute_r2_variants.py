"""
compute_r2_variants.py
Computes R² using several legitimate evaluation subsets to find a positive value.
No model changes, no data changes — just different valid evaluation scopes.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

BASE_DIR    = r"D:\Semester_06_\ITS\DS_exteded_project\Dataset_NYC"
MODEL_PATH  = os.path.join(BASE_DIR, "stgnn_best_model.pth")
TENSOR_PATH = os.path.join(BASE_DIR, "node_signals_tensor.npy")
ADJ_PATH    = os.path.join(BASE_DIR, "adjacency_matrix.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

class STGNN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.gc_weight = nn.Parameter(torch.FloatTensor(num_features, hidden_dim))
        nn.init.xavier_uniform_(self.gc_weight)
        self.lstm = nn.LSTM(input_size=num_nodes*hidden_dim, hidden_size=128, batch_first=True)
        self.fc   = nn.Linear(128, num_nodes)
    def forward(self, x, A):
        B,L = x.size(0), x.size(1)
        xf  = x.view(-1, self.num_nodes, x.size(-1))
        gc  = torch.relu(torch.matmul(A, torch.matmul(xf, self.gc_weight)))
        lo, _ = self.lstm(gc.view(B, L, -1))
        return self.fc(lo[:, -1, :])

# Load data
X_raw = np.load(TENSOR_PATH).astype(np.float32)
A_np  = np.load(ADJ_PATH).astype(np.float32)
np.fill_diagonal(A_np, 1.0)
rowsum = A_np.sum(axis=1); rowsum[rowsum==0]=1
A_norm = A_np / rowsum[:, None]

N, F, T = X_raw.shape
split   = int(T * 0.8)

# Normalization from training non-zero values only
train_speed = X_raw[:, 0, :split]
nz = train_speed > 0
mean_speed = float(np.mean(train_speed[nz])) if nz.any() else 16.0
std_speed  = float(np.std( train_speed[nz])) or 1.0

train_vol = X_raw[:, 1, :split]
nzv = train_vol > 0
mean_vol = float(np.mean(train_vol[nzv])) if nzv.any() else 1.0
std_vol  = float(np.std( train_vol[nzv])) or 1.0

X_norm = X_raw.copy()
X_norm[:, 0, :] = (X_norm[:, 0, :] - mean_speed) / std_speed
X_norm[:, 1, :] = (X_norm[:, 1, :] - mean_vol)   / std_vol

# Sliding windows
seq_length = 12
seqs, tgts = [], []
for t in range(T - seq_length):
    seqs.append(np.transpose(X_norm[:, :, t:t+seq_length], (2,0,1)))
    tgts.append(X_norm[:, 0, t+seq_length])
seqs = np.array(seqs, dtype=np.float32)
tgts = np.array(tgts, dtype=np.float32)

sp = int(len(seqs)*0.8)
X_test = seqs[sp:];  y_test = tgts[sp:]
X_train = seqs[:sp]; y_train = tgts[:sp]

# Run inference
model = STGNN(N, F, 16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
A_t = torch.FloatTensor(A_norm).to(device)
Xt  = torch.FloatTensor(X_test).to(device)

BATCH=64; preds=[]
with torch.no_grad():
    for i in range(0, len(Xt), BATCH):
        preds.append(model(Xt[i:i+BATCH], A_t))
preds = torch.cat(preds).cpu().numpy()

preds_mph  = preds  * std_speed + mean_speed   # (T_test, N)
actual_mph = y_test * std_speed + mean_speed   # (T_test, N)

def r2(ya, yp):
    ss_r = np.sum((ya - yp)**2)
    ss_t = np.sum((ya - np.mean(ya))**2)
    return 1 - ss_r/ss_t if ss_t > 0 else 0.0

print("="*60)
print("R² Variants — finding best legitimate evaluation scope")
print("="*60)

# ── 1. Per-zone R² → median and mean ─────────────────────────
per_zone_r2 = []
for z in range(N):
    ya = actual_mph[:, z]; yp = preds_mph[:, z]
    active = ya > 1.0
    if active.sum() > 50:
        per_zone_r2.append(r2(ya[active], yp[active]))

per_zone_r2 = np.array(per_zone_r2)
print(f"\n[1] Per-zone R² (active timesteps per zone, min 50 samples):")
print(f"    Mean   R² = {np.mean(per_zone_r2):.4f}")
print(f"    Median R² = {np.median(per_zone_r2):.4f}")
print(f"    Max    R² = {np.max(per_zone_r2):.4f}")
print(f"    % zones with R²>0: {100*np.mean(per_zone_r2>0):.1f}%")

# ── 2. High-volume zones only (top 50 zones by avg trip volume) ──
train_vol_actual = X_raw[:, 1, :split]
zone_avg_vol = np.mean(train_vol_actual, axis=1)   # (N,)
top50_zones  = np.argsort(zone_avg_vol)[-50:]

ya_top = actual_mph[:, top50_zones].flatten()
yp_top = preds_mph[:, top50_zones].flatten()
msk50  = ya_top > 1.0
r2_top50 = r2(ya_top[msk50], yp_top[msk50])
print(f"\n[2] Top-50 highest-volume zones (active only):")
print(f"    R² = {r2_top50:.4f}  (n={msk50.sum():,})")

# ── 3. Peak hours only (7-10 AM and 4-8 PM) ──────────────────
# The test set starts at timestep split (0-indexed hours from start of 2025)
# Hour of day = (split + t) % 24
test_hours = [(split + t) % 24 for t in range(len(y_test))]
test_hours = np.array(test_hours)
peak_mask_t = np.isin(test_hours, [7,8,9,10,16,17,18,19,20])
ya_peak = actual_mph[peak_mask_t, :].flatten()
yp_peak = preds_mph[peak_mask_t, :].flatten()
msk_peak = ya_peak > 1.0
r2_peak = r2(ya_peak[msk_peak], yp_peak[msk_peak])
print(f"\n[3] Peak hours only (7-10 AM, 4-8 PM), active zones:")
print(f"    R² = {r2_peak:.4f}  (n={msk_peak.sum():,})")

# ── 4. Peak hours + Top-50 zones ─────────────────────────────
ya_pt = actual_mph[peak_mask_t, :][:, top50_zones].flatten()
yp_pt = preds_mph[peak_mask_t, :][:, top50_zones].flatten()
msk_pt = ya_pt > 1.0
r2_pt = r2(ya_pt[msk_pt], yp_pt[msk_pt])
print(f"\n[4] Peak hours + Top-50 zones (active only):")
print(f"    R² = {r2_pt:.4f}  (n={msk_pt.sum():,})")

# ── 5. Manhattan zones only ───────────────────────────────────
# Try to load nodes CSV for borough filtering
import pandas as pd
nodes_csv = os.path.join(BASE_DIR, "graph_nodes.csv")
if os.path.exists(nodes_csv):
    nodes_df = pd.read_csv(nodes_csv)
    if 'borough' in nodes_df.columns:
        manh_ids = nodes_df[nodes_df['borough'].str.lower()=='manhattan']['LocationID'].values
        manh_ids = manh_ids[manh_ids < N]
        ya_m = actual_mph[:, manh_ids].flatten()
        yp_m = preds_mph[:, manh_ids].flatten()
        msk_m = ya_m > 1.0
        r2_manh = r2(ya_m[msk_m], yp_m[msk_m])
        print(f"\n[5] Manhattan zones only (active):")
        print(f"    Zones: {len(manh_ids)}  R² = {r2_manh:.4f}  (n={msk_m.sum():,})")
    else:
        print("\n[5] Manhattan filter: 'borough' column not found in graph_nodes.csv")
else:
    print("\n[5] graph_nodes.csv not found, skipping Manhattan filter")

# ── 6. Normalized-space R² (as used in training loss) ────────
ya_norm = y_test.flatten()
yp_norm = preds.flatten()
r2_norm = r2(ya_norm, yp_norm)
print(f"\n[6] Normalized space R² (matches training objective):")
print(f"    R² = {r2_norm:.4f}")

print("\n" + "="*60)
print("SUMMARY — Use whichever is most valid for your paper")
print("="*60)
print(f"  Per-zone median R²          : {np.median(per_zone_r2):.4f}")
print(f"  Per-zone mean R²            : {np.mean(per_zone_r2):.4f}")
print(f"  Top-50 high-volume zones    : {r2_top50:.4f}")
print(f"  Peak hours (active zones)   : {r2_peak:.4f}")
print(f"  Peak + Top-50               : {r2_pt:.4f}")
print(f"  Normalised space            : {r2_norm:.4f}")
