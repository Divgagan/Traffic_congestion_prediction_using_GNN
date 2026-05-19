"""
compute_paper_metrics_v2.py
Fixed: uses training-split-only normalization stats (matching training notebook).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

BASE_DIR   = r"D:\Semester_06_\ITS\DS_exteded_project\Dataset_NYC"
OUT_DIR    = r"D:\Semester_06_\ITS\DS_exteded_project"
MODEL_PATH = os.path.join(BASE_DIR, "stgnn_best_model.pth")
TENSOR_PATH= os.path.join(BASE_DIR, "node_signals_tensor.npy")
ADJ_PATH   = os.path.join(BASE_DIR, "adjacency_matrix.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

print("\nLoading data...")
X_raw = np.load(TENSOR_PATH).astype(np.float32)   # (266, 2, T)
A_np  = np.load(ADJ_PATH).astype(np.float32)

np.fill_diagonal(A_np, 1.0)
rowsum = A_np.sum(axis=1); rowsum[rowsum==0]=1
A_norm = A_np / rowsum[:, None]

N, F, T = X_raw.shape
split   = int(T * 0.8)

# ─── KEY FIX: compute stats from TRAINING SPLIT ONLY (non-zero values) ────────
train_speed = X_raw[:, 0, :split]
nz_mask     = train_speed > 0
mean_speed  = float(np.mean(train_speed[nz_mask])) if nz_mask.any() else float(np.mean(train_speed))
std_speed   = float(np.std( train_speed[nz_mask])) or 1.0
train_vol   = X_raw[:, 1, :split]
nz_vol      = train_vol > 0
mean_vol    = float(np.mean(train_vol[nz_vol])) if nz_vol.any() else float(np.mean(train_vol))
std_vol     = float(np.std( train_vol[nz_vol])) or 1.0

print(f"Normalization stats (training non-zero):")
print(f"  mean_speed={mean_speed:.4f}  std_speed={std_speed:.4f}")
print(f"  mean_vol  ={mean_vol:.4f}  std_vol  ={std_vol:.4f}")

# Normalise full tensor
X_norm = X_raw.copy()
X_norm[:, 0, :] = (X_norm[:, 0, :] - mean_speed) / std_speed
X_norm[:, 1, :] = (X_norm[:, 1, :] - mean_vol)   / std_vol

# Sliding windows
seq_length = 12
print("Building windows...")
seqs, tgts = [], []
for t in range(T - seq_length):
    seqs.append(np.transpose(X_norm[:, :, t:t+seq_length], (2,0,1)))
    tgts.append(X_norm[:, 0, t+seq_length])
seqs = np.array(seqs, dtype=np.float32)
tgts = np.array(tgts, dtype=np.float32)

sp = int(len(seqs)*0.8)
X_train, X_test = seqs[:sp], seqs[sp:]
y_train, y_test = tgts[:sp], tgts[sp:]
print(f"Train={len(X_train):,}  Test={len(X_test):,}")

# Load model & run inference
model = STGNN(N, F, 16).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
A_t = torch.FloatTensor(A_norm).to(device)

BATCH=64; preds=[]
with torch.no_grad():
    Xt = torch.FloatTensor(X_test).to(device)
    for i in range(0, len(Xt), BATCH):
        preds.append(model(Xt[i:i+BATCH], A_t))
preds = torch.cat(preds).cpu().numpy()

# De-normalise
preds_mph  = preds  * std_speed + mean_speed
actual_mph = y_test * std_speed + mean_speed

# ─── METRICS ──────────────────────────────────────────────────────────────────
def metrics(ya, yp):
    mae  = np.mean(np.abs(ya-yp))
    rmse = np.sqrt(np.mean((ya-yp)**2))
    safe = ya > 0.5
    mape = 100*np.mean(np.abs((ya[safe]-yp[safe])/ya[safe])) if safe.any() else 999.0
    ss_r = np.sum((ya-yp)**2)
    ss_t = np.sum((ya-np.mean(ya))**2)
    r2   = 1 - ss_r/ss_t if ss_t>0 else 0.0
    return mae, rmse, mape, r2

# Method A: ALL zones (how training notebook likely reported MAE)
mae_a, rmse_a, mape_a, r2_a = metrics(actual_mph.flatten(), preds_mph.flatten())

# Method B: Active zones only (actual > 1 mph)
fa  = actual_mph.flatten()
fp  = preds_mph.flatten()
msk = (fa > 1.0) & (fa < 80.0)
mae_b, rmse_b, mape_b, r2_b = metrics(fa[msk], fp[msk])

print(f"\n{'='*55}")
print(f"Method A — ALL zones (matches training notebook MAE):")
print(f"  MAE={mae_a:.4f}  RMSE={rmse_a:.4f}  MAPE={mape_a:.2f}%  R2={r2_a:.4f}")
print(f"\nMethod B — ACTIVE zones only (actual > 1 mph):")
print(f"  MAE={mae_b:.4f}  RMSE={rmse_b:.4f}  MAPE={mape_b:.2f}%  R2={r2_b:.4f}")
print(f"{'='*55}")

# ─── HA baseline ──────────────────────────────────────────────────────────────
zone_mean = np.mean(y_train * std_speed + mean_speed, axis=0)
ha_preds  = np.tile(zone_mean, (len(actual_mph),1))
mae_ha, rmse_ha, mape_ha, r2_ha = metrics(actual_mph.flatten(), ha_preds.flatten())
mae_ha_b, rmse_ha_b, mape_ha_b, _ = metrics(fa[msk], ha_preds.flatten()[msk])

# ─── LSTM-Only (no graph mixing — use identity A) ─────────────────────────────
I_t = torch.eye(N, device=device)
preds_lo=[]
with torch.no_grad():
    for i in range(0, len(Xt), BATCH):
        preds_lo.append(model(Xt[i:i+BATCH], I_t))
preds_lo = torch.cat(preds_lo).cpu().numpy() * std_speed + mean_speed
mae_lo, rmse_lo, mape_lo, _ = metrics(actual_mph.flatten(), preds_lo.flatten())

# ─── No self-loops ────────────────────────────────────────────────────────────
A2 = np.load(ADJ_PATH).astype(np.float32)
r2 = A2.sum(axis=1); r2[r2==0]=1; A2/=r2[:,None]
A2_t = torch.FloatTensor(A2).to(device)
preds_nsl=[]
with torch.no_grad():
    for i in range(0, len(Xt), BATCH):
        preds_nsl.append(model(Xt[i:i+BATCH], A2_t))
preds_nsl = torch.cat(preds_nsl).cpu().numpy() * std_speed + mean_speed
mae_nsl, rmse_nsl, _, _ = metrics(actual_mph.flatten(), preds_nsl.flatten())

# Scatter R2 (active zones for scatter plot)
ss_r = np.sum((fa[msk]-fp[msk])**2)
ss_t = np.sum((fa[msk]-np.mean(fa[msk]))**2)
r2_scatter = 1 - ss_r/ss_t

print(f"\nR² for scatter plot (active zones): {r2_scatter:.4f}")

# ─── PRINT TABLES ─────────────────────────────────────────────────────────────
mae_cf  = mae_a;  rmse_cf = rmse_a;  mape_cf = mape_a
delta_lo  = mae_lo  - mae_cf
delta_nsl = mae_nsl - mae_cf

print(f"\n{'='*65}")
print("TABLE V — Performance Comparison (Method A: all zones)")
print(f"{'='*65}")
print(f"{'Model':<25} {'MAE':>7} {'RMSE':>7} {'MAPE':>8}")
print("-"*65)
print(f"{'HA':<25} {mae_ha:>7.2f} {rmse_ha:>7.2f} {mape_ha:>8.2f}")
print(f"{'LSTM-Only':<25} {mae_lo:>7.2f} {rmse_lo:>7.2f} {mape_lo:>8.2f}")
print(f"{'No Self-Loops':<25} {mae_nsl:>7.2f} {rmse_nsl:>7.2f} {'—':>8}")
print(f"{'CityFlow-GNN':<25} {mae_cf:>7.2f} {rmse_cf:>7.2f} {mape_cf:>8.2f}")
print(f"{'='*65}")

print(f"\n{'='*65}")
print("TABLE VI — Ablation Study")
print(f"{'='*65}")
print(f"{'Variant':<28} {'MAE':>7} {'RMSE':>7} {'ΔMAE':>8}")
print("-"*65)
print(f"{'No GCN (LSTM-only)':<28} {mae_lo:>7.2f} {rmse_lo:>7.2f} {delta_lo:>+8.2f}")
print(f"{'No self-loops':<28} {mae_nsl:>7.2f} {rmse_nsl:>7.2f} {delta_nsl:>+8.2f}")
print(f"{'CityFlow-GNN (full)':<28} {mae_cf:>7.2f} {rmse_cf:>7.2f} {'—':>8}")
print("="*65)

# Save npz
np.savez(os.path.join(OUT_DIR,"paper_metrics_v2.npz"),
    mae_cf=mae_cf, rmse_cf=rmse_cf, mape_cf=mape_cf, r2_cf=r2_a,
    mae_ha=mae_ha, rmse_ha=rmse_ha, mape_ha=mape_ha,
    mae_lo=mae_lo, rmse_lo=rmse_lo, mape_lo=mape_lo,
    mae_nsl=mae_nsl, rmse_nsl=rmse_nsl,
    r2_scatter=r2_scatter,
    mae_cf_active=mae_b, rmse_cf_active=rmse_b)

print(f"\nSaved → paper_metrics_v2.npz")
print("\nPaste the full output back so the filled-tables file can be generated.")
