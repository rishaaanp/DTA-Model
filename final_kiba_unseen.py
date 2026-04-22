# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# =========================================
# 2. CI FUNCTION
# =========================================
def concordance_index(y_true, y_pred):
    n, h_sum = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
                   (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                    h_sum += 1
                elif y_pred[i] == y_pred[j]:
                    h_sum += 0.5
    return h_sum / n if n > 0 else 0

# =========================================
# 3. LOAD DATA
# =========================================
df = pd.read_csv("kiba_processed_300_add_range.csv")
df = df[['compound_iso_smiles', 'target_sequence', 'affinity']]

print("\n========== DATASET ==========")
print("Total samples:", len(df))

# =========================================
# 4. NORMALIZATION
# =========================================
mean = df['affinity'].mean()
std = df['affinity'].std()
df['affinity'] = (df['affinity'] - mean) / std

# =========================================
# 5. VOCAB
# =========================================
all_sequences = list(df['compound_iso_smiles']) + list(df['target_sequence'])
chars = set("".join(all_sequences))

char2idx = {c: i+1 for i, c in enumerate(chars)}
char2idx['<pad>'] = 0
vocab_size = len(char2idx)

MAX_DRUG_LEN = 80
MAX_PROTEIN_LEN = 150

def encode(seq, max_len):
    seq = [char2idx.get(c, 0) for c in seq][:max_len]
    seq += [0] * (max_len - len(seq))
    return torch.tensor(seq)

# =========================================
# 6. DATASET
# =========================================
class DTADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            encode(row['compound_iso_smiles'], MAX_DRUG_LEN),
            encode(row['target_sequence'], MAX_PROTEIN_LEN),
            torch.tensor(row['affinity'], dtype=torch.float)
        )

# =========================================
# 7. UNSEEN SPLITS
# =========================================
unique_drugs = df['compound_iso_smiles'].unique()
unique_targets = df['target_sequence'].unique()

train_drugs, test_drugs = train_test_split(unique_drugs, test_size=0.2, random_state=42)
train_targets, test_targets = train_test_split(unique_targets, test_size=0.2, random_state=42)

train_df = df[
    df['compound_iso_smiles'].isin(train_drugs) &
    df['target_sequence'].isin(train_targets)
]

test_df_drug = df[
    df['compound_iso_smiles'].isin(test_drugs) &
    df['target_sequence'].isin(train_targets)
]

test_df_target = df[
    df['compound_iso_smiles'].isin(train_drugs) &
    df['target_sequence'].isin(test_targets)
]

test_df_both = df[
    df['compound_iso_smiles'].isin(test_drugs) &
    df['target_sequence'].isin(test_targets)
]

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print("\n========== DATA SPLIT ==========")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Unseen Drug: {len(test_df_drug)}")
print(f"Unseen Target: {len(test_df_target)}")
print(f"Both Unseen: {len(test_df_both)}")

train_loader = DataLoader(DTADataset(train_df), batch_size=128, shuffle=True)
val_loader = DataLoader(DTADataset(val_df), batch_size=128)

# =========================================
# 8. MODEL
# =========================================
class ImprovedDTA(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.drug_conv3 = nn.Conv1d(embed_dim, 128, 3, padding=1)
        self.drug_conv5 = nn.Conv1d(embed_dim, 128, 5, padding=2)
        self.drug_conv7 = nn.Conv1d(embed_dim, 128, 7, padding=3)

        self.protein_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU()
        )

        self.protein_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

        self.drug_proj = nn.Linear(128, 256)
        self.attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, drug, protein):
        emb_d = self.embedding(drug)
        d = emb_d.permute(0, 2, 1)

        d = (torch.relu(self.drug_conv3(d)) +
             torch.relu(self.drug_conv5(d)) +
             torch.relu(self.drug_conv7(d))) / 3

        d = d.permute(0, 2, 1) + emb_d

        p = self.embedding(protein).permute(0, 2, 1)
        p = self.protein_conv(p).permute(0, 2, 1)
        p, _ = self.protein_lstm(p)

        d_proj = self.drug_proj(d)
        attn_out, _ = self.attn(p, d_proj, d_proj)
        attn_out = attn_out + p

        d_feat = torch.mean(d, dim=1) + torch.max(d, dim=1).values
        p_feat = torch.mean(attn_out, dim=1) + torch.max(attn_out, dim=1).values

        return self.fc(torch.cat([d_feat, p_feat], dim=1))

# =========================================
# 9. TRAINING
# =========================================
model = ImprovedDTA(vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_fn = nn.SmoothL1Loss()

best_loss = float('inf')

for epoch in range(35):
    model.train()
    train_loss = 0

    for drug, protein, label in train_loader:
        drug, protein, label = drug.to(device), protein.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(drug, protein).squeeze()
        loss = loss_fn(out, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for drug, protein, label in val_loader:
            drug, protein, label = drug.to(device), protein.to(device), label.to(device)
            val_loss += ((model(drug, protein).squeeze() - label) ** 2).mean().item()

    print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_kiba_model.pth")

model.load_state_dict(torch.load("best_kiba_model.pth"))

# =========================================
# 10. EVALUATION
# =========================================
def evaluate(df):
    loader = DataLoader(DTADataset(df), batch_size=128)
    y_true, y_pred = [], []

    with torch.no_grad():
        for drug, protein, label in loader:
            drug, protein = drug.to(device), protein.to(device)
            pred = model(drug, protein).squeeze().cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(label.numpy())

    y_pred = [p * std + mean for p in y_pred]
    y_true = [t * std + mean for t in y_true]

    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred), concordance_index(y_true, y_pred)

# =========================================
# 11. FINAL RESULTS
# =========================================
print("\n========== FINAL PERFORMANCE ==========")

mse_d, r2_d, ci_d = evaluate(test_df_drug)
mse_t, r2_t, ci_t = evaluate(test_df_target)
mse_b, r2_b, ci_b = evaluate(test_df_both)

print("\nUNSEEN DRUG")
print(f"MSE: {mse_d:.4f} | R²: {r2_d:.4f} | CI: {ci_d:.4f}")

print("\nUNSEEN TARGET")
print(f"MSE: {mse_t:.4f} | R²: {r2_t:.4f} | CI: {ci_t:.4f}")

print("\nBOTH UNSEEN")
print(f"MSE: {mse_b:.4f} | R²: {r2_b:.4f} | CI: {ci_b:.4f}")

print(f"\nTOTAL TIME: {(time.time()-start_time)/60:.2f} mins")