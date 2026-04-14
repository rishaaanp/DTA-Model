# 1. IMPORTS
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CI FUNCTION
def concordance_index(y_true, y_pred):
    n = 0
    h_sum = 0
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

# 3. LOAD DATASET
df = pd.read_csv("davis_processed_300_add_range.csv")
df = df[['compound_iso_smiles', 'target_sequence', 'affinity']]

print("\nDATASET INFO")
print("Total samples:", len(df))
print("Avg drug length:", df['compound_iso_smiles'].apply(len).mean())
print("Avg protein length:", df['target_sequence'].apply(len).mean())

# 4. NORMALIZE TARGET
mean = df['affinity'].mean()
std = df['affinity'].std()
df['affinity'] = (df['affinity'] - mean) / std

# 5. VOCAB
all_sequences = list(df['compound_iso_smiles']) + list(df['target_sequence'])
chars = set("".join(all_sequences))

char2idx = {c: i+1 for i, c in enumerate(chars)}
char2idx['<pad>'] = 0

vocab_size = len(char2idx)

MAX_DRUG_LEN = 80
MAX_PROTEIN_LEN = 150

# 6. ENCODING
def encode(seq, max_len):
    seq = [char2idx.get(c, 0) for c in seq]
    seq = seq[:max_len]
    seq += [0] * (max_len - len(seq))
    return torch.tensor(seq)

# 7. DATASET
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

# 8. SPLIT
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print("\nDATA SPLIT")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_loader = DataLoader(DTADataset(train_df), batch_size=64, shuffle=True)
val_loader = DataLoader(DTADataset(val_df), batch_size=64)
test_loader = DataLoader(DTADataset(test_df), batch_size=64)

# 9. MODEL
class DTA_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.drug_conv1 = nn.Conv1d(embed_dim, 128, 3, padding=1)
        self.drug_conv2 = nn.Conv1d(embed_dim, 128, 5, padding=2)

        self.protein_conv1 = nn.Conv1d(embed_dim, 128, 3, padding=1)
        self.protein_conv2 = nn.Conv1d(embed_dim, 128, 5, padding=2)

        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def encode(self, x, c1, c2):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(c1(x)) + torch.relu(c2(x))
        return x.permute(0, 2, 1)

    def cross_attention(self, d, p):
        Q = self.query(d)
        K = self.key(p)
        V = self.value(p)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (128**0.5), dim=-1)
        return attn @ V

    def forward(self, drug, protein):
        d = self.encode(drug, self.drug_conv1, self.drug_conv2)
        p = self.encode(protein, self.protein_conv1, self.protein_conv2)
        c = self.cross_attention(d, p)

        d_feat = torch.max(d, dim=1).values
        p_feat = torch.max(p, dim=1).values
        c_feat = torch.max(c, dim=1).values

        return self.fc(torch.cat([d_feat, p_feat, c_feat], dim=1))

# 10. TRAINING
model = DTA_Model(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

print("\nMODEL PARAMETERS:", sum(p.numel() for p in model.parameters()))

best_loss = float('inf')
best_epoch = 0

for epoch in range(20):
    model.train()
    total_loss = 0

    for drug, protein, label in train_loader:
        drug, protein, label = drug.to(device), protein.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(drug, protein).squeeze()
        loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for drug, protein, label in val_loader:
            drug, protein, label = drug.to(device), protein.to(device), label.to(device)
            val_loss += ((model(drug, protein).squeeze() - label) ** 2).mean().item()

    print(f"Epoch {epoch+1} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model_davis.pth")

print(f"\nBEST EPOCH: {best_epoch}")

model.load_state_dict(torch.load("best_model_davis.pth"))

# 11. TEST + TIMING
model.eval()
y_true, y_pred = [], []

start_test = time.time()

with torch.no_grad():
    for drug, protein, label in test_loader:
        drug, protein = drug.to(device), protein.to(device)
        pred = model(drug, protein).squeeze().cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(label.numpy())

end_test = time.time()

# De-normalize
y_pred = [p * std + mean for p in y_pred]
y_true = [t * std + mean for t in y_true]

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
ci = concordance_index(y_true, y_pred)

# 12. FINAL OUTPUT
print("\nFINAL MODEL PERFORMANCE")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")
print(f"CI: {ci:.4f}")

# Baseline
baseline_mse = 0.284
baseline_r2 = 0.624
baseline_ci = 0.869

print("\nCOMPARISON WITH BASELINE (Davis)")
print(f"Baseline -> MSE {baseline_mse} | R2 {baseline_r2} | CI {baseline_ci}")
print(f"Your Model -> MSE {mse:.4f} | R2 {r2:.4f} | CI {ci:.4f}")

# Improvements
mse_imp = ((baseline_mse - mse) / baseline_mse) * 100
r2_imp = ((r2 - baseline_r2) / baseline_r2) * 100
ci_imp = ((ci - baseline_ci) / baseline_ci) * 100

print("\nIMPROVEMENT OVER BASELINE")
print(f"MSE Improvement: {mse_imp:.2f}%")
print(f"R2 Improvement: {r2_imp:.2f}%")
print(f"CI Improvement: {ci_imp:.2f}%")

print(f"\nTraining Time: {(time.time() - start_time)/60:.2f} minutes")
print(f"Inference Time: {(end_test - start_test):.2f} seconds")

print("\nMODEL SUMMARY")
print("Deep CNN + Cross Attention + Global Max Pooling")
