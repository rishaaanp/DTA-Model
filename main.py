# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("davis_processed_300_add_range.csv")
df = df[['compound_iso_smiles', 'target_sequence', 'affinity']]

# ===============================
# 3. BUILD VOCAB
# ===============================
all_sequences = list(df['compound_iso_smiles']) + list(df['target_sequence'])
chars = set("".join(all_sequences))

char2idx = {c: i+1 for i, c in enumerate(chars)}
char2idx['<pad>'] = 0

vocab_size = len(char2idx)

# ===============================
# 4. SETTINGS
# ===============================
MAX_DRUG_LEN = 100
MAX_PROTEIN_LEN = 300

# ===============================
# 5. ENCODING FUNCTION
# ===============================
def encode(seq, max_len):
    seq = [char2idx.get(c, 0) for c in seq]
    seq = seq[:max_len]
    seq += [0] * (max_len - len(seq))
    return torch.tensor(seq)

# ===============================
# 6. DATASET CLASS
# ===============================
class DTADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        drug = encode(row['compound_iso_smiles'], MAX_DRUG_LEN)
        protein = encode(row['target_sequence'], MAX_PROTEIN_LEN)
        label = torch.tensor(row['affinity'], dtype=torch.float)

        return drug, protein, label

# ===============================
# 7. SPLIT DATA
# ===============================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader = DataLoader(DTADataset(train_df), batch_size=32, shuffle=True)
test_loader = DataLoader(DTADataset(test_df), batch_size=32)

# ===============================
# 8. POSITIONAL ENCODING
# ===============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos][i+1] = math.cos(pos / (10000 ** (i / d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ===============================
# 9. CROSS ATTENTION
# ===============================
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, drug, protein):
        Q = self.query(drug)
        K = self.key(protein)
        V = self.value(protein)

        scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        out = attn @ V
        return out

# ===============================
# 10. MODEL
# ===============================
class DTA_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)

        self.drug_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.protein_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cross_attention = CrossAttention(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, drug, protein):
        drug = self.embedding(drug)
        protein = self.embedding(protein)

        drug = self.pos(drug)
        protein = self.pos(protein)

        drug = self.drug_encoder(drug)
        protein = self.protein_encoder(protein)

        cross_out = self.cross_attention(drug, protein)

        drug_feat = drug.mean(dim=1)
        protein_feat = protein.mean(dim=1)
        cross_feat = cross_out.mean(dim=1)

        combined = torch.cat([drug_feat, protein_feat, cross_feat], dim=1)

        return self.fc(combined)

# ===============================
# 11. TRAINING
# ===============================
model = DTA_Model(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 5

for epoch in range(EPOCHS):
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

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ===============================
# 12. EVALUATION
# ===============================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for drug, protein, label in test_loader:
        drug, protein = drug.to(device), protein.to(device)

        output = model(drug, protein).squeeze().cpu().numpy()

        y_pred.extend(output)
        y_true.extend(label.numpy())

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nFinal Results:")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")