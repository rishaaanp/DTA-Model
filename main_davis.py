# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("davis_processed_300_add_range.csv")
df = df[['compound_iso_smiles', 'target_sequence', 'affinity']]

# ===============================
# 3. NORMALIZE TARGET
# ===============================
mean = df['affinity'].mean()
std = df['affinity'].std()
df['affinity'] = (df['affinity'] - mean) / std

# ===============================
# 4. BUILD VOCAB
# ===============================
all_sequences = list(df['compound_iso_smiles']) + list(df['target_sequence'])
chars = set("".join(all_sequences))

char2idx = {c: i+1 for i, c in enumerate(chars)}
char2idx['<pad>'] = 0

vocab_size = len(char2idx)

# ===============================
# 5. SETTINGS
# ===============================
MAX_DRUG_LEN = 80
MAX_PROTEIN_LEN = 150

# ===============================
# 6. ENCODING
# ===============================
def encode(seq, max_len):
    seq = [char2idx.get(c, 0) for c in seq]
    seq = seq[:max_len]
    seq += [0] * (max_len - len(seq))
    return torch.tensor(seq)

# ===============================
# 7. DATASET CLASS
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
# 8. SPLIT DATA
# ===============================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader = DataLoader(DTADataset(train_df), batch_size=64, shuffle=True)
test_loader = DataLoader(DTADataset(test_df), batch_size=64)

# ===============================
# 9. MODEL 
# ===============================
class DTA_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Multi-kernel CNN
        self.drug_conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.drug_conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)

        self.protein_conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.protein_conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

        # Cross Attention
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)

        # Fully Connected
        self.bn = nn.BatchNorm1d(256)

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            self.bn,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def encode(self, x, conv1, conv2):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x1 = self.relu(conv1(x))
        x2 = self.relu(conv2(x))

        x = x1 + x2

        x = x.permute(0, 2, 1)
        return x

    def cross_attention(self, drug, protein):
        Q = self.query(drug)
        K = self.key(protein)
        V = self.value(protein)

        scores = Q @ K.transpose(-2, -1) / (128 ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        return attn @ V

    def forward(self, drug, protein):
        drug = self.encode(drug, self.drug_conv1, self.drug_conv2)
        protein = self.encode(protein, self.protein_conv1, self.protein_conv2)

        cross = self.cross_attention(drug, protein)

        #  Global Max Pooling
        drug_feat = torch.max(drug, dim=1).values
        protein_feat = torch.max(protein, dim=1).values
        cross_feat = torch.max(cross, dim=1).values

        combined = torch.cat([drug_feat, protein_feat, cross_feat], dim=1)

        return self.fc(combined)

# ===============================
# 10. TRAINING
# ===============================
model = DTA_Model(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()
 
EPOCHS = 20

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
# 11. EVALUATION
# ===============================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for drug, protein, label in test_loader:
        drug, protein = drug.to(device), protein.to(device)

        output = model(drug, protein).squeeze().cpu().numpy()

        y_pred.extend(output)
        y_true.extend(label.numpy())

# De-normalize
y_pred = [p * std + mean for p in y_pred]
y_true = [t * std + mean for t in y_true]

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n FINAL RESULTS ")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")
