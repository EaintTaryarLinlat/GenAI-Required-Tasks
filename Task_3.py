"""
Student Name: [Eaint Taryar Linlat]
Key Summary – Task 3: End-to-End Binary Classification — Loan Default Prediction
In this task, I built a complete machine learning pipeline in PyTorch to predict credit risk using the German Credit Dataset (1,000 bank customers, 20 features, binary target: Good/Bad credit).
What I did:

Loaded the dataset directly from the UCI repository — space-separated with no header, requiring manual column name assignment
Preprocessed the data: remapped the target (1→1 Good, 2→0 Bad) for BCELoss compatibility, one-hot encoded categorical columns with pd.get_dummies, applied StandardScaler fitted only on training data to prevent data leakage, and split 80/20 train/test
Built a 3-hidden-layer MLP (LoanDefaultMLP) with layers 64→32→16→1, using BatchNorm, ReLU, Dropout (0.3), and a final Sigmoid output
Trained for 100 epochs using BCELoss, Adam (lr=0.001) with L2 weight decay, and a StepLR scheduler that halves the learning rate every 25 epochs
Evaluated using accuracy, precision, recall, F1-score, and a confusion matrix

Key lessons:

Data leakage prevention — fitting the scaler only on training data is critical; fitting on the full dataset would artificially inflate test performance by leaking test statistics into training
Target remapping matters — BCELoss requires labels in {0, 1}; the raw dataset uses {1, 2} so the remap step is essential or the loss function produces incorrect gradients
Dropout + BatchNorm — with only 800 training samples, regularisation is essential to prevent overfitting on a small tabular dataset
LR scheduling — halving the learning rate periodically allows the model to make large updates early and fine-tune precisely later, improving final convergence.
"""

# 0.  Imports
# ═══════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load Data
# ═══════════════════════════════════════════════════════════════════════════
# Column names (no header in the raw file)
col_names = [
    "Status of existing checking account",
    "Duration in month",
    "Credit history",
    "Purpose",
    "Credit amount",
    "Savings account/bonds",
    "Present employment since",
    "Installment rate in percentage of disposable income",
    "Personal status and sex",
    "Other debtors / guarantors",
    "Present residence since",
    "Property",
    "Age in years",
    "Other installment plans",
    "Housing",
    "Number of existing credits at this bank",
    "Job",
    "Number of people being liable to provide maintenance for",
    "Telephone",
    "foreign worker",
    "Creditability"
]

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "statlog/german/german.data")

# sep=" " because the file is space-separated; header=None because there is none
df = pd.read_csv(url, sep=" ", header=None, names=col_names)

print("─" * 55)
print("STEP 1 — Data Loaded")
print("─" * 55)
print(f"Shape      : {df.shape}  (rows × columns)")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
print(f"\nCreditability counts (raw):\n{df['Creditability'].value_counts()}")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Preprocess the Data
# ═══════════════════════════════════════════════════════════════════════════

# ── 2a. Separate features and target
X = df.drop("Creditability", axis=1)
y = df["Creditability"]

# ── 2b. Remap target: 1 (Good) → 1,  2 (Bad) → 0 
# BCELoss expects values in {0, 1}
y = y.replace({1: 1, 2: 0})
print("\n" + "─" * 55)
print("STEP 2 — Preprocessing")
print("─" * 55)
print(f"Target after remapping:\n{y.value_counts()}")

# ── 2c. One-hot encode all categorical (object) columns
non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
print(f"\nCategorical columns ({len(non_numeric_cols)}):\n{non_numeric_cols}")

X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
print(f"\nShape after one-hot encoding : {X.shape}")

# ── 2d. Train / Test split  (80 / 20) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size : {X_train.shape[0]} samples")
print(f"Test  size : {X_test.shape[0]} samples")

# ── 2e. Standardise features
# IMPORTANT: fit ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform on train
X_test_sc  = scaler.transform(X_test)        # transform only on test

print(f"\nFeature mean after scaling (train, should be ~0): "
      f"{X_train_sc.mean():.4f}")
print(f"Feature std  after scaling (train, should be ~1): "
      f"{X_train_sc.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Create PyTorch Tensors and DataLoaders
# ═══════════════════════════════════════════════════════════════════════════
X_train_t = torch.FloatTensor(X_train_sc)
X_test_t  = torch.FloatTensor(X_test_sc)
y_train_t = torch.FloatTensor(y_train.values).view(-1, 1)
y_test_t  = torch.FloatTensor(y_test.values).view(-1, 1)

print("\n" + "─" * 55)
print("STEP 3 — Tensors & DataLoaders")
print("─" * 55)
print(f"X_train tensor shape : {X_train_t.shape}")
print(f"y_train tensor shape : {y_train_t.shape}")

BATCH_SIZE = 32

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches : {len(train_loader)}  (batch_size={BATCH_SIZE})")
print(f"Test  batches : {len(test_loader)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Build the MLP
# ═══════════════════════════════════════════════════════════════════════════
class LoanDefaultMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary credit-risk classification.

    Architecture
    ─────────────
    Input  →  Hidden-1 (64)  →  BN → ReLU → Dropout
           →  Hidden-2 (32)  →  BN → ReLU → Dropout
           →  Hidden-3 (16)  →  ReLU
           →  Output  (1)    →  Sigmoid
    """
    def __init__(self, input_dim: int, dropout_p: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # ── Hidden layer 1 ──────────────────────────────────────────
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            # ── Hidden layer 2 ──────────────────────────────────────────
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            # ── Hidden layer 3 ──────────────────────────────────────────
            nn.Linear(32, 16),
            nn.ReLU(),

            # ── Output layer ────────────────────────────────────────────
            nn.Linear(16, 1),
            nn.Sigmoid()          # output ∈ (0, 1) → probability of "Good"
        )

    def forward(self, x):
        return self.network(x)


# Detect GPU; fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + "─" * 55)
print("STEP 4 — Model Architecture")
print("─" * 55)

input_dim = X_train_t.shape[1]
model = LoanDefaultMLP(input_dim=input_dim).to(device)
print(model)
print(f"\nInput dimension : {input_dim} features")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters : {total_params:,}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Loss, Optimiser and Learning-Rate Scheduler
# ═══════════════════════════════════════════════════════════════════════════
criterion = nn.BCELoss()                          # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(),
                       lr=0.001,
                       weight_decay=1e-4)         # L2 regularisation
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=25,
                                      gamma=0.5)  # halve LR every 25 epochs


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Training Loop
# ═══════════════════════════════════════════════════════════════════════════
NUM_EPOCHS  = 100          # well above the required minimum of 50
LOG_EVERY   = 10           # print a summary every 10 epochs

train_losses = []
test_accuracies = []

print("\n" + "─" * 55)
print("STEP 5 — Training")
print("─" * 55)
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc (%)':>13}  {'LR':>12}")
print("─" * 55)

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Training phase ────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        preds = model(batch_X)              # forward pass
        loss  = criterion(preds, batch_y)   # compute BCE loss

        optimizer.zero_grad()               # clear old gradients
        loss.backward()                     # backpropagate
        optimizer.step()                    # update weights

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # ── Evaluation phase ──────────────────────────────────────────────────
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            probs     = model(batch_X)
            predicted = (probs >= 0.5).float()   # threshold at 0.5
            total    += batch_y.size(0)
            correct  += (predicted == batch_y).sum().item()

    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)

    scheduler.step()   # update learning rate

    if epoch % LOG_EVERY == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6}  {epoch_loss:>12.4f}  {test_acc:>12.2f}%  "
              f"{current_lr:>12.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Final Evaluation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 55)
print("FINAL EVALUATION ON TEST SET")
print("═" * 55)

model.eval()
all_probs  = []
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        probs   = model(batch_X).cpu()
        preds   = (probs >= 0.5).float()
        all_probs.extend(probs.squeeze().tolist())
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(batch_y.squeeze().tolist())

all_preds  = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

final_acc = (all_preds == all_labels).float().mean().item() * 100

# ── Confusion matrix counts ───────────────────────────────────────────────
TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
TN = ((all_preds == 0) & (all_labels == 0)).sum().item()
FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
FN = ((all_preds == 0) & (all_labels == 1)).sum().item()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)

print(f"\nTest Accuracy  : {final_acc:.2f}%")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"  {'':12}  Predicted 1  Predicted 0")
print(f"  {'Actual 1':12}  {TP:>11}  {FN:>11}")
print(f"  {'Actual 0':12}  {FP:>11}  {TN:>11}")

# ── Predicted vs Actual table (first 20 test samples) ────────────────────
print("\n" + "─" * 40)
print("Predicted vs Actual (first 20 test samples):")
print("─" * 40)
results_df = pd.DataFrame({
    "Actual"   : [int(v) for v in all_labels[:20].tolist()],
    "Predicted": [int(v) for v in all_preds[:20].tolist()],
    "Prob(Good)": [f"{p:.3f}" for p in all_probs[:20]]
})
print(results_df.to_string(index=True))
print("─" * 40)
print("\n✓  Task 3 complete.")