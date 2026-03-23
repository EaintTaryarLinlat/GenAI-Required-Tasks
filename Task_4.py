"""
Student Name: [Eaint Taryar Linlat]
 Task 4: Image Classification with an MLP on Fashion-MNIST
In this task, I trained a Multi-Layer Perceptron to classify 10 categories of clothing images using the Fashion-MNIST dataset — a harder version of MNIST because clothing items look more similar to each other than handwritten digits do.
What I did:

Loaded Fashion-MNIST using torchvision.datasets.FashionMNIST with normalisation (mean=0.2860, std=0.3530, pre-computed from the training set)
Built a 3-hidden-layer MLP (784→256→128→64→10) with BatchNorm, ReLU, and Dropout — no Sigmoid on the output since CrossEntropyLoss applies Softmax internally
Trained for 10 epochs using CrossEntropyLoss and Adam with L2 weight decay, plus a ReduceLROnPlateau scheduler that halves the learning rate if accuracy stops improving for 3 epochs
Evaluated overall accuracy and per-class accuracy across all 10,000 test images
Visualised both correct (green border) and incorrect (red border) predictions with actual vs predicted labels

Key lessons:

CrossEntropyLoss vs BCELoss, multi-class classification uses CrossEntropyLoss which combines LogSoftmax and NLLLoss internally, so raw logits go in — no activation needed on the output layer
Visually similar classes cause most errors — Shirt, T-shirt, and Pullover are frequently confused because they share similar shapes in 28×28 greyscale
Same architecture, different task — switching from MNIST to Fashion-MNIST required changing only one line (datasets.MNIST → datasets.FashionMNIST), which shows how modular and reusable PyTorch pipelines are
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Imports
# ═══════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Class Labels
#     Fashion-MNIST has 10 clothing categories (same index order as the
#     dataset's internal label encoding 0-9).
# ═══════════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "T-shirt/top",   # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot"     # 9
]

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Device
# ═══════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Load Data
#     Key change from MNIST tutorial: datasets.MNIST → datasets.FashionMNIST
#     Everything else (transforms, DataLoader setup) is identical.
# ═══════════════════════════════════════════════════════════════════════════

# ToTensor()    : converts PIL image → Float tensor, scales pixels to [0, 1]
# Normalize()   : shifts to zero-mean / unit-std using Fashion-MNIST statistics
#                 mean=0.2860, std=0.3530  (pre-computed from the training set)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root      = "./data",
    train     = True,
    transform = transform,
    download  = True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root      = "./data",
    train     = False,
    transform = transform,
    download  = True
)

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print("\n" + "─" * 55)
print("STEP 1 — Data Loaded")
print("─" * 55)
print(f"Training samples  : {len(train_dataset):,}")
print(f"Test samples      : {len(test_dataset):,}")
print(f"Training batches  : {len(train_loader)}")
print(f"Test batches      : {len(test_loader)}")

# ─── quick shape sanity check ───────────────────────────────────────────────
sample_images, sample_labels = next(iter(train_loader))
print(f"Image batch shape : {sample_images.shape}  → [N, C, H, W]")
print(f"Label batch shape : {sample_labels.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Visualise Sample Images from the Training Set
# ═══════════════════════════════════════════════════════════════════════════
def imshow_grid(images, labels, title="Sample Fashion-MNIST Images"):
    """Display a row of images with their class names."""
    fig, axes = plt.subplots(1, len(images), figsize=(14, 2.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    for ax, img, lbl in zip(axes, images, labels):
        # un-normalise: pixel = pixel * std + mean
        img_np = img.squeeze().numpy() * 0.3530 + 0.2860
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np, cmap="gray")
        ax.set_title(CLASS_NAMES[lbl], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("\n" + "─" * 55)
print("STEP 2 — Sample Training Images")
print("─" * 55)
imshow_grid(sample_images[:8], sample_labels[:8])
plt.savefig("sample_images.png", dpi=100, bbox_inches="tight")
print("Saved: sample_images.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Define the MLP Model
#     Architecture matches the MNIST tutorial ImageMLP.
#     Input  : 28 × 28 = 784 pixels (flattened inside forward())
#     Hidden : 784 → 256 → 128 → 64
#     Output : 64 → 10 logits  (one per class)
#     No Softmax here — nn.CrossEntropyLoss applies it internally.
# ═══════════════════════════════════════════════════════════════════════════
class FashionMLP(nn.Module):
    """
    Multi-Layer Perceptron for Fashion-MNIST classification.

    Layer sizes deliberately wider than the tutorial's basic version
    (256 / 128 / 64) to give the model more capacity for the harder
    Fashion-MNIST task.  Dropout regularises against overfitting.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        self.flatten = nn.Flatten()          # [N,1,28,28] → [N,784]

        self.network = nn.Sequential(
            # ── Hidden layer 1 ──────────────────────────────────────────
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            # ── Hidden layer 2 ──────────────────────────────────────────
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            # ── Hidden layer 3 ──────────────────────────────────────────
            nn.Linear(128, 64),
            nn.ReLU(),

            # ── Output layer (raw logits, no activation) ─────────────────
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = FashionMLP().to(device)

print("\n" + "─" * 55)
print("STEP 3 — Model Architecture")
print("─" * 55)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Loss Function and Optimiser
#     CrossEntropyLoss = LogSoftmax + NLLLoss (handles multi-class correctly)
#     Adam with weight decay for L2 regularisation
# ═══════════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Reduce LR by 50 % if test accuracy plateaus for 3 consecutive epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# ═══════════════════════════════════════════════════════════════════════════
# 7.  Training Loop  (10 epochs as required)
# ═══════════════════════════════════════════════════════════════════════════
NUM_EPOCHS = 10

train_loss_history = []
test_acc_history   = []

print("\n" + "─" * 55)
print("STEP 4 — Training")
print("─" * 55)
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc (%)':>13}")
print("─" * 40)

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Training phase ────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)            # forward pass → raw scores
        loss   = criterion(logits, labels)

        optimizer.zero_grad()             # reset gradients
        loss.backward()                   # backpropagate
        optimizer.step()                  # update weights

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)

    # ── Evaluation phase ──────────────────────────────────────────────────
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits    = model(images)
            _, preds  = torch.max(logits, dim=1)   # class with highest logit
            total    += labels.size(0)
            correct  += (preds == labels).sum().item()

    test_acc = 100.0 * correct / total
    test_acc_history.append(test_acc)

    scheduler.step(test_acc)             # adjust LR if accuracy plateaus
    print(f"{epoch:>6}  {epoch_loss:>12.4f}  {test_acc:>12.2f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  Plot Training Curves
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Fashion-MNIST MLP — Training Curves", fontsize=13,
             fontweight="bold")

axes[0].plot(range(1, NUM_EPOCHS + 1), train_loss_history,
             marker="o", color="steelblue", linewidth=2)
axes[0].set_title("Training Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(True, alpha=0.4)

axes[1].plot(range(1, NUM_EPOCHS + 1), test_acc_history,
             marker="s", color="darkorange", linewidth=2)
axes[1].set_title("Test Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim([50, 100])
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=100, bbox_inches="tight")
print("\nSaved: training_curves.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 9.  Final Evaluation — Per-Class Accuracy
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 55)
print("STEP 5 — Final Evaluation on 10,000 Test Images")
print("═" * 55)

model.eval()
all_images    = []
all_labels    = []
all_predicted = []

class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        _, preds = torch.max(logits, dim=1)

        all_images.append(images.cpu())
        all_labels.append(labels)
        all_predicted.append(preds.cpu())

        for label, pred in zip(labels, preds.cpu()):
            class_total[label]   += 1
            class_correct[label] += int(pred == label)

# Concatenate all batches
all_images    = torch.cat(all_images)
all_labels    = torch.cat(all_labels)
all_predicted = torch.cat(all_predicted)

overall_acc = (all_predicted == all_labels).float().mean().item() * 100

print(f"\nOverall Test Accuracy : {overall_acc:.2f}%\n")
print(f"{'Class':<16}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>10}")
print("─" * 48)
for i, name in enumerate(CLASS_NAMES):
    acc = 100.0 * class_correct[i] / class_total[i]
    bar = "█" * int(acc / 5)          # simple ASCII progress bar
    print(f"{name:<16}  {class_correct[i]:>8}  {class_total[i]:>7}  "
          f"{acc:>8.2f}%  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
# 10.  Visualise Correct and Incorrect Predictions
# ═══════════════════════════════════════════════════════════════════════════
correct_mask   = (all_predicted == all_labels)
incorrect_mask = ~correct_mask

correct_idx   = correct_mask.nonzero(as_tuple=True)[0]
incorrect_idx = incorrect_mask.nonzero(as_tuple=True)[0]

NUM_SHOW = 5   # number of examples to display in each group

# Randomly sample for variety each run
rng = torch.Generator().manual_seed(99)
correct_sample   = correct_idx[
    torch.randperm(len(correct_idx),   generator=rng)[:NUM_SHOW]]
incorrect_sample = incorrect_idx[
    torch.randperm(len(incorrect_idx), generator=rng)[:NUM_SHOW]]


def visualise_predictions(indices, title, border_color, filename=None):
    """Plot a row of images with actual vs predicted labels."""
    fig, axes = plt.subplots(1, NUM_SHOW, figsize=(14, 3))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, idx in zip(axes, indices):
        img  = all_images[idx].squeeze().numpy()
        img  = img * 0.3530 + 0.2860          # un-normalise
        img  = np.clip(img, 0, 1)

        actual    = CLASS_NAMES[all_labels[idx].item()]
        predicted = CLASS_NAMES[all_predicted[idx].item()]

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Actual:\n{actual}\nPred:\n{predicted}",
            fontsize=8,
            color=border_color
        )
        ax.axis("off")

        # Coloured border to make correct/incorrect obvious
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()


print("\n" + "─" * 55)
print("STEP 6 — Visualising Predictions")
print("─" * 55)

visualise_predictions(
    correct_sample,
    "✅  Correct Predictions (green border)",
    border_color="green",
    filename="correct_predictions.png"
)
print("Saved: correct_predictions.png")

visualise_predictions(
    incorrect_sample,
    "❌  Incorrect Predictions (red border)",
    border_color="red",
    filename="incorrect_predictions.png"
)
print("Saved: incorrect_predictions.png")

# ─── print a quick text summary of the incorrect ones ──────────────────────
print("\nIncorrect prediction details:")
print(f"  {'Index':>6}  {'Actual':<16}  {'Predicted':<16}")
print("  " + "─" * 42)
for idx in incorrect_sample:
    actual    = CLASS_NAMES[all_labels[idx].item()]
    predicted = CLASS_NAMES[all_predicted[idx].item()]
    print(f"  {idx.item():>6}  {actual:<16}  {predicted:<16}")

print(f"\n{'═'*55}")
print(f"✓  Task 4 complete.  Final accuracy: {overall_acc:.2f}%")
print(f"{'═'*55}")