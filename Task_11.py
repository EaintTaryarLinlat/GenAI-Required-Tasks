"""Student Name : Eaint Taryar Linlat
Task 11: Flow Matching for Synthetic Floorplan Generation
In this task, I trained a generative model using Flow Matching to synthesise new floorplan images from random noise — a more stable alternative to GANs.
What I did:

Preprocessed the floorplan dataset by resizing all images to 128×128 grayscale and caching them locally to avoid repeated processing
Built a conditional U-Net with sinusoidal time embeddings, residual blocks with GroupNorm and SiLU activations, skip connections between encoder and decoder, and a class embedding — the architecture predicts the velocity field needed to transform noise into a real image
Implemented Flow Matching — instead of predicting noise like a diffusion model, the model learns to predict the direction (x1 - x0) that moves a noisy image xt toward a real image x1, trained with MSE loss on interpolated samples at random timesteps t ∈ [0, 1]
Used two samplers — Euler (simple step-by-step integration) and Heun (corrected two-step integration, more accurate), both used at inference to walk from noise to a generated floorplan
Trained for 20 epochs with AdamW, cosine learning rate annealing, and gradient clipping

Key lessons:

Flow Matching vs Diffusion — both start from Gaussian noise, but Flow Matching uses straight-line trajectories between noise and data which means fewer inference steps are needed compared to diffusion models
Heun sampler improves quality — it computes a corrected velocity estimate using a predictor-corrector step, giving better results than Euler for the same number of steps
Memory constraints matter — batch size 2 and base channels 32 were necessary to run within Codespace RAM limits, which is a real engineering trade-off between model capacity and available hardware"""
###############################################################################

import os
import glob
import zipfile
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate slightly broken image files

# ── 1. Device (CUDA -> Apple MPS -> CPU) ─────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon GPU
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── 2. Paths ──────────────────────────────────────────────────────────────────
ZIP_PATH    = "Task_11/floorplans_v2-20251223T170650Z-3-001.zip"
EXTRACT_DIR = "Task_11/floorplans_v2-20251223T170650Z-3-001/floorplans_v2"
CACHE_DIR   = "Task_11/floorplans_256"
SAVE_PATH   = "Task_11/floorplan_flow_model.pth"
OUTPUT_DIR  = "Task_11/generated_floorplans"

# ── 3. Hyperparameters ────────────────────────────────────────────────────────
HPARAMS = {
    "img_size":        128,   # reduced from 256 to save RAM
    "channels":          1,   # grayscale; set 3 for RGB
    "batch_size":        2,   # reduced for Codespace RAM limit
    "lr":             2e-4,
    "weight_decay":   1e-4,
    "epochs":           20,
    "inference_steps": 100,
    "grad_clip":        1.0,
    "base_channels":    32,   # reduced from 64 to save RAM
    "num_classes":       1,
    "t_dim":           128,
}
print("Hyperparameters:", HPARAMS)

# ── 4. Extract ZIP ────────────────────────────────────────────────────────────
# Skip extraction if already extracted (folder exists)
if not os.path.isdir(EXTRACT_DIR):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Cannot find '{ZIP_PATH}'.\n"
            "Place the zip file in the same folder as this script."
        )
    print(f"Extracting '{ZIP_PATH}' to '{EXTRACT_DIR}' ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete.")
else:
    print(f"'{EXTRACT_DIR}' already exists - skipping extraction.")

raw_imgs = []
for ext in ("png", "jpg", "jpeg", "webp"):
    raw_imgs += glob.glob(os.path.join(EXTRACT_DIR, "**", f"*.{ext}"), recursive=True)
print(f"Found {len(raw_imgs)} raw images.")
if not raw_imgs:
    raise RuntimeError("No images found after extraction. Check zip structure.")

# ── 5. Preprocess to 256x256 cache ───────────────────────────────────────────
if not os.path.isdir(CACHE_DIR) or len(glob.glob(f"{CACHE_DIR}/*.png")) == 0:
    print(f"Preprocessing to {HPARAMS['img_size']}x{HPARAMS['img_size']} cache ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    saved = 0
    for i, p in enumerate(raw_imgs):
        try:
            img = Image.open(p)
            img = img.convert("L") if HPARAMS["channels"] == 1 else img.convert("RGB")
            img = img.resize((HPARAMS["img_size"], HPARAMS["img_size"]), resample=Image.BILINEAR)
            img.save(os.path.join(CACHE_DIR, f"fp_{i:05d}.png"), optimize=True)
            saved += 1
        except Exception:
            continue
    print(f"Saved {saved} preprocessed images to '{CACHE_DIR}'.")
else:
    print(f"Cache '{CACHE_DIR}' already exists - skipping.")

# ── 6. Dataset ────────────────────────────────────────────────────────────────
class FloorplanDataset(Dataset):
    def __init__(self, root_dir, img_size=256, channels=1):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True))
        if not self.paths:
            for ext in ("jpg", "jpeg", "webp"):
                self.paths += glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        if not self.paths:
            raise RuntimeError(f"No images found under '{root_dir}'.")
        print(f"Dataset: {len(self.paths)} images from '{root_dir}'.")
        self.channels  = channels
        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx])
            img = img.convert("L") if self.channels == 1 else img.convert("RGB")
            return self.transform(img), 0
        except Exception:
            return self.__getitem__(random.randint(0, len(self.paths) - 1))


dataset = FloorplanDataset(CACHE_DIR, img_size=HPARAMS["img_size"], channels=HPARAMS["channels"])
loader  = DataLoader(dataset, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=0, drop_last=True)
print(f"Batches per epoch: {len(loader)}")

# ── 7. Model ──────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(torch.linspace(0, np.log(10_000), half, device=t.device) * -1)
        args  = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.skip     = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, img_ch, base_ch, num_classes=1, t_dim=128):
        super().__init__()
        self.time_emb  = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.class_emb = nn.Embedding(num_classes, t_dim)
        self.in_conv   = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        self.rb1   = ResBlock(base_ch,     base_ch,     t_dim)
        self.rb2   = ResBlock(base_ch,     base_ch,     t_dim)
        self.dn1   = Downsample(base_ch)
        self.rb3   = ResBlock(base_ch,     base_ch * 2, t_dim)
        self.rb4   = ResBlock(base_ch * 2, base_ch * 2, t_dim)
        self.dn2   = Downsample(base_ch * 2)
        self.rb5   = ResBlock(base_ch * 2, base_ch * 4, t_dim)
        self.rb6   = ResBlock(base_ch * 4, base_ch * 4, t_dim)
        self.mid1  = ResBlock(base_ch * 4, base_ch * 4, t_dim)
        self.mid2  = ResBlock(base_ch * 4, base_ch * 4, t_dim)
        self.up1   = Upsample(base_ch * 4)
        self.rb7   = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, t_dim)
        self.rb8   = ResBlock(base_ch * 2, base_ch * 2, t_dim)
        self.up2   = Upsample(base_ch * 2)
        self.rb9   = ResBlock(base_ch * 2 + base_ch, base_ch, t_dim)
        self.rb10  = ResBlock(base_ch, base_ch, t_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, img_ch, 1)

    def forward(self, x, t, y):
        t_emb = self.time_emb(t) + self.class_emb(y)
        x  = self.in_conv(x)
        s1 = self.rb2(self.rb1(x,  t_emb), t_emb)
        x  = self.dn1(s1)
        s2 = self.rb4(self.rb3(x,  t_emb), t_emb)
        x  = self.dn2(s2)
        x  = self.rb6(self.rb5(x,  t_emb), t_emb)
        x  = self.mid2(self.mid1(x, t_emb), t_emb)
        x  = self.up1(x)
        x  = self.rb8(self.rb7(torch.cat([x, s2], 1), t_emb), t_emb)
        x  = self.up2(x)
        x  = self.rb10(self.rb9(torch.cat([x, s1], 1), t_emb), t_emb)
        return self.out_conv(F.silu(self.out_norm(x)))

# ── 8. Flow Matching ──────────────────────────────────────────────────────────
class FlowMatching:
    @staticmethod
    def compute_loss(model, x1, labels):
        b      = x1.size(0)
        x0     = torch.randn_like(x1)
        t      = torch.rand(b, device=x1.device)
        xt     = (1 - t.view(b,1,1,1)) * x0 + t.view(b,1,1,1) * x1
        return F.mse_loss(model(xt, t, labels), x1 - x0)

    @torch.no_grad()
    def sample_euler(self, model, n, steps=100):
        model.eval()
        x  = torch.randn(n, HPARAMS["channels"], HPARAMS["img_size"], HPARAMS["img_size"], device=device)
        y  = torch.zeros(n, dtype=torch.long, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            x = x + model(x, torch.full((n,), i/steps, device=device), y) * dt
        model.train()
        return x

    @torch.no_grad()
    def sample_heun(self, model, n, steps=100):
        model.eval()
        x  = torch.randn(n, HPARAMS["channels"], HPARAMS["img_size"], HPARAMS["img_size"], device=device)
        y  = torch.zeros(n, dtype=torch.long, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t0 = torch.full((n,), i/steps,       device=device)
            t1 = torch.full((n,), (i+1)/steps,   device=device)
            v0 = model(x, t0, y)
            v1 = model(x + dt * v0, t1, y)
            x  = x + dt * 0.5 * (v0 + v1)
        model.train()
        return x

# ── 9. Viz helper ─────────────────────────────────────────────────────────────
def show_samples(tensor_batch, title="Samples", save_path=None):
    x    = tensor_batch.detach().cpu()
    grid = make_grid(x, nrow=min(4, x.size(0)), padding=2, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1,2,0).numpy().squeeze(), cmap="gray")
    plt.axis("off"); plt.title(title); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.show()

# ── 10. Train ─────────────────────────────────────────────────────────────────
model = ConditionalUNet(HPARAMS["channels"], HPARAMS["base_channels"], HPARAMS["num_classes"], HPARAMS["t_dim"]).to(device)
flow  = FlowMatching()
opt   = torch.optim.AdamW(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HPARAMS["epochs"])

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")
print(f"\nTraining for {HPARAMS['epochs']} epochs ...\n")

losses = []
model.train()
for epoch in range(1, HPARAMS["epochs"] + 1):
    running = 0.0
    pbar    = tqdm(loader, desc=f"Epoch {epoch:>3}/{HPARAMS['epochs']}")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = (y if isinstance(y, torch.Tensor) else torch.tensor(y)).long().to(device)
        opt.zero_grad(set_to_none=True)
        loss = flow.compute_loss(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
        opt.step()
        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    sched.step()
    avg = running / len(loader)
    losses.append(avg)
    print(f"  Epoch {epoch:>3} - avg loss: {avg:.6f}")
    if epoch % 5 == 0:
        show_samples(flow.sample_heun(model, n=4, steps=50), title=f"Epoch {epoch} preview")

# ── 11. Save ──────────────────────────────────────────────────────────────────
torch.save({"model_state": model.state_dict(), "hparams": HPARAMS}, SAVE_PATH)
print(f"\nModel saved -> {SAVE_PATH}")

# ── 12. Loss curve ────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(losses, linewidth=2, color="steelblue")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Flow Matching - Training Loss")
plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150); plt.show()
print("Loss curve saved -> loss_curve.png")

# ── 13. Generate ──────────────────────────────────────────────────────────────
print("\nGenerating synthetic floorplans ...")
ckpt = torch.load(SAVE_PATH, map_location=device)
h    = ckpt["hparams"]
gen_model = ConditionalUNet(h["channels"], h["base_channels"], h["num_classes"], h["t_dim"]).to(device)
gen_model.load_state_dict(ckpt["model_state"])
gen_model.eval()
gen_flow = FlowMatching()

with torch.no_grad():
    gen = gen_flow.sample_heun(gen_model, n=12, steps=h["inference_steps"])

show_samples(gen, title="Generated Synthetic Floorplans", save_path="generated_floorplans_grid.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(gen.size(0)):
    img = gen[i].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    T.ToPILImage()(img).save(os.path.join(OUTPUT_DIR, f"floorplan_{i:03d}.png"))
print(f"Saved {gen.size(0)} PNGs -> '{OUTPUT_DIR}/'")

print("\nDone!")
print("  floorplan_flow_model.pth        <- model weights")
print("  generated_floorplans_grid.png   <- 12-image grid")
print(f"  {OUTPUT_DIR}/floorplan_000.png <- individual samples")
print("  loss_curve.png                  <- training loss")

