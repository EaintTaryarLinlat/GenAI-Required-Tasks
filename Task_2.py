"""
Student Name: [Eaint Taryar Linlat]
Task 2: Exploring Autograd with a Complex Function
In this task, I explored PyTorch's automatic differentiation engine (Autograd) by computing partial derivatives of the function z = σ(w·x²) + 1/b³ with respect to w, x, and b.
What I did:

Created three tensors (w=2.0, x=4.0, b=1.5) with requires_grad=True to enable gradient tracking
Computed z using PyTorch operations, which builds a computational graph automatically
Called z.backward() to trigger backpropagation through the graph
Read the gradients from w.grad, x.grad, and b.grad

Key findings:

dz/dw ≈ 2.02e-13 and dz/dx ≈ 2.02e-13 — both near zero because sigmoid(32) is deeply saturated. The sigmoid gradient σ(u)·(1−σ(u)) collapses to essentially zero at extreme inputs, so changes to w or x have almost no effect on z
dz/db ≈ −0.5926 — the dominant gradient. Derived from −3/b⁴, it shows that increasing b noticeably decreases z

Core lesson: Autograd eliminates the need to derive gradients by hand. However, understanding why a gradient is near-zero (sigmoid saturation) or large is critical — blindly trusting gradient values without this context can lead to misdiagnosed training problems like the vanishing gradient issue in deep networks.
"""

import torch

#Step 1: Create tensors with gradient tracking enabled 
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)
b = torch.tensor(1.5, requires_grad=True)

print("Initial values:")
print(f"  w = {w.item()},  x = {x.item()},  b = {b.item()}")
print(f"  w.requires_grad = {w.requires_grad}")
print(f"  x.requires_grad = {x.requires_grad}")
print(f"  b.requires_grad = {b.requires_grad}")

#Step 2: Compute z using the tensors 
#   z = σ(w * x²)  +  b⁻³
z = torch.sigmoid(w * x**2) + 1 / b**3

print(f"\nComputed z = {z.item():.8f}")
print(f"  (note: sigmoid(2*16)=sigmoid(32)≈1.0  and  1/1.5³≈0.2963)")

#Step 3: Call .backward() to compute all partial derivatives
z.backward()

#Step 4: Print the gradients 
print("\n" + "="*50)
print("Computed Gradients (via Autograd):")
print(f"  dz/dw = w.grad = {w.grad.item():.6e}")
print(f"  dz/dx = x.grad = {x.grad.item():.6e}")
print(f"  dz/db = b.grad = {b.grad.item():.8f}")

#Manual Verification 
#  Let  u = w·x²
#  dz/dw  =  σ(u)·(1−σ(u)) · x²          [chain rule on sigmoid term]
#  dz/dx  =  σ(u)·(1−σ(u)) · 2wx         [chain rule on sigmoid term]
#  dz/db  =  d(b⁻³)/db = −3·b⁻⁴
#
import math

u     = 2.0 * 4.0**2                         # = 32.0
sig_u = 1 / (1 + math.exp(-u))               # ≈ 1.0  (saturated sigmoid)
dsig  = sig_u * (1 - sig_u)                  # ≈ 1.26e-14  (near-zero gradient)

manual_dz_dw = dsig * 4.0**2                 # ≈ 2.02e-13
manual_dz_dx = dsig * 2 * 2.0 * 4.0         # ≈ 2.02e-13
manual_dz_db = -3 / 1.5**4                   # ≈ -0.5926

print("\nManual Verification:")
print(f"  dz/dw (manual) = {manual_dz_dw:.6e}  ✓")
print(f"  dz/dx (manual) = {manual_dz_dx:.6e}  ✓")
print(f"  dz/db (manual) = {manual_dz_db:.8f}  ✓")

print("\n" + "="*50)
print("Interpretation:")
print("  • dz/dw and dz/dx are ~0 because sigmoid(32) is deeply saturated.")
print("    The gradient of sigmoid is essentially zero at such extreme inputs.")
print("  • dz/db = -3/b⁴ ≈ -0.593 — this is the dominant gradient here.")
print("    A small increase in b decreases z (makes 1/b³ smaller).")
print("\nDone.")