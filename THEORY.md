# Theoretical Background: Depth-Supervised Neural Radiance Fields

This document provides detailed theoretical explanations of the depth supervision strategies implemented in this project.

---

## Table of Contents

1. [Neural Radiance Fields (NeRF) Foundations](#nerf-foundations)
2. [Volume Rendering](#volume-rendering)
3. [Depth Supervision Strategies](#depth-supervision-strategies)
4. [Loss Functions](#loss-functions)
5. [Mathematical Derivations](#mathematical-derivations)
6. [Implementation Details](#implementation-details)

---

## 1. NeRF Foundations

### 1.1 Scene Representation

Neural Radiance Fields represent a 3D scene as a continuous function:

```
F_Θ: (x, d) → (c, σ)
```

Where:
- **x = (x, y, z)** ∈ ℝ³ is a 3D spatial location
- **d = (θ, φ)** ∈ S² is a 2D viewing direction (unit vector)
- **c = (r, g, b)** ∈ [0,1]³ is emitted RGB color
- **σ ∈ ℝ⁺** is volume density (opacity)
- **Θ** are learned MLP parameters

### 1.2 Network Architecture

**Input Processing:**

1. **Positional Encoding** (Γ): Maps low-dim coordinates to high-dim space
   ```
   Γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
   ```
   - L = 10 for positions (60D output)
   - L = 4 for directions (24D output)

2. **MLP Forward Pass:**
   ```
   x_enc = Γ(x)  # 63D (3 + 60)
   d_enc = Γ(d)  # 27D (3 + 24)
   
   h = MLP_pts(x_enc)              # 8 layers, 256 width, skip at layer 4
   σ = softplus(Linear_σ(h))       # Density head
   features = Linear_feat(h)        # Feature vector
   
   h_dir = MLP_dir(concat(features, d_enc))  # View-dependent color
   rgb = sigmoid(Linear_rgb(h_dir))
   ```

---

## 2. Volume Rendering

### 2.1 Classical Volume Rendering Equation

The expected color C(r) along camera ray **r**(t) = **o** + t**d** is:

```
C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt
```

Where:
- **T(t) = exp(-∫[t_n to t] σ(r(s)) ds)** is transmittance (accumulated transparency)
- **σ(r(t))** is volume density at point r(t)
- **c(r(t), d)** is RGB color at point r(t) viewed from direction d
- **[t_n, t_f]** is the near-far bounds of the ray

### 2.2 Discrete Approximation

In practice, we discretize the integral using **stratified sampling**:

```
C(r) ≈ Σ[i=1 to N] T_i · α_i · c_i
```

Where:
- **N** = number of samples along ray
- **t_i ~ U[t_n + (i-1)/N · (t_f - t_n), t_n + i/N · (t_f - t_n)]** (stratified sampling)
- **δ_i = t_{i+1} - t_i** (distance between samples)
- **α_i = 1 - exp(-σ_i · δ_i)** (alpha compositing)
- **T_i = exp(-Σ[j=1 to i-1] σ_j · δ_j) = Π[j=1 to i-1] (1 - α_j)** (transmittance)

### 2.3 Depth Computation

Expected depth (distance from camera):

```
D(r) = Σ[i=1 to N] w_i · t_i
```

Where **w_i = T_i · α_i** are the accumulated weights.

**Property:** Σw_i = accumulated opacity (1.0 for opaque surfaces, <1.0 for transparent)

---

## 3. Depth Supervision Strategies

### 3.1 Baseline (No Depth)

**Loss:**
```
L_baseline = ||C_pred(r) - C_gt(r)||²
```

**Limitations:**
- Depth ambiguity: multiple geometry configurations can explain same RGB
- Floaters: spurious density in empty space
- Surface ambiguity: density spread over volume instead of concentrated at surface

---

### 3.2 Soft Depth Supervision

**Motivation:** Directly supervise expected depth prediction.

**Depth Prediction:**
```
D_pred(r) = Σ[i=1 to N] w_i · t_i
```

**Soft Depth Loss:**
```
L_soft = 1/|R| Σ[r∈R] ||D_pred(r) - D_gt(r)||²
```

**Total Loss:**
```
L = L_RGB + λ_soft · L_soft
```

**Hyperparameter:** λ_soft = 0.01 (tuned empirically)

**Advantages:**
- Simple to implement
- Differentiable end-to-end
- Improves depth accuracy

**Limitations:**
- Doesn't enforce sharp surfaces
- Still allows density before/after true surface

---

### 3.3 Hard Depth-Guided Sampling

**Motivation:** Concentrate samples where surfaces exist.

**Sampling Strategy:**

1. **Uniform samples** (25%): Exploration
   ```
   t_uniform ~ U[t_n, t_f]
   ```

2. **Depth-guided samples** (75%): Exploitation
   ```
   t_guided ~ U[D_gt - ε, D_gt + ε]
   ```
   Where ε = 0.3m is the depth window.

3. **Combined:** Merge and sort samples

**Free-Space Loss:** Penalize density before known surface

```
L_freespace = 1/|R| Σ[r∈R] Σ[i: t_i < D_gt(r)] σ_i
```

**Surface Concentration Loss:** Encourage sharp weight distribution

```
L_surface = ||D_pred - D_gt||₁ + λ_var · Var(w)
```

Where:
- **||D_pred - D_gt||₁** = L1 depth error
- **Var(w) = Σw_i · (t_i - D_pred)²** = variance of weight distribution
- **λ_var = 0.1**

**Total Loss:**
```
L = L_RGB + λ_hard · (L_freespace + L_surface)
```

**Hyperparameter:** λ_hard = 0.005

**Advantages:**
- Sharp surfaces
- No density in free space
- Efficient sampling

**Limitations:**
- Requires accurate depth ground truth
- More computational cost (non-uniform sampling)

---

### 3.4 Hybrid Strategy (Recommended)

**Motivation:** Combine benefits of soft and hard strategies.

**Sampling:** 50% uniform + 50% depth-guided

**Loss:**
```
L = L_RGB + λ_soft · L_soft + λ_hard · (L_freespace + L_surface)
```

**Hyperparameters:**
- λ_soft = 0.01
- λ_hard = 0.005

**Advantages:**
- Best of both worlds
- Robust to depth errors
- Sharp surfaces + accurate depth

---

## 4. Loss Functions

### 4.1 RGB Loss (All Strategies)

**Mean Squared Error:**
```
L_RGB = 1/|R| Σ[r∈R] ||C_pred(r) - C_gt(r)||²
```

### 4.2 Depth Loss (Soft & Hybrid)

**L2 Depth Error:**
```
L_depth = 1/|R| Σ[r∈R] (D_pred(r) - D_gt(r))²
```

### 4.3 Free-Space Loss (Hard & Hybrid)

**Penalize density before surface:**
```
L_freespace = Σ[i: t_i < D_gt] σ(t_i)
```

### 4.4 Surface Concentration Loss (Hard & Hybrid)

**Encourage peaked weight distribution:**
```
L_surface = |D_pred - D_gt| + 0.1 · √(Σ w_i · (t_i - D_pred)²)
```

---

## 5. Mathematical Derivations

### 5.1 Gradient of Volume Rendering

**Forward pass:**
```
C = Σ[i=1 to N] w_i · c_i
w_i = T_i · α_i
T_i = Π[j=1 to i-1] (1 - α_j)
α_i = 1 - exp(-σ_i · δ_i)
```

**Backpropagation through volume rendering:**
```
∂C/∂c_i = w_i
∂C/∂σ_i = δ_i · exp(-σ_i · δ_i) · (T_i · c_i - Σ[j>i] T_j · c_j)
```

### 5.2 Depth Gradient

**Depth prediction:**
```
D = Σ[i=1 to N] w_i · t_i
```

**Gradient:**
```
∂D/∂σ_i = δ_i · exp(-σ_i · δ_i) · (T_i · t_i - Σ[j>i] T_j · t_j)
```

---

## 6. Implementation Details

### 6.1 Network Architecture

```python
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_ch_xyz=63, in_ch_dir=27, skips=[4]):
        # D: network depth (8 layers)
        # W: network width (256 neurons)
        # skips: skip connections at layer 4
```

### 6.2 Training Hyperparameters

**Optimizer:** Adam
- Learning rate: 5e-4
- β₁ = 0.9, β₂ = 0.999
- Weight decay: 0 (no L2 regularization)

**Batch size:**
- Rays per batch: 1024
- Samples per ray: 64 (baseline/soft) or 64 (hard/hybrid)

**Training iterations:**
- 20,000 iterations (paper reports 200k for best results)
- ~12-18 hours on single V100 GPU

### 6.3 Sampling Strategy

**Stratified Sampling (Baseline/Soft):**
```python
t_vals = torch.linspace(0, 1, N_samples)
z_vals = near * (1 - t_vals) + far * t_vals
# Add noise
mids = 0.5 * (z_vals[:-1] + z_vals[1:])
z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
```

**Depth-Guided Sampling (Hard/Hybrid):**
```python
N_uniform = N_samples // 4  # 25%
N_guided = N_samples - N_uniform  # 75%

t_uniform = torch.linspace(near, far, N_uniform)
t_guided = torch.linspace(depth - eps, depth + eps, N_guided)
z_vals = torch.cat([t_uniform, t_guided]).sort()
```

### 6.4 Evaluation Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
```
PSNR = -10 · log₁₀(MSE)
```

**SSIM (Structural Similarity Index):**
```python
from skimage.metrics import structural_similarity
ssim_val = structural_similarity(img1, img2, multichannel=True)
```

**LPIPS (Learned Perceptual Image Patch Similarity):**
```python
import lpips
loss_fn = lpips.LPIPS(net='alex')
lpips_val = loss_fn(img1, img2)
```

---

## References

1. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020
2. Deng et al., "Depth-supervised NeRF", arXiv 2021
3. Roessle et al., "Dense Depth Priors for Neural Radiance Fields from Sparse Input Views", CVPR 2022
4. Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions", NeurIPS 2020

---

**Author:** Ayan Ali  
**Course:** Machine Learning in Robotics, ITMO University  
**Last Updated:** December 13, 2025
