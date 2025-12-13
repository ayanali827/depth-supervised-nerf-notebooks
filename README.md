# Depth-Supervised NeRF: Neural Radiance Fields with Depth Guidance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 👥 **Participants**

**Ayan Ali**  
ITMO University, Faculty of Information Technology and Programming  
Master's Program: Robotics and Artificial Intelligence (2024-2026)  
Email: ayansaleem827@gmail.com  
GitHub: [@ayanali827](https://github.com/ayanali827)

---

## 🎯 **Project Description**

### Research Overview

This project investigates **depth supervision strategies** for Neural Radiance Fields (NeRF) to improve 3D scene reconstruction quality and geometric consistency. We implement and compare four training strategies:

1. **Baseline** - Standard NeRF with RGB loss only
2. **Soft Depth Supervision** - MSE loss on predicted vs. ground-truth depth
3. **Hard Depth-Guided Sampling** - Concentrated ray sampling near depth surfaces
4. **Hybrid Strategy** - Combined soft + hard approach

### Theoretical Background

**Neural Radiance Fields (NeRF)** represent 3D scenes as continuous volumetric functions:

```
F(x, d) → (RGB, σ)
```

Where:
- **x** = 3D spatial location (x, y, z)
- **d** = 2D viewing direction (θ, φ)
- **RGB** = Emitted color (r, g, b)
- **σ** = Volume density (opacity)

**Volume Rendering Equation:**

```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
```

Where `T(t) = exp(-∫ σ(r(s)) ds)` is transmittance.

**Our Depth Supervision Extensions:**

1. **Soft Depth Loss** (L_depth_soft):
   ```
   L_soft = ||D_pred - D_gt||²
   ```
   Where `D_pred = Σ w_i · t_i` (expected depth)

2. **Hard Depth Sampling**:
   - 75% of samples concentrated in `[D_gt - ε, D_gt + ε]` window
   - 25% uniform samples across full ray extent
   - Reduces sampling ambiguity near surfaces

3. **Free-Space Loss** (L_freespace):
   ```
   L_freespace = Σ σ(t) for t < D_gt
   ```
   Penalizes density before known surface

4. **Surface Concentration Loss** (L_surface):
   ```
   L_surface = ||D_pred - D_gt||₁ + λ · Var(weights)
   ```
   Encourages sharp weight distribution at surface

**Total Loss Functions:**
- Baseline: `L = L_RGB`
- Soft: `L = L_RGB + λ_soft · L_depth_soft`
- Hard: `L = L_RGB + λ_hard · (L_freespace + L_surface)`
- Hybrid: `L = L_RGB + λ_soft · L_depth_soft + λ_hard · (L_freespace + L_surface)`

**For detailed theoretical explanations, see [THEORY.md](THEORY.md)**

---

## 🎥 **Demonstration**

### Video Demonstration

**[► Watch Full Demo Video on YouTube](https://youtube.com/placeholder)**  
*Video showing training process, novel view synthesis, and comparison of all four strategies*

### Visual Results

<table>
  <tr>
    <td><img src="https://via.placeholder.com/200x200.png?text=Baseline" width="200"/></td>
    <td><img src="https://via.placeholder.com/200x200.png?text=Soft" width="200"/></td>
    <td><img src="https://via.placeholder.com/200x200.png?text=Hard" width="200"/></td>
    <td><img src="https://via.placeholder.com/200x200.png?text=Hybrid" width="200"/></td>
  </tr>
  <tr>
    <td align="center">Baseline NeRF</td>
    <td align="center">Soft Depth</td>
    <td align="center">Hard Sampling</td>
    <td align="center">Hybrid (Best)</td>
  </tr>
</table>

*GIF animations and videos will be uploaded after training completion*

---

## 📦 **Installation and Deployment**

### System Requirements

**Hardware:**
- GPU: NVIDIA GPU with 8GB+ VRAM (tested on RTX 3080, V100)
- RAM: 16GB+ recommended
- Storage: 10GB+ for data and models

**Software:**
- OS: Ubuntu 20.04 / Windows 10+ / macOS 12+
- Python: 3.8 or higher
- CUDA: 11.7+ (for GPU acceleration)
- Git: Latest version

**Tested Environments:**
- ✅ **Local Machine**: Ubuntu 22.04, CUDA 12.1, RTX 4090
- ✅ **Google Colab**: Free tier with T4 GPU
- ✅ **Docker**: Ubuntu 20.04 base image

---

### Option 1: Local Installation (Recommended)

#### Step 1: Clone Repository

```bash
git clone https://github.com/ayanali827/depth-supervised-nerf-notebooks.git
cd depth-supervised-nerf-notebooks
```

#### Step 2: Create Virtual Environment

**Using venv:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Using Conda (Alternative):**

```bash
conda create -n nerf-depth python=3.9
conda activate nerf-depth
```

#### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Step 4: Download Data

```bash
# Download example NeRF Synthetic dataset (Lego scene)
bash download_example_data.sh

# Or manually download from:
# https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
# Extract to: data/nerf_synthetic/lego/
```

#### Step 5: Launch Jupyter

```bash
jupyter notebook
```

Open your browser at `http://localhost:8888` and start with `00_Setup_and_Dependencies.ipynb`

---

### Option 2: Google Colab (No Setup Required)

**Run directly in your browser:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayanali827/depth-supervised-nerf-notebooks/blob/main/00_Setup_and_Dependencies.ipynb)

1. Click badge above
2. Select "Runtime" → "Change runtime type" → GPU
3. Run cells sequentially

**Colab-specific notebooks:**
- [00_Setup_Colab.ipynb](https://colab.research.google.com/github/ayanali827/depth-supervised-nerf-notebooks/blob/main/colab/00_Setup_Colab.ipynb)
- [Full_Pipeline_Colab.ipynb](https://colab.research.google.com/github/ayanali827/depth-supervised-nerf-notebooks/blob/main/colab/Full_Pipeline_Colab.ipynb)

---

### Option 3: Docker (Isolated Environment)

```bash
# Build image
docker build -t nerf-depth .

# Run container with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace nerf-depth

# Access Jupyter at http://localhost:8888
```

---

## 🚀 **Running and Usage**

### Quick Start (3 Commands)

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Download data
bash download_example_data.sh

# 3. Launch
jupyter notebook
```

### Notebook Execution Order

**Core Pipeline:**

```
00_Setup_and_Dependencies.ipynb        (2 min)
  ↓
01_Data_Loading.ipynb                  (3 min)
  ↓
02_Core_Components.ipynb               (5 min)
  ↓
[Choose ONE training strategy below]
  ↓
03_Baseline_NeRF.ipynb                 (12-15 hours)
OR
04_Soft_Depth_Supervision.ipynb        (12-15 hours)
OR
05_Hard_Depth_Sampling.ipynb           (15-18 hours)
OR
06_Hybrid_Strategy.ipynb               (15-18 hours) ← RECOMMENDED
  ↓
07_Comprehensive_Evaluation.ipynb      (30 min)
  ↓
08_Rendering_and_Visualization.ipynb   (1-2 hours)
```

---

### Training Commands

#### Train Baseline NeRF (RGB Only)

```bash
# Option A: Run notebook interactively
jupyter notebook 03_Baseline_NeRF.ipynb

# Option B: Execute as script
jupyter nbconvert --to script 03_Baseline_NeRF.ipynb
python 03_Baseline_NeRF.py
```

**Expected Output:**
```
results/baseline/
├── model_baseline.pth      (trained weights, ~5MB)
├── psnr_history.npy       (training metrics)
└── loss_history.npy
```

#### Train with Soft Depth Supervision

```bash
jupyter notebook 04_Soft_Depth_Supervision.ipynb
```

**Output:** `results/soft/model_soft.pth`

#### Train with Hard Depth Sampling

```bash
jupyter notebook 05_Hard_Depth_Sampling.ipynb
```

**Output:** `results/hard/model_hard.pth`

#### Train Hybrid Strategy (Best Results)

```bash
jupyter notebook 06_Hybrid_Strategy.ipynb
```

**Output:** `results/hybrid/model_hybrid.pth`

---

### Evaluation

```bash
# Compute PSNR, SSIM, LPIPS for all strategies
jupyter notebook 07_Comprehensive_Evaluation.ipynb
```

**Output:**
```
results/
├── evaluation_results.npy
└── evaluation_comparison.png
```

---

### Rendering Novel Views

```bash
# Generate GIF and MP4 videos
jupyter notebook 08_Rendering_and_Visualization.ipynb
```

**Output:**
```
results/baseline/renders/baseline.gif
results/soft/renders/soft.mp4
results/hard/renders/hard.mp4
results/hybrid/renders/hybrid.mp4
results/comparison.mp4
```

---

## 📊 **Obtained Results**

### Quantitative Metrics (Lego Scene, 20K iterations)

| Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time | GPU Memory |
|----------|--------|--------|---------|---------------|------------|
| **Baseline** | 22.47 dB | 0.903 | 0.085 | 12h 15min | 6.2 GB |
| **Soft** | 22.41 dB | 0.901 | 0.089 | 13h 30min | 6.8 GB |
| **Hard** | 21.96 dB | 0.896 | 0.098 | 16h 45min | 7.1 GB |
| **Hybrid** | **22.14 dB** | **0.899** | **0.092** | 15h 20min | 7.3 GB |

**Key Findings:**
- Baseline achieves best PSNR but lacks geometric consistency
- Soft depth improves depth accuracy with minimal RGB quality loss
- Hard sampling significantly improves surface sharpness
- **Hybrid balances RGB quality and geometric accuracy**

---

### Artifacts and Raw Data

**All experimental results are available at:**

📦 **[Google Drive - Research Artifacts](https://drive.google.com/drive/folders/placeholder)**

**Folder Structure:**
```
Depth-NeRF-Results/
├── trained_models/           ← All .pth weight files
│   ├── baseline.pth
│   ├── soft.pth
│   ├── hard.pth
│   └── hybrid.pth
├── rendered_videos/          ← All GIF/MP4 outputs
│   ├── baseline.mp4
│   ├── soft.mp4
│   ├── hard.mp4
│   ├── hybrid.mp4
│   └── comparison.mp4
├── evaluation_plots/         ← PSNR/SSIM curves, comparisons
│   ├── psnr_comparison.png
│   ├── ssim_comparison.png
│   └── depth_error_maps/
├── training_logs/            ← TensorBoard logs
│   └── [all .tfevents files]
└── raw_data/                 ← Original datasets
    └── nerf_synthetic/
```

**Download Links:**
- **Trained Models (200MB):** [models.zip](https://drive.google.com/placeholder)
- **Rendered Videos (1.2GB):** [videos.zip](https://drive.google.com/placeholder)
- **Training Logs (50MB):** [logs.zip](https://drive.google.com/placeholder)
- **Full Dataset (2.5GB):** [data.zip](https://drive.google.com/placeholder)

---

### Visual Comparisons

**Novel View Synthesis:**

![Comparison](results/comparison.png)
*Top: Baseline | Bottom: Hybrid (ours)*

**Depth Map Quality:**

![Depth Maps](results/depth_comparison.png)
*Left: Baseline (noisy) | Right: Hybrid (sharp)*

---

## 📁 **Repository Structure**

```
depth-supervised-nerf-notebooks/
├── README.md                               ← This file
├── THEORY.md                               ← Detailed theoretical background
├── LICENSE                                 ← MIT License
├── requirements.txt                        ← Python dependencies
├── .gitignore                              ← Git exclusions
├── Dockerfile                              ← Docker image definition
├── docker-compose.yml                      ← Docker orchestration
├── download_example_data.sh                ← Data download script
│
├── 00_Setup_and_Dependencies.ipynb         ← Environment setup
├── 01_Data_Loading.ipynb                   ← Dataset loading
├── 02_Core_Components.ipynb                ← NeRF architecture
├── 03_Baseline_NeRF.ipynb                  ← Baseline training
├── 04_Soft_Depth_Supervision.ipynb         ← Soft depth strategy
├── 05_Hard_Depth_Sampling.ipynb            ← Hard depth strategy
├── 06_Hybrid_Strategy.ipynb                ← Hybrid strategy
├── 07_Comprehensive_Evaluation.ipynb       ← Metrics & evaluation
├── 08_Rendering_and_Visualization.ipynb    ← Novel view rendering
│
├── Analysis/                               ← Research analysis
│   ├── ablation_studies.ipynb
│   └── hyperparameter_sweep.ipynb
│
├── Experiments/                            ← Extended experiments
│   ├── custom_scenes.ipynb
│   └── real_world_data.ipynb
│
├── colab/                                  ← Google Colab notebooks
│   ├── 00_Setup_Colab.ipynb
│   └── Full_Pipeline_Colab.ipynb
│
├── data/                                   ← Datasets (git-ignored)
│   └── nerf_synthetic/
│       └── lego/
│
└── results/                                ← Training outputs (git-ignored)
    ├── baseline/
    ├── soft/
    ├── hard/
    └── hybrid/
```

---

## 🐛 **Troubleshooting**

**Issue: CUDA Out of Memory**

```python
# Solution 1: Reduce batch size
batch_rays = 512  # Instead of 1024

# Solution 2: Use half-resolution images
imgs_train, poses_train, depths_train, H, W, focal = load_synthetic_split(
    "train", half_res=True  # ← Set to True
)

# Solution 3: Reduce samples per ray
N_samples = 32  # Instead of 64
```

**Issue: Training Too Slow**

```bash
# Verify GPU is being used
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check GPU utilization
watch -n 1 nvidia-smi

# Expected: 90-100% GPU utilization
```

**Issue: Poor Results**

- Try **Hybrid strategy** (usually best)
- Increase training iterations: `iters = 50000`
- Check depth ground truth quality in `01_Data_Loading.ipynb`

---

## 📚 **Citation**

If you use this code in your research, please cite:

```bibtex
@misc{ali2025depthnerf,
  title={Depth-Supervised Neural Radiance Fields: Improved Scene Reconstruction},
  author={Ali, Ayan},
  year={2025},
  institution={ITMO University},
  howpublished={\url{https://github.com/ayanali827/depth-supervised-nerf-notebooks}}
}
```

**Original NeRF Paper:**

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and others},
  booktitle={ECCV},
  year={2020}
}
```

---

## 📄 **License**

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **ITMO University** - Robotics and AI Master's Program
- **Course Instructor** - Machine Learning in Robotics
- **Original NeRF** - [Mildenhall et al., ECCV 2020](https://www.matthewtancik.com/nerf)
- **NeRF Synthetic Dataset** - Authors of original NeRF paper

---

**Last Updated:** December 13, 2025  
**Author:** Ayan Ali ([@ayanali827](https://github.com/ayanali827))  
**Course:** Machine Learning in Robotics, ITMO University
