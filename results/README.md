# Results Directory

This directory contains all experimental outputs: trained models, evaluation metrics, and rendered visualizations.

---

## 🚨 **Important: Results are NOT Included in Repository**

The results are **NOT stored in Git** because:
- Model weights (.pth files): ~5 MB each × 4 = 20 MB
- Rendered videos (.mp4 files): ~100 MB each × 5 = 500 MB
- Total size: ~600 MB (exceeds GitHub limits)

**Results are hosted on Google Drive** (links in main README.md)

---

## 📁 **Directory Structure**

After running all experiments, this folder will contain:

```
results/
├── README.md                           ← This file
│
├── baseline/                           ← Experiment 1: RGB loss only
│   ├── model_baseline.pth             (~5 MB) Trained NeRF weights
│   ├── psnr_history.npy               Training PSNR over iterations
│   ├── loss_history.npy               Training loss over iterations
│   └── renders/
│       ├── novel_view_000.png
│       ├── novel_view_001.png
│       ├── ...
│       ├── novel_view_119.png
│       ├── baseline.mp4                (~100 MB) 360° rotation video
│       └── baseline.gif                (~20 MB) GIF animation
│
├── soft/                               ← Experiment 2: + Soft depth loss
│   ├── model_soft.pth
│   ├── psnr_history.npy
│   ├── loss_history.npy
│   └── renders/
│       ├── soft.mp4
│       └── soft.gif
│
├── hard/                               ← Experiment 3: + Hard depth sampling
│   ├── model_hard.pth
│   ├── psnr_history.npy
│   ├── loss_history.npy
│   └── renders/
│       ├── hard.mp4
│       └── hard.gif
│
├── hybrid/                             ← Experiment 4: Combined approach
│   ├── model_hybrid.pth
│   ├── psnr_history.npy
│   ├── loss_history.npy
│   └── renders/
│       ├── hybrid.mp4
│       └── hybrid.gif
│
├── evaluation_results.npy              ← Comprehensive metrics (all strategies)
├── comparison_plot.png                 ← Side-by-side visual comparison
├── psnr_curves.png                     ← Training PSNR over time
├── comparison.mp4                      (~200 MB) All 4 strategies side-by-side
│
└── depth_error_maps/                   ← Depth accuracy visualizations
    ├── baseline_depth_error.png
    ├── soft_depth_error.png
    ├── hard_depth_error.png
    └── hybrid_depth_error.png
```

**Total size after all experiments:** ~600 MB

---

## 🎯 **How to Generate Results**

### Step 1: Run Experiments (Generate Models)

```bash
# From repository root

# Experiment 1: Baseline (12-15 hours)
jupyter notebook 03_Baseline_NeRF.ipynb
# → Creates results/baseline/model_baseline.pth

# Experiment 2: Soft Depth (13-15 hours)
jupyter notebook 04_Soft_Depth_Supervision.ipynb
# → Creates results/soft/model_soft.pth

# Experiment 3: Hard Depth (15-18 hours)
jupyter notebook 05_Hard_Depth_Sampling.ipynb
# → Creates results/hard/model_hard.pth

# Experiment 4: Hybrid (15-18 hours)
jupyter notebook 06_Hybrid_Strategy.ipynb
# → Creates results/hybrid/model_hybrid.pth
```

**Total training time:** 55-66 hours (2-3 days on single GPU)

**Tip:** Run in parallel on multiple GPUs if available

---

### Step 2: Generate Comparison Metrics

```bash
# After all 4 models are trained
jupyter notebook 07_Comprehensive_Evaluation.ipynb

# Outputs:
# - results/evaluation_results.npy
# - results/comparison_plot.png
# - results/psnr_curves.png
# - results/depth_error_maps/*.png
```

**Time:** 30-60 minutes

---

### Step 3: Render Videos

```bash
# Generate novel view videos for all strategies
jupyter notebook 08_Rendering_and_Visualization.ipynb

# Outputs:
# - results/baseline/renders/baseline.mp4
# - results/soft/renders/soft.mp4
# - results/hard/renders/hard.mp4
# - results/hybrid/renders/hybrid.mp4
# - results/comparison.mp4 (side-by-side)
```

**Time:** 1-2 hours

---

## 📦 **Download Pre-Trained Results**

If you want to skip training and directly evaluate/visualize:

**Google Drive:** [Download All Results (~600 MB)](https://drive.google.com/drive/folders/placeholder)

After downloading:

```bash
# Extract to results/ folder
unzip results.zip
mv results_backup/* results/

# Verify
ls results/baseline/model_baseline.pth
ls results/hybrid/renders/hybrid.mp4
```

Then jump to notebooks 07 (Evaluation) or 08 (Rendering).

---

## 📊 **Expected Metrics**

After running evaluation (notebook 07), you should see:

| Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth L1 ↓ |
|----------|--------|--------|---------|----------|
| Baseline | ~22.5 dB | ~0.90 | ~0.085 | N/A |
| Soft | ~22.4 dB | ~0.90 | ~0.089 | ~0.34 m |
| Hard | ~22.0 dB | ~0.90 | ~0.098 | ~0.28 m |
| Hybrid | ~22.1 dB | ~0.90 | ~0.092 | ~0.31 m |

*Your results may vary ±1-2 dB depending on random seed and training duration*

---

## 🔍 **Inspect Results**

### Check Model Weights

```python
import torch

# Load a trained model
checkpoint = torch.load('results/hybrid/model_hybrid.pth')
print(f"Model architecture: {checkpoint.keys()}")
```

### Plot Training Curves

```python
import numpy as np
import matplotlib.pyplot as plt

# Load training history
psnr_baseline = np.load('results/baseline/psnr_history.npy')
psnr_hybrid = np.load('results/hybrid/psnr_history.npy')

# Plot comparison
plt.plot(psnr_baseline, label='Baseline')
plt.plot(psnr_hybrid, label='Hybrid')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()
```

### View Rendered Videos

```bash
# Play video (macOS)
open results/hybrid/renders/hybrid.mp4

# Play video (Linux)
vlc results/hybrid/renders/hybrid.mp4

# Play video (Windows)
start results/hybrid/renders/hybrid.mp4
```

---

## 🛠️ **Troubleshooting**

### Problem: "FileNotFoundError: model_baseline.pth"

**Cause:** Model hasn't been trained yet

**Solution:**
```bash
# Train the model first
jupyter notebook 03_Baseline_NeRF.ipynb
```

---

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size in training notebooks

```python
# In 03, 04, 05, 06 notebooks, change:
batch_rays = 512  # Instead of 1024
N_samples = 32    # Instead of 64
```

---

### Problem: Training Too Slow

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
# GPU utilization should be 90-100%
```

**If low utilization:**
- Increase `batch_rays` (if memory allows)
- Check if CPU bottleneck (data loading)
- Ensure CUDA is properly installed

---

### Problem: Poor Results (Low PSNR)

**Possible causes:**
1. **Insufficient training** - Try 50K iterations instead of 20K
2. **Learning rate too high/low** - Default 5e-4 usually works
3. **Data not loaded correctly** - Check `01_Data_Loading.ipynb`
4. **Depth data quality** - Verify depth maps in dataset

---

## 📝 **For Course Reviewers**

If you're reviewing this project and want to verify results without retraining:

1. **Download pre-trained models** from Google Drive (link in main README)

2. **Place in results/ folder:**
   ```bash
   unzip results.zip -d results/
   ```

3. **Run evaluation:**
   ```bash
   jupyter notebook 07_Comprehensive_Evaluation.ipynb
   ```

4. **View videos:**
   ```bash
   open results/comparison.mp4
   ```

**Estimated time to verify:** 15-30 minutes

---

## 🔒 **Why Results are NOT in Git**

### Size Issues:
- Model weights: 20 MB total
- Videos: 500+ MB total
- GitHub limit: 100 MB per file, ~1 GB repo size

### Best Practice:
- **Code** (notebooks) → Git
- **Data** (datasets) → External download
- **Results** (models, videos) → Cloud storage (Google Drive)
- **Small artifacts** (plots) → Git (if <1 MB)

### Alternative Storage:
- **Git LFS** (Large File Storage) - Still problematic for 500+ MB
- **DVC** (Data Version Control) - Overkill for course project
- **Google Drive** - Simple and accessible ✅

---

## 🔗 **Access Full Results**

**Google Drive Folder Structure:**

```
NeRF-Depth-Research-Results/
├── trained_models/
│   ├── baseline.pth (5 MB)
│   ├── soft.pth (5 MB)
│   ├── hard.pth (5 MB)
│   └── hybrid.pth (5 MB)
│
├── rendered_videos/
│   ├── baseline.mp4 (100 MB)
│   ├── soft.mp4 (100 MB)
│   ├── hard.mp4 (100 MB)
│   ├── hybrid.mp4 (100 MB)
│   └── comparison.mp4 (200 MB)
│
├── evaluation_plots/
│   ├── psnr_comparison.png
│   ├── ssim_comparison.png
│   ├── training_curves.png
│   └── depth_error_maps.zip
│
└── README.txt
```

**Download Link:** [See main README.md](../README.md#research-artifacts)

---

## ℹ️ **Questions?**

**Issues with results generation:**
1. Check training logs in notebooks
2. Verify GPU availability: `nvidia-smi`
3. See [GitHub Issues](https://github.com/ayanali827/depth-supervised-nerf-notebooks/issues)
4. Contact: ayansaleem827@gmail.com

**Expected timeline:**
- Training all 4 models: 2-3 days (single GPU)
- Evaluation + rendering: 2-3 hours
- Total: ~3 days from start to completion

---

**Last Updated:** December 13, 2025  
**Author:** Ayan Ali  
**Project:** Improving NeRF Training Quality Using Depth Data
