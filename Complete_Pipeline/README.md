# Complete NeRF Depth Supervision Research Pipeline

**Research Project:** Improving NeRF Training Quality Using Depth Data  
**Author:** Ayan Ali  
**Institution:** ITMO University, Robotics & AI (2024-2026)  
**Date:** December 2025

---

## 📚 Table of Contents

1. [Environment Setup & Dependencies](#setup)
2. [Data Loading & Visualization](#data)
3. [Core NeRF Components](#core)
4. [Experiment 1: Baseline NeRF](#baseline)
5. [Experiment 2: Soft Depth Supervision](#soft)
6. [Experiment 3: Hard Depth Sampling](#hard)
7. [Experiment 4: Hybrid Strategy](#hybrid)
8. [Comprehensive Evaluation](#evaluation)
9. [Rendering & Visualization](#rendering)
10. [Results Summary](#summary)

---

## ℹ️ About This Notebook

This directory contains the **all-in-one complete pipeline notebook** that combines all 9 modular notebooks into a single executable workflow for the research project:

**"Comparing Strategies for Using Depth Data in NeRF Training"**

### Research Questions:
1. How effective is depth-guided sampling (hard constraint)?
2. How effective is depth MSE loss (soft constraint)?
3. Does hybrid approach outperform individual strategies?

### Technical Specifications:
- **Execution Time:** ~60-70 hours (for all 4 experiments)
- **GPU Required:** NVIDIA GPU with 8GB+ VRAM (V100, RTX 3080, or better)
- **Memory:** 16GB+ RAM recommended
- **Storage:** 10GB+ for data and results

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Quick Testing)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayanali827/depth-supervised-nerf-notebooks/blob/main/Complete_Pipeline/Complete_NeRF_Depth_Supervision_Pipeline.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → GPU (T4)
3. Run cells sequentially

**Note:** Colab free tier has 12-hour session limits. Save intermediate results to Google Drive!

---

### Option 2: Local Execution

```bash
# 1. Clone repository
git clone https://github.com/ayanali827/depth-supervised-nerf-notebooks.git
cd depth-supervised-nerf-notebooks/Complete_Pipeline

# 2. Install dependencies
pip install -r ../requirements.txt

# 3. Download data (if not already done)
cd ..
bash download_example_data.sh
cd Complete_Pipeline

# 4. Launch notebook
jupyter notebook Complete_NeRF_Depth_Supervision_Pipeline.ipynb

# 5. Run cells from top to bottom
```

---

## 📋 What's Inside

### Section 1: Environment Setup (5 min)
- Import all required libraries
- Configure GPU/CPU device
- Set random seeds for reproducibility
- Verify installation

### Section 2: Data Loading (10 min)
- Load NeRF Synthetic dataset (Lego scene)
- Visualize training images
- Inspect depth ground truth
- Prepare camera poses

### Section 3: Core Components (5 min)
- Positional encoding implementation
- NeRF MLP architecture
- Ray generation and volume rendering
- Utility functions

### Section 4: Experiment 1 - Baseline (12-15 hours)
- Vanilla NeRF training (RGB loss only)
- No depth supervision
- Establishes performance baseline

### Section 5: Experiment 2 - Soft Depth (13-15 hours)
- Add MSE depth loss to objective
- Soft constraint on expected depth
- Compare to baseline

### Section 6: Experiment 3 - Hard Depth (15-18 hours)
- Depth-guided ray sampling
- Free-space and surface concentration losses
- Hard geometric constraints

### Section 7: Experiment 4 - Hybrid (15-18 hours)
- Combine soft and hard strategies
- Balanced sampling + dual loss
- Best overall performance

### Section 8: Comprehensive Evaluation (30-60 min)
- PSNR, SSIM, LPIPS metrics
- Depth accuracy measurements
- Statistical comparison
- Visualization of results

### Section 9: Rendering & Visualization (1-2 hours)
- Novel view synthesis for all strategies
- Generate 360° rotation videos
- Side-by-side comparisons
- GIF animations

### Section 10: Results Summary
- Quantitative comparison table
- Key findings and conclusions
- Recommendations for applications

---

## 📊 Expected Results

After running the complete pipeline, you will obtain:

### Quantitative Metrics:

| Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth L1 ↓ | Training Time |
|----------|--------|--------|---------|-----------|---------------|
| Baseline | ~22.5 dB | ~0.90 | ~0.085 | N/A | 12-15 hours |
| Soft | ~22.4 dB | ~0.90 | ~0.089 | ~0.34 m | 13-15 hours |
| Hard | ~22.0 dB | ~0.90 | ~0.098 | ~0.28 m | 15-18 hours |
| Hybrid | ~22.1 dB | ~0.90 | ~0.092 | ~0.31 m | 15-18 hours |

*Note: Results may vary ±1-2 dB depending on random seed and training duration*

### Visual Outputs:
- 4 trained NeRF models (.pth files)
- 4 rendered 360° videos (.mp4)
- Comparison plots and figures
- Depth error visualizations

---

## 💡 Usage Tips

### For First-Time Users:

1. **Run sequentially** - Execute cells from top to bottom
2. **Expect long runtime** - Plan for 60-70 hours total
3. **Save frequently** - Download intermediate checkpoints
4. **Monitor GPU** - Check utilization with `nvidia-smi`

### For Advanced Users:

- **Skip experiments**: Modify skip flags to load pre-trained models
- **Adjust hyperparameters**: Edit training parameters in each section
- **Run specific sections**: Use table of contents to navigate
- **Batch processing**: Run experiments in parallel on multiple GPUs

### For Google Colab Users:

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Save results to Drive
results_dir = '/content/drive/MyDrive/NeRF_Results'
os.makedirs(results_dir, exist_ok=True)
```

### For Local Users:

- Ensure CUDA 11.7+ is installed
- Download NeRF Synthetic dataset first (~2.5 GB)
- Allocate 10GB+ disk space for results
- Consider tmux/screen for long training sessions

---

## 🛠️ Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```python
# Reduce batch size
batch_rays = 512  # instead of 1024
N_samples = 32    # instead of 64
```

**2. Training Too Slow**
```bash
# Verify GPU usage
watch -n 1 nvidia-smi
# Should show 90-100% GPU utilization
```

**3. Data Not Found**
```bash
# Download dataset manually
cd ..
bash download_example_data.sh
ls data/nerf_synthetic/lego/  # Verify
```

**4. Colab Session Timeout**
- Save checkpoints every 5000 iterations
- Download results to local machine
- Use Colab Pro for longer sessions

---

## 📁 Output Structure

After running the complete pipeline:

```
results/
├── baseline/
│   ├── model_baseline.pth
│   ├── psnr_history.npy
│   └── renders/
│       └── baseline.mp4
│
├── soft/
│   ├── model_soft.pth
│   └── renders/
│       └── soft.mp4
│
├── hard/
│   ├── model_hard.pth
│   └── renders/
│       └── hard.mp4
│
├── hybrid/
│   ├── model_hybrid.pth
│   └── renders/
│       └── hybrid.mp4
│
├── evaluation_results.npy
├── comparison_plot.png
└── comparison.mp4
```

**Total size:** ~600 MB

---

## 🔗 Related Resources

### Modular Notebooks:

For more control and easier debugging, use individual notebooks:

- [00_Setup_and_Dependencies.ipynb](../00_Setup_and_Dependencies.ipynb)
- [01_Data_Loading.ipynb](../01_Data_Loading.ipynb)
- [02_Core_Components.ipynb](../02_Core_Components.ipynb)
- [03_Baseline_NeRF.ipynb](../03_Baseline_NeRF.ipynb)
- [04_Soft_Depth_Supervision.ipynb](../04_Soft_Depth_Supervision.ipynb)
- [05_Hard_Depth_Sampling.ipynb](../05_Hard_Depth_Sampling.ipynb)
- [06_Hybrid_Strategy.ipynb](../06_Hybrid_Strategy.ipynb)
- [07_Comprehensive_Evaluation.ipynb](../07_Comprehensive_Evaluation.ipynb)
- [08_Rendering_and_Visualization.ipynb](../08_Rendering_and_Visualization.ipynb)

### Documentation:

- [Main README](../README.md) - Complete project documentation
- [THEORY.md](../THEORY.md) - Mathematical background
- [Data README](../data/README.md) - Dataset instructions
- [Results README](../results/README.md) - Output structure

### External Links:

- **Original NeRF Paper:** https://www.matthewtancik.com/nerf
- **NeRF Synthetic Dataset:** https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
- **Project Repository:** https://github.com/ayanali827/depth-supervised-nerf-notebooks

---

## 📝 Citation

If you use this implementation in your research:

```bibtex
@misc{ali2025depthnerf,
  title={Research on Improving NeRF Training Quality Using Depth Data},
  author={Ali, Ayan},
  year={2025},
  institution={ITMO University},
  howpublished={\url{https://github.com/ayanali827/depth-supervised-nerf-notebooks}}
}
```

---

## 🎯 Key Findings

### Main Conclusions:

1. **Depth supervision improves geometric accuracy** at minimal cost to RGB quality
2. **Hard constraints** provide best depth accuracy (0.28m L1 error)
3. **Soft constraints** preserve photometric quality better
4. **Hybrid approach** balances both objectives - recommended for robotics/AR
5. **Computational overhead** is 10-35% compared to baseline

### Recommended Use Cases:

- **Baseline**: When only RGB quality matters
- **Soft**: Easy to implement, modest improvements
- **Hard**: When geometry is critical (3D reconstruction)
- **Hybrid**: Production applications (AR, robotics, autonomous navigation)


