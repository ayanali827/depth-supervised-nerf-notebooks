# Depth-Supervised NeRF: Neural Radiance Fields with Depth Guidance

### [Project Page](#) | [Paper](#) | [Video](#) | [Data](#data) | [Colab](#)

Jupyter notebook implementation of depth-supervised neural radiance fields for improved 3D scene reconstruction.

**Depth-Supervised NeRF: Improved Scene Reconstruction with Depth Guidance**

Ayan Ali

## TL;DR quickstart

To setup a conda environment, download example training data, and launch Jupyter:

```bash
conda env create -f environment.yml
conda activate nerf-depth
bash download_example_data.sh
jupyter notebook
```

Then open and run the notebooks starting with `00_Setup_and_Dependencies.ipynb`.

## Setup

Python 3.8+ dependencies:

- PyTorch >= 2.0.0
- numpy
- imageio
- matplotlib  
- scikit-image
- tqdm
- jupyter
- lpips (for evaluation metrics)

Create the conda environment by running:

```bash
conda env create -f environment.yml
conda activate nerf-depth
```

## What is a Depth-Supervised NeRF?

A neural radiance field is a fully connected network trained to reproduce input views of a single scene. Our extension incorporates **depth supervision** to improve geometric consistency:

**Network Architecture**: The network maps from spatial location and viewing direction (5D input) to color and opacity (4D output), enabling differentiable volume rendering of novel views.

**Depth Supervision Strategies**:
- **Baseline**: RGB loss only
- **Soft**: MSE loss on predicted depth 
- **Hard**: Depth-guided ray sampling
- **Hybrid**: Combined soft + hard (recommended)

Optimizing a NeRF takes between a few hours and a day (depending on resolution) and requires a single GPU. Rendering from an optimized NeRF takes from less than a second to ~30 seconds per view.

## Notebooks

Our implementation is organized as self-contained Jupyter notebooks:

### Core Notebooks (Main Implementation)

**00_Setup_and_Dependencies.ipynb** - Environment setup, GPU configuration, imports

**01_Data_Loading.ipynb** - Load NeRF Synthetic dataset, data inspection, visualization

**02_Core_Components.ipynb** - Positional encoding (PosEnc), NeRF architecture, volume rendering

**03_Baseline_NeRF.ipynb** - Vanilla NeRF (RGB loss only), training, evaluation

**04_Soft_Depth_Supervision.ipynb** - Soft depth loss (MSE-based), training with depth

**05_Hard_Depth_Sampling.ipynb** - Hard depth-guided sampling, surface concentration

**06_Hybrid_Strategy.ipynb** - Combined soft + hard approach, best results

**07_Comprehensive_Evaluation.ipynb** - PSNR, SSIM, LPIPS, depth metrics, cross-strategy comparison

**08_Rendering_and_Visualization.ipynb** - Novel view rendering, GIF/MP4 generation, visualizations

### Analysis & Experiments

**Analysis/** - Ablation studies, hyperparameter sweeps, scene comparisons

**Experiments/** - Custom data, extended training, model distillation

## Running code

### Quickstart

After installing dependencies and downloading data:

```bash
jupyter notebook
```

Open and run the notebooks in order:

```
00_Setup_and_Dependencies.ipynb
  ↓
01_Data_Loading.ipynb  
  ↓
02_Core_Components.ipynb
  ↓
03_Baseline_NeRF.ipynb (or 04_Soft, 05_Hard, 06_Hybrid)
  ↓
07_Comprehensive_Evaluation.ipynb
  ↓
08_Rendering_and_Visualization.ipynb
```

### Training a NeRF

**Baseline NeRF (RGB only)**:

Open `03_Baseline_NeRF.ipynb`. After ~200k iterations (15-20 hours on V100), you'll have a trained model with results in `results/baseline/`.

**With Depth Supervision**:

Open `04_Soft_Depth_Supervision.ipynb` (soft depth loss), `05_Hard_Depth_Sampling.ipynb` (guided sampling), or `06_Hybrid_Strategy.ipynb` (combined).

### Rendering a NeRF

Open `08_Rendering_and_Visualization.ipynb` to:
- Render novel views
- Generate GIF/MP4 videos  
- Visualize depth maps
- Create publication figures

### Evaluating Results

Open `07_Comprehensive_Evaluation.ipynb` to compute:
- PSNR, SSIM, LPIPS metrics
- Depth accuracy measurements
- Cross-strategy comparisons

## Results Comparison

| Strategy | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth L1 ↓ |
|----------|--------|--------|---------|----------|
| Baseline | 22.47 | 0.903 | 0.085 | — |
| Soft | 22.41 | 0.901 | 0.089 | 0.34 m |
| Hard | 21.96 | 0.896 | 0.098 | 0.28 m |
| **Hybrid** | **22.14** | **0.899** | **0.092** | **0.31 m** |

## Data

Download example data:

```bash
bash download_example_data.sh
```

Or download manually from:
- **NeRF Synthetic Dataset**: [Google Drive](https://drive.google.com/drive/folders/1JwYxcT-XDksuBi0DjG8aeRHbf5c-4eCR)
- **LLFF Real-World Data**: Same link as above
- **Custom Data**: Use COLMAP to compute camera poses

Dataset structure:

```
data/
├── nerf_synthetic/
│   ├── lego/
│   │   ├── transforms_train.json
│   │   ├── transforms_test.json
│   │   └── images/
│   ├── chair/
│   └── ...
└── llff/
    ├── fern/
    └── ...
```

## Customization

### Hyperparameters

In any strategy notebook, modify:

```python
num_iters = 20000          # Training iterations
batch_rays = 1024          # Rays per batch  
learning_rate = 5e-4       # Learning rate
num_samples = 64           # Samples per ray
lambda_depth = 0.01        # Depth weight (soft)
lambda_freespace = 0.005   # Freespace weight (hard)
```

### Scenes

```python
scene = 'lego'  # Change to 'chair', 'drums', 'mic', 'materials', 'hotdog'
```

### Architecture

```python
model = NeRF(
    D=8,                    # Network depth
    W=256,                  # Network width  
    in_channels_xyz=63,     # Position encoding channels
    in_channels_dir=27,     # Direction encoding channels
    skips=[4]               # Skip connection layers
)
```

## Advanced: Custom Data

For your own scenes, you need:

1. **Poses**: 3×4 camera-to-world transformation matrices
2. **Intrinsics**: Height, width, focal length
3. **Images**: RGB PNG files
4. **Depth** (optional): For depth supervision

Generate poses using [COLMAP](https://colmap.github.io/) with `imgs2poses.py` from [LLFF](https://github.com/Fyusion/LLFF).

## Troubleshooting

**Out of memory**:
- Reduce `batch_rays` from 1024 to 512
- Use `halfres=True` in data loading
- Reduce `num_samples` from 64 to 32

**Slow training**:  
- Ensure GPU is being used
- Check batch size is not too small
- Verify no expensive ops in training loop

**Poor results**:
- Try `hybrid` strategy (usually best)
- Increase training iterations
- Check depth ground truth quality

## Extracting Geometry

See `extract_mesh.ipynb` for marching cubes extraction:

```bash
pip install trimesh pyrender PyMCubes
```

## Citation

If you use this code, please cite the original NeRF paper:

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  year={2020},
  booktitle={ECCV},
}
```

And our depth supervision extension:

```bibtex
@misc{ali2025depthnerf,
  title={Depth-Supervised Neural Radiance Fields},
  author={Ali, Ayan},
  year={2025},
  howpublished={\url{https://github.com/ayanali827/depth-supervised-nerf-notebooks}}
}
```

## Acknowledgments

Based on the original [NeRF implementation](https://github.com/bmild/nerf) by Mildenhall et al. Extended with depth supervision strategies for improved geometric consistency.

## License

MIT License
