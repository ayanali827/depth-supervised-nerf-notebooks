# Data Directory

This directory contains the datasets used for training and evaluating Neural Radiance Fields with depth supervision.

---

## 🚨 **Important: Data is NOT Included in Repository**

The datasets are **NOT stored in Git** due to their large size (~2.5 GB). You must download them before running experiments.

---

## 📥 **Quick Download (Recommended)**

### Option 1: Automatic Download Script

Run the provided script from the repository root:

```bash
cd ..
bash download_example_data.sh
```

This will automatically:
1. Download the NeRF Synthetic dataset
2. Extract to `data/nerf_synthetic/`
3. Verify the structure

**Time:** ~5-10 minutes (depending on internet speed)

---

### Option 2: Manual Download

If the script doesn't work, download manually:

**NeRF Synthetic Dataset (Main Dataset):**

1. **Download Link:** [Google Drive - NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

2. **Direct Download (if available):**
   ```bash
   # Download nerf_synthetic.zip (2.2 GB)
   wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG' -O nerf_synthetic.zip
   
   # Or use gdown
   pip install gdown
   gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
   ```

3. **Extract:**
   ```bash
   unzip nerf_synthetic.zip
   mv nerf_synthetic data/
   ```

---

## 📁 **Expected Directory Structure**

After downloading, your `data/` folder should look like this:

```
data/
├── README.md                    ← This file
└── nerf_synthetic/
    ├── lego/                    ← Main scene for experiments
    │   ├── train/
    │   │   ├── r_0.png
    │   │   ├── r_1.png
    │   │   ├── ...
    │   │   └── r_99.png          (100 images)
    │   ├── test/
    │   │   ├── r_0.png
    │   │   ├── ...
    │   │   └── r_199.png         (200 images)
    │   ├── val/
    │   │   └── ...               (100 images)
    │   ├── transforms_train.json
    │   ├── transforms_test.json
    │   └── transforms_val.json
    │
    ├── chair/                   ← Optional: for scene comparison
    ├── drums/
    ├── ficus/
    ├── hotdog/
    ├── materials/
    ├── mic/
    └── ship/
```

---

## ✅ **Verify Installation**

Check that data downloaded correctly:

```bash
# From repository root
ls -lh data/nerf_synthetic/lego/train/ | head

# Should show 100 PNG files (r_0.png through r_99.png)
```

**Or run the verification in Python:**

```python
from pathlib import Path

data_dir = Path("data/nerf_synthetic/lego")

print(f"Train images: {len(list((data_dir / 'train').glob('*.png')))}")
print(f"Test images: {len(list((data_dir / 'test').glob('*.png')))}")
print(f"Val images: {len(list((data_dir / 'val').glob('*.png')))}")

# Expected output:
# Train images: 100
# Test images: 200
# Val images: 100
```

---

## 📊 **Dataset Details**

### NeRF Synthetic Dataset

**Description:**
- 8 synthetic scenes rendered with Blender
- Each scene: 400 total images (100 train, 200 test, 100 val)
- Resolution: 800x800 pixels
- Ground truth: Camera poses, depth maps, surface normals

**Scenes Included:**
1. **lego** - Bulldozer toy (main scene for our experiments)
2. **chair** - Office chair
3. **drums** - Drum set
4. **ficus** - Potted plant
5. **hotdog** - Food item
6. **materials** - Reflective spheres
7. **mic** - Microphone
8. **ship** - Pirate ship

**Format:**
- Images: PNG (RGBA)
- Camera poses: JSON files (`transforms_*.json`)
- Depth maps: Available in depth channel

**Citation:**
```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and 
          Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={ECCV},
  year={2020}
}
```

---

## 🛠️ **Troubleshooting**

### Problem: Download Script Fails

```bash
# If wget/curl fails, install gdown:
pip install gdown
gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
```

### Problem: "No such file or directory"

```bash
# Make sure you're in the repository root:
pwd  # Should end with "depth-supervised-nerf-notebooks"

# Create data directory if missing:
mkdir -p data
```

### Problem: Insufficient Disk Space

**Required Space:**
- Download: ~2.2 GB (compressed)
- Extracted: ~2.5 GB
- Total needed: ~5 GB (temporarily for extraction)

```bash
# Check available space:
df -h .

# If low on space, download only "lego" scene:
# (Manual download required - see Option 2 above)
```

### Problem: Slow Download

**Alternative mirrors:**
- Original authors' website: https://www.matthewtancik.com/nerf
- Academic torrents: (check NeRF GitHub issues)
- Request from course instructor

---

## 🔒 **Why Data is NOT in Git**

1. **Size Limitation**
   - Dataset: 2.5 GB
   - GitHub limit: 100 MB per file
   - Would make repository unusable

2. **Best Practice**
   - Data should be separate from code
   - Data can change independently
   - Keeps repository lightweight

3. **Already Available**
   - Dataset is publicly hosted
   - No need to duplicate
   - Original source is authoritative

4. **Git Performance**
   - Large files slow down git operations
   - Binary files don't compress well
   - Git LFS would still be problematic

---

## 📝 **For Course Reviewers**

If you're reviewing this project:

1. **Download data first:**
   ```bash
   bash download_example_data.sh
   ```

2. **Verify structure:**
   ```bash
   ls data/nerf_synthetic/lego/
   ```

3. **Run notebooks** in order (00 → 08)

**Estimated setup time:** 5-10 minutes

---

## 🔗 **Additional Datasets (Optional)**

### LLFF Real-World Dataset

For testing on real captured scenes:

```bash
# Download LLFF dataset (~1.5 GB)
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_llff_data.zip
unzip nerf_llff_data.zip
mv nerf_llff_data data/llff
```

**Structure:**
```
data/llff/
├── fern/
├── flower/
├── fortress/
├── horns/
├── leaves/
├── orchids/
├── room/
└── trex/
```

---

## ℹ️ **Need Help?**

**Issues with data download:**
1. Check [GitHub Issues](https://github.com/ayanali827/depth-supervised-nerf-notebooks/issues)
2. Contact: ayansaleem827@gmail.com
3. See original NeRF repo: https://github.com/bmild/nerf

**Alternative download methods:**
- Google Drive web interface (manual download)
- Request via course forum
- Use university VPN if network restricted

---

**Last Updated:** December 13, 2025  
**Maintained by:** Ayan Ali  
**Dataset Source:** [NeRF: Neural Radiance Fields](https://www.matthewtancik.com/nerf)
