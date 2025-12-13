#!/bin/bash

# Download NeRF Synthetic Dataset
# Based on original NeRF repository by Mildenhall et al.

cd "$(dirname "$0")"

echo "Downloading NeRF Synthetic Dataset..."
echo "This will create 'data/nerf_synthetic/' directory with all scenes"

# Create data directory
mkdir -p data
cd data

# Download from Google Drive (NeRF Synthetic Dataset)
# If this doesn't work, download manually from:
# https://drive.google.com/drive/folders/1JwYxcT-XDksuBi0DjG8aeRHbf5c-4eCR

echo ""
echo "Please download the NeRF Synthetic dataset from:"
echo "https://drive.google.com/drive/folders/1JwYxcT-XDksuBi0DjG8aeRHbf5c-4eCR"
echo ""
echo "Extract to: data/nerf_synthetic/"
echo ""
echo "The directory structure should be:"
echo "data/"
echo "├── nerf_synthetic/"
echo "│   ├── lego/"
echo "│   ├── chair/"
echo "│   ├── drums/"
echo "│   ├── ficus/"
echo "│   ├── hotdog/"
echo "│   ├── materials/"
echo "│   ├── mic/"
echo "│   └── ship/"
echo "└── llff/ (optional)"
echo ""

echo "After downloading and extracting, you can start training:"
echo "  jupyter notebook"
echo "  # Open 00_Setup_and_Dependencies.ipynb"
