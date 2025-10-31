# Project Setup Guide

This guide covers setting up the fine-tuning environment for different platforms.

## Table of Contents
- [M4 MacBook with MLX (Recommended)](#m4-macbook-with-mlx-recommended)
- [Google Colab with Unsloth](#google-colab-with-unsloth)
- [M4 MacBook with PyTorch MPS (Fallback)](#m4-macbook-with-pytorch-mps-fallback)

---

## M4 MacBook with MLX (Recommended)

### Option 1: Using uv (Recommended)

**Step 1: Install uv**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**Step 2: Create virtual environment and install dependencies**
```bash
cd /Users/berta/Projects/experiment/fine-tune

# Create virtual environment with Python 3.11+
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements-mlx.txt

# Verify MLX installation
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
python -c "from mlx_vlm import load; print('MLX-VLM ready!')"
```

**Step 3: Run the notebook**
```bash
# Install Jupyter (if not already installed)
uv pip install notebook ipywidgets

# Start Jupyter
jupyter notebook qwen2_5_vl_finetune_mlx_native.ipynb
```

### Option 2: Using traditional venv + pip

```bash
cd /Users/berta/Projects/experiment/fine-tune

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements-mlx.txt

# Verify installation
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

### Why uv is better for this project:

| Feature | uv | pip |
|---------|----|----|
| Speed | 10-100x faster | Baseline |
| Dependency resolution | Advanced | Basic |
| Disk usage | Efficient caching | Redundant downloads |
| ARM64 support | Native | Via wheel |
| Installation time | ~15 seconds | ~3-4 minutes |

**Note:** PyTorch and torchvision are required even for MLX, as Qwen2.5-VL's processor uses PyTorch for image/video preprocessing. The actual model inference runs on MLX.

---

## Google Colab with Unsloth

**No virtual environment needed** - Colab manages its own environment.

**Steps:**

1. Upload `qwen2_5_vl_finetune.ipynb` to Google Colab
2. Upload data to Google Drive at: `MyDrive/Colab Notebooks/data/`
3. Run installation cells in the notebook (they handle all dependencies)

**Important**: Do NOT manually install packages. The notebook uses a specific installation order to avoid conflicts:
- Upgrades bitsandbytes first
- Installs unsloth and dependencies without deps
- Pins compatible versions of transformers and trl

---

## M4 MacBook with PyTorch MPS (Fallback)

Use this if MLX-VLM doesn't work or you need PyTorch compatibility.

### Using uv:
```bash
cd /Users/berta/Projects/experiment/fine-tune

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install PyTorch with MPS support
uv pip install torch torchvision torchaudio

# Install other dependencies
uv pip install transformers datasets accelerate peft pillow scikit-learn tqdm huggingface-hub

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Using traditional venv:
```bash
cd /Users/berta/Projects/experiment/fine-tune

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install torch torchvision torchaudio
```

---

## Quick Start Commands

### For MLX (M4 MacBook) - Recommended workflow:
```bash
# One-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /Users/berta/Projects/experiment/fine-tune
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-mlx.txt

# Every time you work on the project
cd /Users/berta/Projects/experiment/fine-tune
source .venv/bin/activate
jupyter notebook qwen2_5_vl_finetune_mlx_native.ipynb
```

### For Colab:
```bash
# Just upload the notebook and run it!
# No local setup needed
```

---

## Troubleshooting

### uv: command not found
```bash
# Make sure uv is in your PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Add to ~/.zshrc or ~/.bashrc permanently
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
```

### MLX-VLM import error
```bash
# Verify Python version (needs 3.9+)
python --version

# Reinstall MLX-VLM
uv pip install --upgrade mlx-vlm
```

### MPS not available
```bash
# Check macOS version (needs 12.3+)
sw_vers

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

---

## Updating Dependencies

### With uv:
```bash
# Update all packages
uv pip install --upgrade -r requirements-mlx.txt

# Update specific package
uv pip install --upgrade mlx-vlm
```

### With pip:
```bash
# Update all packages
pip install --upgrade -r requirements-mlx.txt
```

---

## Deactivating Virtual Environment

```bash
# When done working
deactivate
```

---

## Performance Comparison

### Installation Speed (M4 MacBook)

| Method | Time to install MLX dependencies |
|--------|----------------------------------|
| uv | ~10 seconds |
| pip | ~2-3 minutes |

### Training Speed

| Platform | Tokens/sec | Memory Usage | Cost |
|----------|-----------|--------------|------|
| M4 Pro (MLX) | 200-400 | 20-30GB RAM | Free (local) |
| M4 Max (MLX) | 300-500 | 30-40GB RAM | Free (local) |
| Colab T4 (Unsloth) | 300-800 | 4-8GB VRAM | Free tier limited |
| Colab A100 (Unsloth) | 800-1500 | 8-16GB VRAM | Paid |

---

## Recommended Setup by Use Case

### Primary Development (M4 MacBook):
```bash
✅ Use uv + MLX-VLM (requirements-mlx.txt)
- Fastest installation
- Native Apple Silicon support
- Best for iteration and testing
```

### Training Large Batches (Google Colab):
```bash
✅ Use Colab + Unsloth (qwen2_5_vl_finetune.ipynb)
- Free GPU access
- Faster for large datasets
- 4-bit quantization support
```

### Cross-Platform Compatibility:
```bash
✅ Use PyTorch MPS (requirements.txt)
- Works on any Mac with Apple Silicon
- Fallback if MLX has issues
```

---

## Next Steps

After setup:

1. **Test installation**:
   ```bash
   python -c "import mlx.core as mx; from mlx_vlm import load; print('✅ Ready!')"
   ```

2. **Update data paths** in the notebook:
   ```python
   DATA_DIR = "/Users/berta/Documents/Projects/mathverse"
   ```

3. **Start training**:
   ```bash
   jupyter notebook qwen2_5_vl_finetune_mlx_native.ipynb
   ```

---

## Resources

- **uv Documentation**: https://docs.astral.sh/uv/
- **MLX-VLM**: https://github.com/Blaizzy/mlx-vlm
- **MLX**: https://ml-explore.github.io/mlx/
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
