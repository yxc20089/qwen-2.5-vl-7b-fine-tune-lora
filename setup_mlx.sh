#!/bin/bash
# Quick setup script for MLX-VLM on M4 MacBook
# Usage: bash setup_mlx.sh

set -e  # Exit on error

echo "🚀 Setting up Qwen2.5-VL Fine-tuning with MLX-VLM"
echo "=================================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo ""
fi

echo "✅ uv is installed: $(uv --version)"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    uv venv --python 3.11
    echo ""
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo ""

# Install dependencies
echo "📦 Installing dependencies (this will be fast with uv)..."
uv pip install -r requirements-mlx.txt
echo ""

# Install Jupyter if not present
echo "📓 Installing Jupyter..."
uv pip install notebook ipywidgets
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python -c "import mlx.core as mx; print(f'✅ MLX version: {mx.__version__}')"
python -c "from mlx_vlm import load; print('✅ MLX-VLM is ready!')"
echo ""

echo "=================================================="
echo "🎉 Setup complete!"
echo ""
echo "To start working:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Start Jupyter: jupyter notebook qwen2_5_vl_finetune_mlx_native.ipynb"
echo "  3. Update DATA_DIR in the notebook to your data path"
echo ""
echo "Deactivate when done: deactivate"
echo "=================================================="
