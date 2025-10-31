# Qwen2.5-VL Fine-tuning on macOS with Apple Silicon

This guide covers fine-tuning Qwen2.5-VL-7B-Instruct on macOS with Apple Silicon (M1/M2/M3/M4).

## Important Note: MLX Limitations

**As of now, MLX does not have native Qwen2.5-VL support.** The provided notebook uses **PyTorch with MPS (Metal Performance Shaders)** backend instead, which provides excellent performance on Apple Silicon.

## System Requirements

### Hardware
- **Mac with Apple Silicon** (M1, M2, M3, or M4)
- **RAM**: 32GB minimum (64GB recommended for full dataset)
- **Storage**: 20GB+ free space

### Software
- macOS 12.0 or later
- Python 3.9+
- Xcode Command Line Tools

## Installation

### 1. Set up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install transformers and related packages
pip install transformers datasets accelerate peft pillow
pip install huggingface-hub

# Optional: Install MLX (for future use when Qwen2.5-VL is supported)
pip install mlx mlx-lm
```

## Data Preparation

1. Place your MathVerse data in a local directory:
   ```
   ~/Documents/Projects/mathverse/
   ├── mathverse_testmini.jsonl
   └── mathverse_testmini_images/
       ├── question_0000.png
       ├── question_0001.png
       └── ...
   ```

2. Update the data paths in the notebook:
   ```python
   DATA_DIR = "/Users/berta/Documents/Projects/mathverse"
   ```

## Usage

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook qwen2_5_vl_finetune_mlx.ipynb
```

Or use VSCode with Jupyter extension.

### Training Configuration

**For M4 (Base):**
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- Effective batch size = 8

**For M4 Pro/Max:**
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- Effective batch size = 8

**For M4 Max/Ultra (64GB+):**
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=2`
- Effective batch size = 8

### Expected Performance

**Training Speed (M4 Pro/Max):**
- ~30-60 seconds per step (depends on batch size)
- ~3-5 hours for full dataset (3940 samples, 3 epochs)

**Memory Usage:**
- Base model loading: ~15GB
- During training: ~20-25GB
- Peak: ~30GB

**Training Tips:**

1. **Monitor Memory:**
   ```bash
   # In terminal
   watch -n 1 "ps aux | grep python"
   ```
   Or use Activity Monitor

2. **Reduce Memory if Needed:**
   - Reduce `max_seq_length` to 1024 or 512
   - Use `batch_size=1`
   - Reduce LoRA rank `r` to 8

3. **Speed Up Training:**
   - Use smaller `max_samples` for testing (e.g., 100)
   - Reduce logging frequency
   - Use `eval_steps=100` instead of 50

## Differences from Colab Version

### What's Changed

1. **Device:**
   - Colab: CUDA/GPU
   - macOS: MPS (Metal Performance Shaders)

2. **Quantization:**
   - Colab: 4-bit quantization with bitsandbytes
   - macOS: float16 (MPS doesn't support 4-bit yet)

3. **Libraries:**
   - Colab: Unsloth for optimization
   - macOS: Standard PyTorch + PEFT

4. **Data Loading:**
   - Same format, but no Google Drive mounting

### What's the Same

- LoRA configuration
- Data format and preprocessing
- Training loop and evaluation
- Model architecture

## Troubleshooting

### Out of Memory Errors

**Solution 1: Reduce Batch Size**
```python
per_device_train_batch_size=1
gradient_accumulation_steps=16  # Increase this
```

**Solution 2: Reduce Sequence Length**
```python
MAX_SEQ_LENGTH = 1024  # Instead of 2048
```

**Solution 3: Reduce LoRA Rank**
```python
lora_config = LoraConfig(
    r=8,  # Instead of 16
    lora_alpha=16,  # Half of r*2
    ...
)
```

### MPS Errors

If you get MPS-related errors:
```python
# Fallback to CPU
device = torch.device("cpu")
model = model.to(device)
```

### Slow Training

- **Close other applications** to free up memory
- **Reduce `max_samples`** for testing
- **Use Activity Monitor** to check CPU/GPU usage

## Future: True MLX Support

When MLX adds Qwen2.5-VL support, you can use:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load_model, generate

# Load model in MLX format
model, processor = load_model("Qwen/Qwen2.5-VL-7B-Instruct")

# Training will be much faster on Apple Silicon
```

Monitor these repositories:
- https://github.com/ml-explore/mlx
- https://github.com/ml-explore/mlx-examples
- https://github.com/Blaizzy/mlx-vlm

## Comparison: MPS vs CUDA

| Feature | MPS (macOS) | CUDA (Colab) |
|---------|-------------|--------------|
| Speed | Good | Faster |
| Memory Efficiency | float16 only | 4-bit quantization |
| Cost | Free (local) | Free tier limited |
| Data Privacy | Local | Upload required |
| Model Size | ~15GB RAM | ~4GB VRAM (quantized) |

## Resources

- **PyTorch MPS Documentation**: https://pytorch.org/docs/stable/notes/mps.html
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **Qwen2-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **PEFT Documentation**: https://huggingface.co/docs/peft

## Support

For issues specific to:
- **MPS/macOS**: Check PyTorch MPS backend issues
- **MLX**: Wait for official Qwen2.5-VL support
- **Model/Training**: Same as Colab version
