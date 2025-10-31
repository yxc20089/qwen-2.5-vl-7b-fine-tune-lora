# Qwen2.5-VL Fine-tuning with Native MLX (MLX-VLM)

This guide uses **MLX-VLM** for truly native MLX fine-tuning on Apple Silicon.

## What is MLX-VLM?

MLX-VLM is a package for inference and fine-tuning of Vision Language Models on Mac using Apple's MLX framework. It provides **native support for Qwen2.5-VL**.

- **Repository**: https://github.com/Blaizzy/mlx-vlm
- **Supported Models**: LLaVA, Qwen2-VL, Qwen2.5-VL, PaliGemma, Phi-3-Vision, and more

## Why MLX-VLM instead of PyTorch?

| Feature | MLX-VLM | PyTorch MPS |
|---------|---------|-------------|
| Framework | Native MLX | PyTorch on Metal |
| Speed | **Faster** (optimized for Apple Silicon) | Good |
| Memory | More efficient | Uses more RAM |
| Integration | Apple Silicon optimized | Cross-platform |
| LoRA Support | Built-in | Via PEFT |

## System Requirements

### Hardware
- **Mac with Apple Silicon** (M1, M2, M3, or M4)
- **RAM**:
  - M4: 16GB minimum, 32GB recommended
  - M4 Pro/Max: 32GB+ recommended
- **Storage**: 20GB+ free space

### Software
- macOS 13.0 or later
- Python 3.9+
- Xcode Command Line Tools

## Installation

### 1. Install MLX-VLM

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install MLX-VLM
pip install mlx-vlm

# Install PyTorch (required for Qwen2.5-VL processor)
pip install torch torchvision

# Install additional packages
pip install pillow datasets scikit-learn tqdm
```

**Note:** PyTorch and torchvision are required even though we use MLX for training. Qwen2.5-VL's processor uses PyTorch for image/video preprocessing, while the actual model inference and training run on MLX.

### 2. Verify Installation

```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
python -c "from mlx_vlm import load; print('MLX-VLM ready!')"
```

## Data Preparation

Your data should be in JSONL format:

```json
{
  "image": "path/to/image.jpg",
  "prompt": "Question text",
  "answer": "Answer text"
}
```

The notebook automatically converts MathVerse format to this structure.

## Usage

### Quick Start

```bash
# Open the notebook
jupyter notebook qwen2_5_vl_finetune_mlx_native.ipynb
```

Run cells sequentially. Key steps:

1. **Install dependencies** (Cell 1)
2. **Load data** (Cells 3-4)
3. **Load model** (Cell 4)
4. **Fine-tune** (Cell 6) - Uses MLX-VLM CLI
5. **Evaluate** (Cell 8)

### Training via Python API (Recommended)

MLX-VLM 0.3.5 uses a Python API for training:

```python
from mlx_vlm import load
from mlx_vlm.trainer.trainer import Trainer, TrainingArgs
from mlx_vlm.trainer.utils import build_dataset
import mlx.optimizers as optim

# Load model
model, processor = load("Qwen/Qwen2.5-VL-7B-Instruct")

# Prepare datasets
train_dataset = build_dataset("train.jsonl", "Qwen/Qwen2.5-VL-7B-Instruct", processor)
val_dataset = build_dataset("val.jsonl", "Qwen/Qwen2.5-VL-7B-Instruct", processor)

# Training configuration
training_args = TrainingArgs(
    batch_size=1,
    iters=300,
    steps_per_report=10,
    steps_per_eval=50,
    adapter_file="adapters.safetensors"
)

# Create optimizer and trainer
optimizer = optim.Adam(learning_rate=1e-5)
trainer = Trainer(model=model, optimizer=optimizer)

# Train
trainer.train(train_dataset, val_dataset, training_args)
```

### Generate with Fine-tuned Model

```python
from mlx_vlm import load, generate
from PIL import Image

# Load model with adapters
model, processor = load(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    adapter_path="adapters.safetensors"
)

# Generate
image = Image.open("test.jpg")
response = generate(
    model,
    processor,
    "Your question here",
    image,
    max_tokens=128
)
```

## Training Configuration

### Recommended Settings

**For M4 (16GB RAM):**
```bash
--batch-size 1
--iters 300
--lora-layers 16
--learning-rate 1e-5
```

**For M4 Pro/Max (32GB+ RAM):**
```bash
--batch-size 2
--iters 500
--lora-layers 16
--learning-rate 1e-5
```

### Parameters Explained

- `--batch-size`: Number of samples per batch
- `--iters`: Total training iterations
- `--learning-rate`: Learning rate (1e-5 to 2e-5 recommended)
- `--lora-layers`: LoRA rank (8, 16, or 32)
- `--adapter-file`: Where to save LoRA adapters
- `--val-data`: Validation data for evaluation during training

## Performance Benchmarks

### M4 (16GB RAM)
- **Training Speed**: ~100-200 tokens/sec
- **Memory Usage**: ~15-20GB
- **Time for 300 iters**: ~30-45 minutes
- **Batch Size**: 1

### M4 Pro (36GB RAM)
- **Training Speed**: ~200-350 tokens/sec
- **Memory Usage**: ~20-30GB
- **Time for 300 iters**: ~20-30 minutes
- **Batch Size**: 2

### M4 Max (64GB RAM)
- **Training Speed**: ~300-500 tokens/sec
- **Memory Usage**: ~30-40GB
- **Time for 300 iters**: ~15-25 minutes
- **Batch Size**: 4

## Memory Optimization

If you run out of memory:

### 1. Reduce LoRA Rank
```bash
--lora-layers 8  # Instead of 16
```

### 2. Use Quantized Model
```bash
# Convert to 4-bit quantized
mlx_vlm.convert --hf-path Qwen/Qwen2.5-VL-7B-Instruct -q

# Use quantized model for training (QLoRA)
mlx_vlm.lora --model Qwen/Qwen2.5-VL-7B-Instruct-4bit ...
```

### 3. Reduce Batch Size
```bash
--batch-size 1
```

### 4. Limit Sequence Length
Edit the config to reduce max sequence length from 2048 to 1024.

## Loading Fine-tuned Models

### Option 1: Load with Adapters

```python
from mlx_vlm import load, generate
from PIL import Image

# Load model with adapters
model, processor = load(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    adapter_path="adapters.safetensors"
)

# Generate
image = Image.open("test.jpg")
response = generate(
    model,
    processor,
    image,
    "Your question here",
    max_tokens=128
)
```

### Option 2: Load Fused Model

```python
# After running mlx_vlm.fuse
model, processor = load("./fused_model")

response = generate(
    model,
    processor,
    image,
    "Your question here"
)
```

## Troubleshooting

### Import Error: "No module named 'mlx_vlm'"

```bash
pip install --upgrade mlx-vlm
```

### Out of Memory During Training

1. Reduce `--batch-size` to 1
2. Use 4-bit quantized model
3. Reduce `--lora-layers` to 8
4. Close other applications

### Slow Training

- **Check Activity Monitor**: Ensure CPU/GPU are being used
- **Reduce logging**: Less frequent validation
- **Use smaller dataset**: Test with `max_samples=100` first

### Model Not Found

```bash
# Download manually first
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

## Comparison: MLX-VLM vs Unsloth (Colab)

| Feature | MLX-VLM (macOS) | Unsloth (Colab) |
|---------|-----------------|-----------------|
| Speed | 200-500 tok/s (M4 Pro/Max) | 300-800 tok/s (A100) |
| Memory | 20-30GB RAM | 4-8GB VRAM |
| Quantization | 4-bit (QLoRA) | 4-bit |
| Cost | Free (local) | Free tier limited |
| Privacy | Local | Cloud |
| Setup | Easy | Very easy |
| Portability | macOS only | Any browser |

## Advanced Usage

### Resume Training

```bash
mlx_vlm.lora \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train \
  --data train.jsonl \
  --resume-adapter-file adapters.safetensors  # Resume from checkpoint
  ...
```

### Custom Validation

```bash
mlx_vlm.lora \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train \
  --data train.jsonl \
  --val-data val.jsonl \
  --test-batches 20  # Validate every 20 batches
  ...
```

### Upload to Hugging Face

You can upload your adapters to Hugging Face Hub:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="adapters.safetensors",
    path_in_repo="adapters.safetensors",
    repo_id="your-username/qwen2.5-vl-mathverse-lora",
    repo_type="model"
)
```

## Examples

### Fine-tune on Custom Dataset

```python
# 1. Prepare data
import json

data = [
    {
        "image": "images/q1.jpg",
        "prompt": "What is the answer?",
        "answer": "A"
    },
    # ... more samples
]

with open('custom_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\\n')

# 2. Train
!mlx_vlm.lora \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train \
    --data custom_train.jsonl \
    --iters 500

# 3. Test
from mlx_vlm import load, generate
model, processor = load(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    adapter_path="adapters.safetensors"
)
```

## Resources

- **MLX-VLM GitHub**: https://github.com/Blaizzy/mlx-vlm
- **MLX Framework**: https://ml-explore.github.io/mlx/
- **MLX Examples**: https://github.com/ml-explore/mlx-examples
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **Discussions**: https://github.com/Blaizzy/mlx-vlm/discussions

## Community

- Report issues: https://github.com/Blaizzy/mlx-vlm/issues
- Join discussions: https://github.com/Blaizzy/mlx-vlm/discussions
- MLX Community: https://ml-explore.github.io/mlx/

## License

Check licenses for:
- MLX-VLM: Apache 2.0
- Qwen2.5-VL: Check Hugging Face model card
- Your data: Ensure you have rights to use
