# Qwen2.5-VL-7B Fine-tuning on MathVerse Dataset

This repository contains code to fine-tune the Qwen2.5-VL-7B-Instruct vision-language model on the MathVerse dataset using Unsloth for efficient training on Google Colab.

## Overview

- **Model**: Qwen2.5-VL-7B-Instruct (Vision-Language Model)
- **Dataset**: MathVerse (Mathematical reasoning with visual diagrams)
- **Framework**: Unsloth for efficient fine-tuning with LoRA
- **Platform**: Google Colab (GPU required)

## Files

- `qwen2_5_vl_finetune.ipynb` - Main training notebook
- `README.md` - This file

## Prerequisites

### Data Preparation

1. Download or prepare your MathVerse dataset with the following structure:
   ```
   mathverse/
   ├── mathverse_testmini.jsonl
   └── mathverse_testmini_images/
       ├── question_0000.png
       ├── question_0001.png
       └── ...
   ```

2. Upload the data to Google Drive:
   - Create a folder in your Google Drive (e.g., `MyDrive/mathverse`)
   - Upload `mathverse_testmini.jsonl`
   - Upload the `mathverse_testmini_images` folder

### Google Colab Setup

1. Open the notebook in Google Colab
2. Enable GPU runtime:
   - Go to: Runtime → Change runtime type
   - Select GPU (T4, V100, or A100)
   - Click Save

## Usage

### Step-by-Step Instructions

1. **Open the Notebook**
   - Upload `qwen2_5_vl_finetune.ipynb` to Google Colab
   - Or use: File → Upload notebook

2. **Mount Google Drive**
   - Run the cell in Section 2 to mount your Google Drive
   - Update the `DATA_DIR` path to point to your data location

3. **Install Dependencies**
   - Run Section 1 to install required packages
   - This will install Unsloth, transformers, and other dependencies

4. **Configure Training**
   - In Section 3, adjust `max_samples` parameter:
     - Set to 100 for quick testing
     - Set to `None` for full dataset training
   - In Section 7, adjust training hyperparameters:
     - `num_train_epochs`: Number of training epochs (default: 3)
     - `per_device_train_batch_size`: Batch size per GPU (default: 1)
     - `gradient_accumulation_steps`: Gradient accumulation (default: 4)
     - `learning_rate`: Learning rate (default: 2e-5)

5. **Run Training**
   - Execute cells sequentially from Section 1 to Section 9
   - Training progress will be displayed with loss metrics

6. **Save the Model**
   - Section 10 saves the LoRA adapters
   - Section 11 (optional) merges LoRA weights with base model

7. **Test the Model**
   - Section 12 runs inference on a validation sample
   - Compare model output with expected answer

## Training Configuration

### Default Settings

```python
# Model Settings
Model: Qwen2.5-VL-7B-Instruct
Max Sequence Length: 2048
Quantization: 4-bit (reduce VRAM usage)

# LoRA Settings
Rank (r): 16
Alpha: 32
Dropout: 0.05
Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

# Training Settings
Epochs: 3
Batch Size: 1
Gradient Accumulation: 4 (effective batch size = 4)
Learning Rate: 2e-5
Warmup Ratio: 0.1
Weight Decay: 0.01
Optimizer: AdamW 8-bit
```

### Memory Requirements

- **T4 GPU (16GB)**: Use 4-bit quantization, batch size 1
- **V100 GPU (16GB)**: Use 4-bit quantization, batch size 1-2
- **A100 GPU (40GB)**: Can use larger batch sizes or full precision

### Adjusting for Your GPU

If you encounter out-of-memory errors:

1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length` to 1024 or 512
4. Reduce LoRA rank `r` to 8 or 4

## Expected Training Time

- **100 samples** (testing): ~10-15 minutes on T4
- **1000 samples**: ~1-2 hours on T4
- **Full dataset (~3940 samples)**: ~4-6 hours on T4

## Output

After training, you'll have:

1. **LoRA Adapters**: Lightweight adapter weights (~100-200MB)
   - Location: `./qwen2.5-vl-mathverse-final`
   - Use with base model for inference

2. **Merged Model** (optional): Full model with merged weights (~15GB)
   - Location: `./qwen2.5-vl-mathverse-merged`
   - Standalone model, no need for base model

## Inference

### Using LoRA Adapters

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Load base model
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "./qwen2.5-vl-mathverse-final"
)

processor = AutoProcessor.from_pretrained("./qwen2.5-vl-mathverse-final")
```

### Using Merged Model

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./qwen2.5-vl-mathverse-merged"
)
processor = AutoProcessor.from_pretrained("./qwen2.5-vl-mathverse-merged")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size to 1
   - Increase gradient accumulation steps
   - Use 4-bit quantization
   - Reduce max sequence length

2. **Slow Training**
   - Reduce `max_samples` for testing
   - Use larger GPU (V100 or A100)
   - Reduce logging frequency

3. **Data Not Found**
   - Check Google Drive paths
   - Verify files uploaded correctly
   - Check permissions

4. **Package Installation Errors**
   - Restart runtime: Runtime → Restart runtime
   - Re-run installation cells
   - Check Colab GPU availability

## Advanced Options

### Using Weights & Biases for Logging

1. Install wandb: `!pip install wandb`
2. Login: `wandb.login()`
3. Change `report_to="wandb"` in training args

### Multi-GPU Training

If using Colab Pro with multiple GPUs:

```python
training_args = TrainingArguments(
    ...
    # Add these lines
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,
)
```

### Custom Data Format

To use your own dataset, modify the `load_mathverse_data` function in Section 3:

```python
def load_custom_data(data_path):
    # Load your data
    # Return list of dicts with: 'image', 'question', 'answer'
    pass
```

## Dataset Information

The MathVerse dataset contains:
- 3,940 samples in testmini split
- Multiple problem versions (Text Dominant, Vision Dominant, etc.)
- Question types: multi-choice and free-form
- Subject: Mathematical reasoning with diagrams
- Source: GeoQA and other math problem sources

## Resources

- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [MathVerse Dataset](https://mathverse-coder.github.io/)
- [Google Colab](https://colab.research.google.com/)

## License

Please check the licenses for:
- Qwen2.5-VL model
- MathVerse dataset
- Unsloth framework

## Citation

If you use this code, please cite:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues:
- Unsloth: [GitHub Issues](https://github.com/unslothai/unsloth/issues)
- Qwen2.5-VL: [Hugging Face Discussions](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions)
