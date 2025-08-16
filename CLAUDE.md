# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**diffusion-pipe** is a pipeline parallel training framework for diffusion models supporting 15+ state-of-the-art models including SDXL, Flux, LTX-Video, HunyuanVideo, Cosmos, Wan, and others. The codebase uses DeepSpeed for distributed training and includes advanced memory optimization techniques.

## Core Architecture

### Training Pipeline
- **Main Entry**: `train.py` - Comprehensive training script with extensive argument parsing
- **Model System**: Modular architecture in `models/` with:
  - `base_model.py` - Abstract base class defining the model interface
  - Model-specific implementations (e.g., `flux_model.py`, `sdxl_model.py`)
  - Each model handles its own forward pass, loss calculation, and parameter management

### Key Components
- **Pipeline Management**: `utils/pipeline_utils.py` - DeepSpeed pipeline parallelism setup
- **Dataset Handling**: `utils/dataset_utils.py` - Unified dataset interface for images/videos
- **Memory Optimization**: Custom optimizers in `utils/optimizers.py` (Automagic, 8-bit variants)
- **Caching System**: Latent and text embedding caching for performance

### Configuration System
- TOML-based configuration files in `configs/` directory
- Each model has its own config template with model-specific parameters
- Pipeline stages and parallelism settings configured per model

## Development Commands

### Environment Setup
```bash
# Create environment
conda create -n diffusion-pipe python=3.11
conda activate diffusion-pipe

# Install dependencies
pip install -r requirements.txt

# For RTX 4000 series GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# For Cosmos models (optional)
pip install transformer-engine
```

### Training
```bash
# Basic training
torchrun --nproc_per_node=2 train.py --config configs/sdxl.toml

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=2 --master_addr=<addr> --master_port=<port> train.py --config configs/flux.toml

# Override config parameters
python train.py --config configs/ltx_video.toml --model.learning_rate 1e-5 --model.batch_size 4
```

### Testing and Debugging
```bash
# Test DeepSpeed initialization
python test_deepspeed_init.py

# Debug memory issues
python train.py --config configs/debug.toml --model.gradient_checkpointing true --model.cpu_offloading true
```

### Data Preparation
```bash
# Cache latents (recommended for repeated training)
python train.py --config configs/sdxl.toml --cache_latents true --cache_only true

# Cache text embeddings
python train.py --config configs/flux.toml --cache_text_embeddings true --cache_only true
```

## Model Addition Framework

To add a new model:

1. **Create model class** in `models/new_model.py`:
   - Inherit from `BaseModel`
   - Implement `forward()`, `compute_loss()`, `get_model_parameters()`
   - Handle model-specific tokenization and encoding

2. **Add configuration** in `configs/new_model.toml`:
   - Define pipeline stages for parallelism
   - Set model-specific hyperparameters
   - Configure memory optimization settings

3. **Register model** in `train.py`:
   - Add import and model mapping
   - Ensure proper initialization handling

## Memory Optimization Strategies

### Low VRAM Training
- Enable gradient checkpointing: `--model.gradient_checkpointing true`
- Use CPU offloading: `--model.cpu_offloading true`
- Reduce batch size: `--model.batch_size 1`
- Cache latents to avoid re-encoding: `--cache_latents true`

### Pipeline Parallelism
- Models are automatically split across available GPUs
- Stage configuration in model TOML files defines split points
- DeepSpeed handles gradient synchronization and optimizer states

## Dataset Integration

### Supported Formats
- **Images**: Standard image datasets with caption files
- **Videos**: Frame sequences or video files with metadata
- **Bucket Sampling**: Automatic aspect ratio bucketing for efficiency

### Configuration
```toml
[dataset]
path = "path/to/dataset"
batch_size = 4
resolution = [1024, 1024]
enable_bucket_sampler = true
```

## Common Patterns

### Error Handling
- Models implement graceful fallbacks for missing components
- Pipeline parallelism includes automatic stage balancing
- Memory errors trigger automatic optimization suggestions

### Performance Optimization
- Latent caching reduces VAE encoding overhead
- Text embedding caching eliminates repeated tokenization
- Mixed precision training enabled by default

### Distributed Training
- Automatic GPU detection and pipeline setup
- Stage-wise gradient accumulation
- Optimizer state partitioning across devices

## Environment Variables

```bash
# Memory management (RTX 4000 series)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DeepSpeed optimization
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_FUSED_ADAM=1

# Flash Attention (if installed)
export FLASH_ATTENTION_FORCE_FP16=1
```

## Dependencies

- PyTorch 2.7.1+ with CUDA support
- DeepSpeed for pipeline parallelism
- flash-attn 2.8.1 for attention optimization
- transformers, diffusers for model components
- accelerate for distributed training utilities