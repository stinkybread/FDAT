# FDAT - FastDAT
DAT Inspired Lightweight Attention Network for Real World Image Super Resolution

This is an inspired super resolution arch based on various archs primarily DAT, PLKSR, SPAN and other.
Intended to be a faster version of DAT family of models while being visually and perceptually close.

Work created with the help of LLMs like Claude & Gemini

contact sharekhan. on the Enhance Everything Discord.

## Overview

FDAT (FastDAT) is a neural network architecture designed for image super-resolution tasks. It combines the strengths of both attention mechanisms and convolutional operations to effectively upscale low-resolution images while preserving fine details and structural information.

## Architecture Highlights

### 🔄 Dual Attention Mechanism
FDAT employs two complementary attention mechanisms:

- **Spatial Window Attention**: Processes spatial relationships within local windows, reducing computational complexity compared to full spatial attention while maintaining effectiveness
- **Channel Attention**: Focuses on inter-channel relationships to enhance feature representation and selection

### ⚡ Fast Implementation
The "Fast" in FDAT refers to several optimizations:

- **Window-based Spatial Attention**: Instead of computing attention across the entire spatial dimension, it uses local windows (default 8×8), significantly reducing computational cost
- **Simplified Components**: Streamlined implementations of attention and feed-forward modules
- **Efficient Channel Attention**: Uses normalized queries and keys for stable and fast channel-wise attention computation

### 🔗 Attention-Convolution Interaction
A key innovation is the **Simplified AIM (Attention Interaction Module)** that intelligently fuses:
- Attention-based global feature understanding
- Convolution-based local feature extraction

This dual pathway allows the model to capture both local textures and global structural patterns effectively.

## How It Works

### Architecture Flow
```
Input Image
    ↓
[Initial Convolution] - Embed input to feature space
    ↓
[Residual Groups] - Multiple groups of dual attention blocks
    ├── Spatial Window Attention + Conv → AIM Fusion
    ├── Channel Attention + Conv → AIM Fusion  
    └── FFN with Spatial Mixing
    ↓
[Final Convolution] - Feature refinement
    ↓
[Flexible Upsampler] - Multiple upsampling strategies
    ↓
Output HR Image
```

### Key Components

#### 1. **SimplifiedDATBlock**
The core building block that:
- Alternates between spatial and channel attention based on block pattern
- Runs parallel convolution pathway for local feature extraction
- Fuses attention and convolution features via AIM
- Applies feed-forward network with spatial mixing

#### 2. **FastSpatialWindowAttention**
- Divides input into non-overlapping windows
- Computes self-attention within each window
- Uses learnable relative position bias
- Handles padding for inputs not divisible by window size

#### 3. **FastChannelAttention**
- Normalizes queries and keys for stable training
- Uses temperature-scaled attention for better control
- Focuses on channel-wise feature relationships

#### 4. **SimplifiedFFN**
- Expands feature dimensions for enhanced representation
- Includes spatial mixing via depthwise convolution
- Applies dropout for regularization

#### 5. **Flexible Upsampling**
Supports multiple upsampling strategies:
- **PixelShuffle**: Efficient sub-pixel convolution
- **Transposed Convolution**: Learnable upsampling
- **Nearest + Conv**: Simple interpolation with refinement
- **DySample**: Advanced dynamic sampling (requires spandrel)

## Benefits for Super-Resolution

### 🎯 **Local-Global Feature Fusion**
- Convolution captures fine textures and local patterns
- Window attention captures spatial relationships and structures
- Channel attention enhances feature discriminability
- AIM module intelligently combines these complementary representations

### 🚀 **Computational Efficiency**
- Window-based attention scales linearly with image size
- Simplified implementations reduce overhead
- Parallel conv-attention pathways enable efficient processing

### 🔧 **Architectural Flexibility**
- Configurable attention patterns (spatial/channel alternation)
- Multiple model sizes (tiny, light, medium, large, xl)
- Various upsampling strategies for different use cases
- Adjustable window sizes and attention heads

### 📈 **Training Stability**
- Stochastic depth (DropPath) for regularization
- Normalized attention for stable gradients
- Residual connections preserve information flow
- Careful weight initialization

## Model Variants

| Model | Embed Dim | Groups | Depth/Group | Heads | Parameters* |
|-------|-----------|--------|-------------|--------|-------------|
| `fdat_tiny` | 96 | 2 | 2 | 3 | ~500K |
| `fdat_light` | 108 | 3 | 2 | 4 | ~800K |
| `fdat_medium` | 120 | 4 | 3 | 4 | ~1.5M |
| `fdat_large` | 180 | 4 | 4 | 6 | ~3.5M |
| `fdat_xl` | 180 | 6 | 6 | 6 | ~7M |

*Approximate parameter counts, varies with configuration

## Usage Example

```python
import torch
from fdat_arch import fdat_tiny, fdat_medium, FDAT

# Use predefined model
model = fdat_tiny(scale=4)

# Or create custom configuration
model = FDAT(
    embed_dim=96,
    num_groups=3,
    depth_per_group=2,
    num_heads=4,
    window_size=8,
    upsampler_type="pixelshuffle",
    scale=4
)

# Super-resolve an image
lr_image = torch.randn(1, 3, 64, 64)  # Low resolution input
with torch.no_grad():
    hr_image = model(lr_image)  # High resolution output (1, 3, 256, 256)
```

## Design Philosophy

FDAT represents a balanced approach to super-resolution that:

1. **Leverages both attention and convolution**: Rather than replacing convolutions entirely, it creates synergy between global attention and local convolution
2. **Prioritizes efficiency**: Uses window attention and optimized implementations to maintain practical computational requirements
3. **Maintains flexibility**: Supports various configurations and upsampling strategies for different use cases
4. **Focuses on stability**: Incorporates modern training techniques like stochastic depth and careful normalization

## Requirements

- PyTorch >= 1.9.0
- numpy
- spandrel (for DySample upsampler)

```bash
pip install torch numpy spandrel
```

## Applications

FDAT is particularly well-suited for:
- **Image Super-Resolution**: Primary use case for enhancing image resolution
- **Image Restoration**: Can be adapted for denoising and artifact removal
- **Real-time Applications**: Efficient architecture suitable for deployment
- **Research**: Flexible foundation for super-resolution research

The architecture's combination of efficiency, effectiveness, and flexibility makes it a strong choice for both practical applications and research in image enhancement tasks.
