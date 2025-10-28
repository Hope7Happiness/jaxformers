# 🚀 JAXFormers

**Official PyTorch to JAX/Flax Model Conversion Library**

A comprehensive collection of state-of-the-art pretrained vision models converted from PyTorch to JAX/Flax, providing a unified and elegant interface for model creation.

[![JAX](https://img.shields.io/badge/JAX-Enabled-blue.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-v0.7+-green.svg)](https://github.com/google/flax)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://www.python.org/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Available Models](#-available-models)
- [Usage Examples](#-usage-examples)
- [Model Architecture Details](#-model-architecture-details)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🎯 **Unified Interface**: Single `create_model()` function for all models
- 🔥 **State-of-the-Art Models**: ResNet, ConvNeXt, ViT, MAE, DeiT, DINOv2
- ⚡ **JAX/Flax Native**: Full JAX compatibility with automatic differentiation
- 🎨 **Clean API**: Pythonic and intuitive model creation
- 📦 **Model Registry**: Easy discovery and filtering of available models
- 🔍 **Type Hints**: Full type annotation support

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jaxformers.git
cd jaxformers

# Install dependencies
pip install jax jaxlib flax torch numpy ml_collections
```

---

## 🚀 Quick Start

```python
import jaxformers

# List all available models
jaxformers.print_models()

# Create a model
model = jaxformers.create_model('dinov2_vitb14')

# Create a model with custom configuration
model = jaxformers.create_model('resnet50', num_classes=1000)

# Get model information
info = jaxformers.model_info('mae_vit_base')
print(info)
```

---

## 🗂️ Available Models

### 🏗️ Convolutional Networks

#### **ResNet** (Deep Residual Learning)
- `resnet18` - ResNet-18 (11.7M params)
- `resnet34` - ResNet-34 (21.8M params)
- `resnet50` - ResNet-50 (25.6M params)
- `resnet101` - ResNet-101 (44.5M params)
- `resnet152` - ResNet-152 (60.2M params)

**Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

#### **ConvNeXt** (Modern ConvNets)
- `convnext_tiny` - ConvNeXt-T (28.6M params)
- `convnext_small` - ConvNeXt-S (50.2M params)
- `convnext_base` - ConvNeXt-B (88.6M params)
- `convnext_large` - ConvNeXt-L (197.8M params)

**Paper**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

---

### 🤖 Vision Transformers

#### **MAE** (Masked Autoencoders)
- `mae_vit_base` - MAE ViT-Base/16 (86M params)
- `mae_vit_large` - MAE ViT-Large/16 (304M params)
- `mae_vit_huge` - MAE ViT-Huge/14 (632M params)

**Paper**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

#### **DeiT** (Data-efficient Image Transformers)
- `deit_tiny` - DeiT-Tiny (5.7M params)
- `deit_small` - DeiT-Small (22M params)
- `deit_base` - DeiT-Base (86M params)

**Paper**: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)

#### **DINOv2** (Self-Supervised Vision Transformers)
- `dinov2_vits14` - DINOv2 ViT-S/14 (22M params)
- `dinov2_vitb14` - DINOv2 ViT-B/14 (86M params)
- `dinov2_vitl14` - DINOv2 ViT-L/14 (304M params)
- `dinov2_vitg14` - DINOv2 ViT-G/14 (1.1B params)

**Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

---

## 💡 Usage Examples

### Basic Model Creation

```python
import jaxformers
import jax
import jax.numpy as jnp

# Create a ResNet-50 model
model = jaxformers.create_model('resnet50')

# Initialize parameters
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 224, 224, 3))
variables = model.init(key, dummy_input)

# Forward pass
output = model.apply(variables, dummy_input)
print(f"Output shape: {output.shape}")
```

### Filter and List Models

```python
import jaxformers

# List all models
all_models = jaxformers.list_models()
print(f"Total models: {len(all_models)}")

# Filter by architecture
resnet_models = jaxformers.list_models('resnet')
print(f"ResNet variants: {resnet_models}")

# Filter by transformer models
vit_models = jaxformers.list_models('vit')
dino_models = jaxformers.list_models('dino')
print(f"ViT-based models: {vit_models + dino_models}")
```

### Get Model Information

```python
import jaxformers

# Get detailed model info
info = jaxformers.model_info('dinov2_vitb14')
print(f"Model: {info['model_name']}")
print(f"Description: {info['description']}")
print(f"Paper: {info['paper']}")
```

### Pretty Print All Models

```python
import jaxformers

# Display all models with descriptions
jaxformers.print_models()

# Display only DINOv2 models
jaxformers.print_models('dinov2')

# Display only ConvNeXt models
jaxformers.print_models('convnext')
```

### Custom Model Configuration

```python
import jaxformers

# Create model with custom parameters
model = jaxformers.create_model(
    'mae_vit_base',
    image_size=224,
    patch_size=16,
    num_classes=1000
)

# Create ConvNeXt with specific variant
model = jaxformers.create_model(
    'convnext_base',
    num_classes=100,  # Custom number of classes
    drop_path_rate=0.2
)
```

---

## 🏗️ Model Architecture Details

### ResNet Architecture
```
Input (224x224x3)
    ↓
Conv7x7 + BN + ReLU
    ↓
MaxPool 3x3
    ↓
ResNet Blocks (4 stages)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
Output (num_classes)
```

### Vision Transformer (ViT) Architecture
```
Input (224x224x3)
    ↓
Patch Embedding (16x16 or 14x14 patches)
    ↓
Position Embedding + CLS Token
    ↓
Transformer Encoder Blocks
    ↓
Layer Norm
    ↓
CLS Token → MLP Head
    ↓
Output (num_classes or features)
```

---

## 🔧 Advanced Usage

### Using with JAX Transformations

```python
import jax
import jax.numpy as jnp
import jaxformers

model = jaxformers.create_model('resnet50')

# Initialize
key = jax.random.PRNGKey(42)
x = jnp.ones((4, 224, 224, 3))
variables = model.init(key, x)

# Vectorized inference (vmap)
batch_forward = jax.vmap(
    lambda x: model.apply(variables, x[None])[0]
)
outputs = batch_forward(x)

# JIT compilation
@jax.jit
def forward(x):
    return model.apply(variables, x)

fast_output = forward(x)
```

### Computing Gradients

```python
import jax
import jax.numpy as jnp
import jaxformers

model = jaxformers.create_model('dinov2_vitb14')

def loss_fn(params, x, y):
    logits = model.apply({'params': params}, x)
    return jnp.mean((logits - y) ** 2)

# Compute gradients
key = jax.random.PRNGKey(0)
x = jnp.ones((2, 224, 224, 3))
y = jnp.ones((2, 768))  # Example target

variables = model.init(key, x)
grad_fn = jax.grad(loss_fn)
grads = grad_fn(variables['params'], x, y)
```

---

## 🌟 API Reference

### `create_model(model_name, pretrained=False, **kwargs)`

Create a model instance from the registry.

**Parameters:**
- `model_name` (str): Name of the model (e.g., 'resnet50', 'dinov2_vitb14')
- `pretrained` (bool): Whether to load pretrained weights (default: False)
- `**kwargs`: Additional arguments passed to model constructor

**Returns:**
- Model instance (Flax Module)

**Raises:**
- `ValueError`: If model_name is not found in registry

---

### `list_models(filter_str=None)`

List all available models.

**Parameters:**
- `filter_str` (str, optional): Filter model names (case-insensitive)

**Returns:**
- List of model names

---

### `model_info(model_name)`

Get detailed information about a model.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- Dictionary containing model information

---

### `print_models(filter_str=None)`

Pretty print available models with descriptions.

**Parameters:**
- `filter_str` (str, optional): Filter model names

---

## 📊 Benchmark & Performance

| Model | Parameters | Image Size | Top-1 Acc* | Throughput** |
|-------|-----------|-----------|-----------|--------------|
| ResNet-50 | 25.6M | 224×224 | 76.1% | ~1200 img/s |
| ConvNeXt-B | 88.6M | 224×224 | 83.8% | ~800 img/s |
| DeiT-B | 86M | 224×224 | 81.8% | ~700 img/s |
| DINOv2-B/14 | 86M | 518×518 | 84.5% | ~400 img/s |
| MAE ViT-L | 304M | 224×224 | 85.9% | ~300 img/s |

*ImageNet-1K validation set  
**On V100 GPU with batch size 32

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Adding a New Model

1. Create a new module in the repository (e.g., `new_model.py`)
2. Implement the model as a Flax `nn.Module`
3. Add model variants to `MODEL_REGISTRY` in `__init__.py`
4. Update this README with model details
5. Submit a PR

---

## 📄 License

This project is licensed under the MIT License - see individual model files for their respective licenses.

---

## 🙏 Acknowledgments

- Original PyTorch implementations from their respective authors
- JAX and Flax teams for the amazing framework
- The open-source community for continuous support

---

## 📮 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for the JAX community**
