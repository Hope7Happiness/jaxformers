# 🚀 Quick Start Guide

Welcome to **JAXFormers**! This guide will get you up and running in minutes.

## Installation

```bash
# Install from source
cd jaxformers
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# If you need PyTorch for weight conversion
pip install -e ".[torch]"
```

## Basic Usage

### 1️⃣ Import the Library

```python
import jaxformers
```

### 2️⃣ Discover Available Models

```python
# See all models with descriptions
jaxformers.print_models()

# List model names
models = jaxformers.list_models()
print(models)

# Filter models
resnet_models = jaxformers.list_models('resnet')
dino_models = jaxformers.list_models('dino')
```

### 3️⃣ Create a Model

```python
# Simple creation
model = jaxformers.create_model('resnet50')

# With custom config
model = jaxformers.create_model('dinov2_vitb14', num_classes=1000)

# Create with pretrained weights (when available)
model = jaxformers.create_model('mae_vit_base', pretrained=True)
```

### 4️⃣ Initialize and Use

```python
import jax
import jax.numpy as jnp

# Create model
model = jaxformers.create_model('resnet50')

# Initialize parameters
key = jax.random.PRNGKey(0)
x = jnp.ones((1, 224, 224, 3))  # [batch, height, width, channels]
variables = model.init(key, x)

# Forward pass
output = model.apply(variables, x)
print(f"Output shape: {output.shape}")
```

## Common Use Cases

### 🖼️ Image Classification

```python
import jax
import jax.numpy as jnp
import jaxformers

# Load model
model = jaxformers.create_model('resnet50', num_classes=1000)

# Initialize
key = jax.random.PRNGKey(42)
image = jnp.ones((1, 224, 224, 3))
params = model.init(key, image)

# Inference
logits = model.apply(params, image)
predictions = jax.nn.softmax(logits)
```

### 🎭 Feature Extraction

```python
import jaxformers

# DINOv2 is excellent for feature extraction
model = jaxformers.create_model('dinov2_vitb14')

# Get features instead of classifications
# (implementation depends on model architecture)
```

### 🔄 Transfer Learning

```python
import jaxformers

# Start with pretrained model
model = jaxformers.create_model('convnext_base', pretrained=True)

# Fine-tune on your dataset
# (add your training loop here)
```

## Available Model Families

### ResNet
```python
jaxformers.create_model('resnet18')
jaxformers.create_model('resnet50')
jaxformers.create_model('resnet101')
```

### ConvNeXt
```python
jaxformers.create_model('convnext_tiny')
jaxformers.create_model('convnext_base')
jaxformers.create_model('convnext_large')
```

### Vision Transformers (MAE)
```python
jaxformers.create_model('mae_vit_base')
jaxformers.create_model('mae_vit_large')
jaxformers.create_model('mae_vit_huge')
```

### DeiT
```python
jaxformers.create_model('deit_tiny')
jaxformers.create_model('deit_small')
jaxformers.create_model('deit_base')
```

### DINOv2
```python
jaxformers.create_model('dinov2_vits14')
jaxformers.create_model('dinov2_vitb14')
jaxformers.create_model('dinov2_vitl14')
jaxformers.create_model('dinov2_vitg14')
```

## Tips & Best Practices

1. **Use JIT compilation** for faster inference:
   ```python
   @jax.jit
   def predict(params, x):
       return model.apply(params, x)
   ```

2. **Batch processing** with vmap:
   ```python
   batch_predict = jax.vmap(lambda x: model.apply(params, x[None])[0])
   ```

3. **Check model info** before creating:
   ```python
   info = jaxformers.model_info('dinov2_vitb14')
   print(info)
   ```

4. **Filter models** to find what you need:
   ```python
   transformer_models = jaxformers.list_models('vit')
   ```

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 💡 Check out [examples.py](examples.py) for more usage patterns
- 🔧 Explore individual model files for architecture details
- 🤝 Contribute new models to the registry

## Need Help?

- 📮 Open an issue on GitHub
- 📚 Check the API reference in README.md
- 💬 Join our community discussions

---

**Happy coding with JAXFormers! 🎉**
