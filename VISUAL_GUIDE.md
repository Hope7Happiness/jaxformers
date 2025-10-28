# 🎨 JAXFormers API Visual Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                      JAXFormers Library                          │
│                   Unified Model Interface                        │
└─────────────────────────────────────────────────────────────────┘

                              │
                              ▼
                    
        ┌────────────────────────────────────┐
        │     import jaxformers              │
        └────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
                
    ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
    │ list_models() │  │ model_info() │  │print_models()│
    │               │  │              │  │              │
    │ Get all model │  │ Get details  │  │ Pretty print │
    │ names         │  │ about model  │  │ all models   │
    └───────────────┘  └──────────────┘  └──────────────┘
                              │
                              ▼
                              
                    ┌──────────────────┐
                    │ create_model()   │
                    │  - model_name    │
                    │  - pretrained    │
                    │  - **kwargs      │
                    └──────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
                
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │  ResNet    │  │  ConvNeXt  │  │   ViT      │
        │            │  │            │  │            │
        │ • resnet18 │  │ • tiny     │  │ • MAE      │
        │ • resnet34 │  │ • small    │  │ • DeiT     │
        │ • resnet50 │  │ • base     │  │ • DINOv2   │
        │ • resnet101│  │ • large    │  │            │
        │ • resnet152│  │            │  │            │
        └────────────┘  └────────────┘  └────────────┘
```

---

## 🎯 API Flow Diagram

```
User Code
    │
    │  import jaxformers
    │  
    ├──────────────────────────────────────────────┐
    │                                              │
    │  Discover Models                             │  Create Models
    │  ───────────────                             │  ─────────────
    │                                              │
    │  jaxformers.list_models()                    │  jaxformers.create_model()
    │  jaxformers.list_models('resnet')            │         │
    │  jaxformers.model_info('resnet50')           │         │
    │  jaxformers.print_models()                   │         │
    │                                              │         ▼
    └──────────────────────────────────────────────┤  Model Registry
                                                   │  ───────────────
                                                   │  • Check if exists
                                                   │  • Get module name
                                                   │  • Get class name
                                                   │  • Get config
                                                   │         │
                                                   │         ▼
                                                   │  Lazy Load Module
                                                   │  ────────────────
                                                   │  • Import module
                                                   │  • Get class
                                                   │  • Cache result
                                                   │         │
                                                   │         ▼
                                                   │  Create Instance
                                                   │  ───────────────
                                                   │  • Merge configs
                                                   │  • Instantiate
                                                   │  • Return model
                                                   │         │
                                                   │         ▼
                                                   └──► Flax Model
```

---

## 🔄 Example Usage Flow

### Flow 1: Discovery
```
┌─────────┐      ┌──────────────┐      ┌──────────────┐
│  User   │─────▶│list_models() │─────▶│ Model Names  │
└─────────┘      └──────────────┘      └──────────────┘
                                              │
                                              ▼
                                        ['resnet50',
                                         'dinov2_vitb14',
                                         'mae_vit_base',
                                         ...]
```

### Flow 2: Creation
```
┌─────────┐      ┌────────────────────┐      ┌──────────┐
│  User   │─────▶│create_model('name')│─────▶│  Model   │
└─────────┘      └────────────────────┘      └──────────┘
                           │
                           ├──▶ Check registry
                           ├──▶ Load module
                           ├──▶ Get class
                           ├──▶ Create instance
                           └──▶ Return model
```

### Flow 3: Complete Workflow
```
1. Discovery
   └─▶ jaxformers.print_models('vit')
        └─▶ See available Vision Transformer models

2. Get Info
   └─▶ jaxformers.model_info('dinov2_vitb14')
        └─▶ Read description, paper, config

3. Create Model
   └─▶ model = jaxformers.create_model('dinov2_vitb14')
        └─▶ Get Flax model instance

4. Use Model
   └─▶ params = model.init(key, x)
   └─▶ output = model.apply(params, x)
```

---

## 📊 Model Registry Structure

```
MODEL_REGISTRY = {
    "model_name": {
        "module": "python_module_name",     # Which file contains the model
        "class": "ClassName",               # Class name to instantiate
        "config": {...},                    # Default configuration
        "description": "Human readable",    # For documentation
        "paper": "Paper title",             # Citation
    }
}

Example:
────────
"dinov2_vitb14": {
    "module": "dino",                      # From dino.py
    "class": "DinoVisionTransformer",      # Class to create
    "config": {"variant": "vitb14"},       # Default args
    "description": "DINOv2 ViT-Base...",  # Info
    "paper": "DINOv2: Learning...",       # Reference
}
```

---

## 🎨 Code Pattern Comparison

### Before (Manual)
```python
# User needs to know module structure
from models.dino import DinoVisionTransformer
from models.resnet import ResNet50
from models.mae import MAE

# Different initialization patterns
dino_model = DinoVisionTransformer(variant='vitb14')
resnet_model = ResNet50()
mae_model = MAE(variant='base')
```

### After (With JAXFormers)
```python
# Simple, unified interface
import jaxformers

# Same pattern for everything!
dino_model = jaxformers.create_model('dinov2_vitb14')
resnet_model = jaxformers.create_model('resnet50')
mae_model = jaxformers.create_model('mae_vit_base')
```

---

## 🚀 Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│                    JAXFormers Cheat Sheet                 │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  📋 List Models                                           │
│  ───────────────────────────────────────────────────     │
│  jaxformers.list_models()         # All models           │
│  jaxformers.list_models('resnet') # Filtered             │
│                                                           │
│  ℹ️  Get Info                                             │
│  ───────────────────────────────────────────────────     │
│  jaxformers.model_info('resnet50')                       │
│  jaxformers.print_models()        # Pretty display       │
│                                                           │
│  🚀 Create Model                                          │
│  ───────────────────────────────────────────────────     │
│  model = jaxformers.create_model('resnet50')             │
│  model = jaxformers.create_model('dinov2_vitb14')        │
│                                                           │
│  🎯 Model Families                                        │
│  ───────────────────────────────────────────────────     │
│  • ResNet:   resnet18, resnet50, resnet152, ...          │
│  • ConvNeXt: convnext_tiny, convnext_base, ...           │
│  • MAE:      mae_vit_base, mae_vit_large, ...            │
│  • DeiT:     deit_tiny, deit_small, deit_base            │
│  • DINOv2:   dinov2_vits14, dinov2_vitb14, ...           │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 💡 Design Philosophy

```
┌────────────────────────────────────────────┐
│         Simplicity                         │
│  ────────────────────────────────────      │
│  One function to rule them all:            │
│  create_model()                            │
└────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│         Discoverability                    │
│  ────────────────────────────────────      │
│  Easy to find what's available:            │
│  list_models(), print_models()             │
└────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│         Consistency                        │
│  ────────────────────────────────────      │
│  Same interface for all models:            │
│  Unified creation pattern                  │
└────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│         Extensibility                      │
│  ────────────────────────────────────      │
│  Simple to add new models:                 │
│  Just update registry                      │
└────────────────────────────────────────────┘
```

---

**🎉 Welcome to JAXFormers - Where Model Creation is Simple & Elegant!**
