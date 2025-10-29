"""
JAXFormers: Official PyTorch to JAX Model Conversion Library

A collection of state-of-the-art pretrained vision models converted to JAX/Flax.
Supports seamless model loading with a unified interface.
"""

from typing import Dict, Any, Optional, Callable
import importlib
from functools import lru_cache

__version__ = "0.1.1"
__all__ = ["create_model", "list_models", "model_info"]


# Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ResNet Models
    "resnet18": {
        "module": "resnet",
        "class": "ResNet18",
        "description": "ResNet-18 architecture (Flax implementation)",
        "paper": "Deep Residual Learning for Image Recognition",
    },
    "resnet34": {
        "module": "resnet",
        "class": "ResNet34",
        "description": "ResNet-34 architecture (Flax implementation)",
        "paper": "Deep Residual Learning for Image Recognition",
    },
    "resnet50": {
        "module": "resnet",
        "class": "ResNet50",
        "description": "ResNet-50 architecture (Flax implementation)",
        "paper": "Deep Residual Learning for Image Recognition",
    },
    "resnet101": {
        "module": "resnet",
        "class": "ResNet101",
        "description": "ResNet-101 architecture (Flax implementation)",
        "paper": "Deep Residual Learning for Image Recognition",
    },
    "resnet152": {
        "module": "resnet",
        "class": "ResNet152",
        "description": "ResNet-152 architecture (Flax implementation)",
        "paper": "Deep Residual Learning for Image Recognition",
    },
    
    # ConvNeXt Models
    "convnext_tiny": {
        "module": "convnext",
        "class": "ConvNeXt",
        "config": {"variant": "tiny"},
        "description": "ConvNeXt Tiny model",
        "paper": "A ConvNet for the 2020s",
    },
    "convnext_small": {
        "module": "convnext",
        "class": "ConvNeXt",
        "config": {"variant": "small"},
        "description": "ConvNeXt Small model",
        "paper": "A ConvNet for the 2020s",
    },
    "convnext_base": {
        "module": "convnext",
        "class": "ConvNeXt",
        "config": {"variant": "base"},
        "description": "ConvNeXt Base model",
        "paper": "A ConvNet for the 2020s",
    },
    "convnext_large": {
        "module": "convnext",
        "class": "ConvNeXt",
        "config": {"variant": "large"},
        "description": "ConvNeXt Large model",
        "paper": "A ConvNet for the 2020s",
    },
    
    # Vision Transformer Models
    "mae_vit_base": {
        "module": "mae",
        "class": "MAE",
        "config": {"variant": "base"},
        "description": "Masked Autoencoder ViT-Base model",
        "paper": "Masked Autoencoders Are Scalable Vision Learners",
    },
    "mae_vit_large": {
        "module": "mae",
        "class": "MAE",
        "config": {"variant": "large"},
        "description": "Masked Autoencoder ViT-Large model",
        "paper": "Masked Autoencoders Are Scalable Vision Learners",
    },
    "mae_vit_huge": {
        "module": "mae",
        "class": "MAE",
        "config": {"variant": "huge"},
        "description": "Masked Autoencoder ViT-Huge model",
        "paper": "Masked Autoencoders Are Scalable Vision Learners",
    },
    
    # DeiT Models
    "deit_tiny": {
        "module": "deit",
        "class": "DeiT",
        "config": {"variant": "tiny"},
        "description": "Data-efficient Image Transformer Tiny",
        "paper": "Training data-efficient image transformers",
    },
    "deit_small": {
        "module": "deit",
        "class": "DeiT",
        "config": {"variant": "small"},
        "description": "Data-efficient Image Transformer Small",
        "paper": "Training data-efficient image transformers",
    },
    "deit_base": {
        "module": "deit",
        "class": "DeiT",
        "config": {"variant": "base"},
        "description": "Data-efficient Image Transformer Base",
        "paper": "Training data-efficient image transformers",
    },
    
    # DINOv2 Models
    "dinov2_vits14": {
        "module": "dino",
        "class": "DinoVisionTransformer",
        "config": {"variant": "vits14"},
        "description": "DINOv2 ViT-Small with 14x14 patches",
        "paper": "DINOv2: Learning Robust Visual Features without Supervision",
    },
    "dinov2_vitb14": {
        "module": "dino",
        "class": "DinoVisionTransformer",
        "config": {"variant": "vitb14"},
        "description": "DINOv2 ViT-Base with 14x14 patches",
        "paper": "DINOv2: Learning Robust Visual Features without Supervision",
    },
    "dinov2_vitl14": {
        "module": "dino",
        "class": "DinoVisionTransformer",
        "config": {"variant": "vitl14"},
        "description": "DINOv2 ViT-Large with 14x14 patches",
        "paper": "DINOv2: Learning Robust Visual Features without Supervision",
    },
    "dinov2_vitg14": {
        "module": "dino",
        "class": "DinoVisionTransformer",
        "config": {"variant": "vitg14"},
        "description": "DINOv2 ViT-Giant with 14x14 patches",
        "paper": "DINOv2: Learning Robust Visual Features without Supervision",
    },
}


@lru_cache(maxsize=None)
def _get_model_class(module_name: str, class_name: str) -> Callable:
    """Lazy load model class from module."""
    try:
        module = importlib.import_module(f".{module_name}", package="jaxformers")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load {class_name} from {module_name}: {e}")


def create_model(
    model_name: str,
    pretrained: bool = False,
    **kwargs
) -> Any:
    """
    Create a model instance from the model registry.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'dinov2_vitb14', 'mae_vit_base')
        pretrained: Whether to load pretrained weights (default: False)
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Model instance (Flax Module)
        
    Example:
        >>> import jaxformers
        >>> model = jaxformers.create_model('resnet50', pretrained=True)
        >>> model = jaxformers.create_model('dinov2_vitb14')
        >>> model = jaxformers.create_model('mae_vit_base', image_size=224)
    
    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in registry.\n"
            f"Available models: {available_models}\n"
            f"Use jaxformers.list_models() to see all available models."
        )
    
    model_info = MODEL_REGISTRY[model_name]
    module_name = model_info["module"]
    class_name = model_info["class"]
    default_config = model_info.get("config", {})
    
    # Merge default config with user kwargs
    model_kwargs = {**default_config, **kwargs}
    
    # Get model class
    model_class = _get_model_class(module_name, class_name)
    
    # Create model instance
    model = model_class(**model_kwargs)
    
    # TODO: Load pretrained weights if requested
    if pretrained:
        print(f"Warning: Pretrained weights loading not yet implemented for {model_name}")
    
    return model


def list_models(filter_str: Optional[str] = None) -> list:
    """
    List all available models in the registry.
    
    Args:
        filter_str: Optional string to filter model names (case-insensitive)
        
    Returns:
        List of model names
        
    Example:
        >>> jaxformers.list_models()  # All models
        >>> jaxformers.list_models('resnet')  # Only ResNet models
        >>> jaxformers.list_models('dino')  # Only DINO models
    """
    models = sorted(MODEL_REGISTRY.keys())
    
    if filter_str:
        filter_str = filter_str.lower()
        models = [m for m in models if filter_str in m.lower()]
    
    return models


def model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing model information
        
    Example:
        >>> jaxformers.model_info('dinov2_vitb14')
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    info = MODEL_REGISTRY[model_name].copy()
    info["model_name"] = model_name
    return info


def print_models(filter_str: Optional[str] = None) -> None:
    """
    Pretty print all available models with descriptions.
    
    Args:
        filter_str: Optional string to filter model names
        
    Example:
        >>> jaxformers.print_models()
        >>> jaxformers.print_models('vit')
    """
    models = list_models(filter_str)
    
    if not models:
        print(f"No models found matching filter: '{filter_str}'")
        return
    
    print(f"\n{'='*80}")
    print(f"JAXFormers - Available Models ({len(models)} total)")
    print(f"{'='*80}\n")
    
    for model_name in models:
        info = MODEL_REGISTRY[model_name]
        print(f"  • {model_name}")
        print(f"    {info['description']}")
        if 'paper' in info:
            print(f"    Paper: {info['paper']}")
        print()
    
    print(f"{'='*80}\n")
    print(f"Usage: jaxformers.create_model('<model_name>', pretrained=True)")
    print(f"{'='*80}\n")


# Expose commonly used items at package level
__all__.extend(["MODEL_REGISTRY", "print_models"])
