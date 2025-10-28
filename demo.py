#!/usr/bin/env python3
"""
JAXFormers Demo Script
======================

A comprehensive demonstration of the JAXFormers library features.
This showcases the official API for creating and using JAX/Flax models.
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import __init__ as jaxformers


def demo_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_1_discover_models():
    """Demonstrate model discovery features."""
    demo_header("🔍 Demo 1: Discover Available Models")
    
    # Get all models
    all_models = jaxformers.list_models()
    print(f"📊 Total Models Available: {len(all_models)}")
    print(f"   {', '.join(all_models[:5])}...")
    
    # Filter by architecture family
    print("\n🏗️  Model Families:")
    families = {
        'ResNet': 'resnet',
        'ConvNeXt': 'convnext',
        'MAE': 'mae',
        'DeiT': 'deit',
        'DINOv2': 'dinov2'
    }
    
    for name, filter_str in families.items():
        models = jaxformers.list_models(filter_str)
        print(f"   • {name:12s}: {len(models)} variants")


def demo_2_model_details():
    """Demonstrate getting detailed model information."""
    demo_header("📋 Demo 2: Model Details & Information")
    
    example_models = ['resnet50', 'dinov2_vitb14', 'mae_vit_base']
    
    for model_name in example_models:
        info = jaxformers.model_info(model_name)
        print(f"Model: {model_name}")
        print(f"  Description: {info['description']}")
        print(f"  Paper: {info['paper'][:50]}...")
        print(f"  Module: {info['module']}.py")
        print()


def demo_3_create_models():
    """Demonstrate model creation."""
    demo_header("🚀 Demo 3: Create Models with Unified API")
    
    models_to_create = [
        ('resnet50', {}),
        ('convnext_tiny', {}),
        ('dinov2_vitb14', {}),
    ]
    
    for model_name, kwargs in models_to_create:
        try:
            model = jaxformers.create_model(model_name, **kwargs)
            print(f"✅ Successfully created: {model_name}")
            print(f"   Type: {type(model).__name__}")
        except Exception as e:
            print(f"❌ Failed to create {model_name}: {str(e)[:60]}...")


def demo_4_api_features():
    """Demonstrate advanced API features."""
    demo_header("⚡ Demo 4: Advanced API Features")
    
    print("🎯 Feature 1: Error Handling")
    print("-" * 40)
    try:
        model = jaxformers.create_model('nonexistent_model')
    except ValueError as e:
        print(f"✓ Gracefully handles invalid model names")
        print(f"  Error message provides helpful suggestions")
    
    print("\n🎯 Feature 2: Model Registry")
    print("-" * 40)
    registry = jaxformers.MODEL_REGISTRY
    print(f"✓ Total models registered: {len(registry)}")
    print(f"✓ Registry structure: module, class, config, description, paper")
    
    print("\n🎯 Feature 3: Flexible Model Search")
    print("-" * 40)
    vit_models = jaxformers.list_models('vit')
    print(f"✓ Find all Vision Transformer variants: {len(vit_models)} models")
    print(f"  {', '.join(vit_models)}")


def demo_5_pretty_display():
    """Demonstrate pretty printing."""
    demo_header("🎨 Demo 5: Beautiful Model Display")
    
    print("Displaying all ResNet models:\n")
    jaxformers.print_models('resnet')


def demo_6_use_cases():
    """Show common use cases."""
    demo_header("💡 Demo 6: Common Use Cases")
    
    use_cases = {
        "Image Classification": [
            "resnet50",
            "convnext_base",
            "deit_base"
        ],
        "Self-Supervised Learning": [
            "mae_vit_base",
            "dinov2_vitb14"
        ],
        "Transfer Learning": [
            "resnet50",
            "dinov2_vitl14",
            "convnext_large"
        ],
        "Feature Extraction": [
            "dinov2_vitb14",
            "dinov2_vitl14",
            "mae_vit_large"
        ]
    }
    
    for use_case, models in use_cases.items():
        print(f"📌 {use_case}:")
        for model in models:
            print(f"   • {model}")
        print()


def show_quick_start():
    """Show quick start code examples."""
    demo_header("⚡ Quick Start Code Examples")
    
    code_examples = """
# Example 1: List all models
import jaxformers
models = jaxformers.list_models()

# Example 2: Get model info
info = jaxformers.model_info('resnet50')

# Example 3: Create a model
model = jaxformers.create_model('dinov2_vitb14')

# Example 4: Create with custom config
model = jaxformers.create_model('resnet50', num_classes=100)

# Example 5: Pretty print models
jaxformers.print_models('convnext')

# Example 6: Filter models
transformer_models = jaxformers.list_models('vit')
"""
    
    print("Copy and paste these examples to get started:\n")
    print(code_examples)


def main():
    """Run all demonstrations."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "JAXFormers Official Demo" + " "*35 + "║")
    print("║" + " "*15 + "PyTorch to JAX Model Conversion Library" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run all demos
    demo_1_discover_models()
    demo_2_model_details()
    demo_3_create_models()
    demo_4_api_features()
    demo_5_pretty_display()
    demo_6_use_cases()
    show_quick_start()
    
    # Final message
    print("\n" + "="*80)
    print("✨ Demo Complete! JAXFormers provides a clean, unified API for JAX models.")
    print("="*80)
    print("\n📚 Next Steps:")
    print("   • Read QUICKSTART.md for quick setup")
    print("   • Check README.md for detailed documentation")
    print("   • Run examples.py for more usage patterns")
    print("   • Explore individual model files for architecture details")
    print("\n💬 Questions? Open an issue on GitHub!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
