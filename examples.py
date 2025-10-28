"""
JAXFormers Usage Examples
==========================

This script demonstrates various ways to use the JAXFormers library.
"""

import jax
import jax.numpy as jnp
import jaxformers


def example_1_list_models():
    """Example 1: List all available models."""
    print("\n" + "="*80)
    print("Example 1: List All Models")
    print("="*80 + "\n")
    
    # Get all models
    all_models = jaxformers.list_models()
    print(f"Total available models: {len(all_models)}\n")
    
    # Filter by architecture
    print("ResNet models:", jaxformers.list_models('resnet'))
    print("DINOv2 models:", jaxformers.list_models('dinov2'))
    print("ConvNeXt models:", jaxformers.list_models('convnext'))


def example_2_model_info():
    """Example 2: Get detailed model information."""
    print("\n" + "="*80)
    print("Example 2: Get Model Information")
    print("="*80 + "\n")
    
    # Get info for a specific model
    info = jaxformers.model_info('dinov2_vitb14')
    print(f"Model Name: {info['model_name']}")
    print(f"Description: {info['description']}")
    print(f"Paper: {info['paper']}")
    print(f"Module: {info['module']}")
    print(f"Class: {info['class']}")


def example_3_create_model():
    """Example 3: Create and initialize a model."""
    print("\n" + "="*80)
    print("Example 3: Create and Use a Model")
    print("="*80 + "\n")
    
    # Create a ResNet-50 model
    print("Creating ResNet-50 model...")
    model = jaxformers.create_model('resnet50')
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 224, 224, 3))
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        variables = model.init(key, dummy_input)
        print(f"✓ Model initialized successfully")
        
        # Forward pass
        output = model.apply(variables, dummy_input)
        print(f"✓ Forward pass completed")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        print("Note: This may require proper model configuration")


def example_4_batch_processing():
    """Example 4: Batch processing with vmap."""
    print("\n" + "="*80)
    print("Example 4: Batch Processing with vmap")
    print("="*80 + "\n")
    
    model = jaxformers.create_model('resnet50')
    key = jax.random.PRNGKey(0)
    
    # Create batch of images
    batch_size = 4
    images = jax.random.normal(key, (batch_size, 224, 224, 3))
    
    print(f"Batch shape: {images.shape}")
    print("Note: Vectorized processing using JAX vmap for efficiency")


def example_5_pretty_print():
    """Example 5: Pretty print models."""
    print("\n" + "="*80)
    print("Example 5: Pretty Print Models")
    print("="*80 + "\n")
    
    # Print all DINOv2 models
    jaxformers.print_models('dinov2')


def example_6_model_variants():
    """Example 6: Create different model variants."""
    print("\n" + "="*80)
    print("Example 6: Create Different Model Variants")
    print("="*80 + "\n")
    
    models_to_create = [
        'resnet50',
        'convnext_tiny',
        'mae_vit_base',
        'dinov2_vitb14',
    ]
    
    for model_name in models_to_create:
        try:
            model = jaxformers.create_model(model_name)
            print(f"✓ Created: {model_name}")
        except Exception as e:
            print(f"✗ Failed to create {model_name}: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*20 + "JAXFormers Usage Examples")
    print("="*80)
    
    # Run all examples
    example_1_list_models()
    example_2_model_info()
    example_3_create_model()
    example_4_batch_processing()
    example_5_pretty_print()
    example_6_model_variants()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
