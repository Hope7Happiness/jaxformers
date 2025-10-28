"""
Simple test script to verify JAXFormers API functionality.
Run this to ensure the model registry and creation API works correctly.
"""

import sys
import os

# Add parent directory to path to import jaxformers
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the __init__ module directly
import __init__ as jaxformers


def test_list_models():
    """Test listing models."""
    print("Testing list_models()...")
    models = jaxformers.list_models()
    assert len(models) > 0, "No models found in registry"
    print(f"✓ Found {len(models)} models")
    
    # Test filtering
    resnet_models = jaxformers.list_models('resnet')
    assert len(resnet_models) > 0, "No ResNet models found"
    print(f"✓ Found {len(resnet_models)} ResNet models")


def test_model_info():
    """Test getting model info."""
    print("\nTesting model_info()...")
    
    # Test valid model
    info = jaxformers.model_info('resnet50')
    assert 'description' in info, "Missing description in model info"
    assert 'paper' in info, "Missing paper in model info"
    print(f"✓ Got info for resnet50: {info['description']}")
    
    # Test invalid model
    try:
        jaxformers.model_info('invalid_model')
        assert False, "Should have raised ValueError for invalid model"
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid model")


def test_print_models():
    """Test pretty printing models."""
    print("\nTesting print_models()...")
    jaxformers.print_models('dinov2')
    print("✓ Successfully printed DINOv2 models")


def test_model_registry():
    """Test model registry structure."""
    print("\nTesting MODEL_REGISTRY...")
    registry = jaxformers.MODEL_REGISTRY
    
    assert len(registry) > 0, "Registry is empty"
    print(f"✓ Registry contains {len(registry)} models")
    
    # Check a sample model has required fields
    sample_model = 'resnet50'
    assert sample_model in registry, f"{sample_model} not in registry"
    
    model_def = registry[sample_model]
    required_fields = ['module', 'class', 'description']
    for field in required_fields:
        assert field in model_def, f"Missing required field: {field}"
    
    print(f"✓ Registry structure is valid")


def test_create_model_api():
    """Test create_model API (without actually initializing)."""
    print("\nTesting create_model() API...")
    
    # Test invalid model name
    try:
        model = jaxformers.create_model('nonexistent_model')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly raised error for nonexistent model")
    
    print("✓ create_model() API validation works")


def main():
    """Run all tests."""
    print("="*80)
    print(" "*25 + "JAXFormers API Tests")
    print("="*80 + "\n")
    
    try:
        test_list_models()
        test_model_info()
        test_model_registry()
        test_create_model_api()
        test_print_models()
        
        print("\n" + "="*80)
        print(" "*30 + "All tests passed! ✓")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
