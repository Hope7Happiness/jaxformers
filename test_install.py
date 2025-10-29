#!/usr/bin/env python3
"""
Test script to verify jaxformers installation
"""

import sys
import os

# Try to import from installed location
try:
    import jaxformers
    print("✅ Successfully imported jaxformers from installed package")
    print(f"   Version: {jaxformers.__version__}")
    print(f"   Location: {jaxformers.__file__}")
    print(f"   Available models: {len(jaxformers.list_models())}")
except ImportError as e:
    print(f"❌ Failed to import jaxformers: {e}")
    print("\nTrying to import from current directory...")
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        import __init__ as jaxformers
        print("✅ Successfully imported from local directory")
        print(f"   Version: {jaxformers.__version__}")
        print(f"   Available models: {len(jaxformers.list_models())}")
        
        # Test basic functionality
        print("\n🧪 Testing basic functionality...")
        models = jaxformers.list_models('resnet')
        print(f"   ResNet models: {models}")
        
        info = jaxformers.model_info('resnet50')
        print(f"   ResNet50 info: {info['description']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
