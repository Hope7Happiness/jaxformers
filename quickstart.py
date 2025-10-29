#!/usr/bin/env python3
"""
JAXFormers Quick Start Script
Run this to verify everything is working!
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import __init__ as jaxformers

def main():
    print("\n" + "="*70)
    print(" "*20 + "🚀 JAXFormers Quick Start")
    print("="*70 + "\n")
    
    # 1. Show version
    print(f"📦 Version: {jaxformers.__version__}")
    
    # 2. List all models
    all_models = jaxformers.list_models()
    print(f"📊 Total Models: {len(all_models)}")
    
    # 3. Show model families
    print("\n🏗️  Model Families:")
    families = {
        'ResNet': 'resnet',
        'ConvNeXt': 'convnext', 
        'MAE': 'mae',
        'DeiT': 'deit',
        'DINOv2': 'dinov2'
    }
    
    for name, filter_str in families.items():
        count = len(jaxformers.list_models(filter_str))
        print(f"   • {name:12s}: {count} variants")
    
    # 4. Show example usage
    print("\n💡 Example Usage:")
    print("   " + "-"*60)
    print("   import __init__ as jaxformers")
    print("   ")
    print("   # List models")
    print("   models = jaxformers.list_models('resnet')")
    print("   ")
    print("   # Get model info")
    print("   info = jaxformers.model_info('dinov2_vitb14')")
    print("   ")
    print("   # Create a model")
    print("   model = jaxformers.create_model('resnet50')")
    print("   " + "-"*60)
    
    # 5. Next steps
    print("\n📚 Next Steps:")
    print("   1. Run: python demo.py          (Full demo)")
    print("   2. Run: python test_api.py      (Run tests)")
    print("   3. Read: README.md              (Full documentation)")
    print("   4. Read: QUICKSTART.md          (Quick guide)")
    print("   5. Read: HOW_TO_USE.md          (Usage instructions)")
    
    print("\n" + "="*70)
    print(" "*22 + "✅ Everything is ready!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
