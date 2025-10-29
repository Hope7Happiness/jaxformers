"""Setup configuration for JAXFormers."""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open(os.path.join(this_directory, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jformers",
    version="0.1.1",
    author="JAXFormers Team",
    author_email="zhh24@mit.edu, xbwang@mit.edu",
    description="Official PyTorch to JAX/Flax Model Conversion Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hope7Happiness/jaxformers",
    project_urls={
        "Bug Tracker": "https://github.com/Hope7Happiness/jaxformers/issues",
        "Documentation": "https://github.com/Hope7Happiness/jaxformers#readme",
        "Source Code": "https://github.com/Hope7Happiness/jaxformers",
    },
    # Include all Python files in the current directory (not subdirectories)
    py_modules=[
        "jformers",
        "convnext",
        "deit",
        "demo",
        "dino",
        "examples",
        "mae",
        "resnet",
        "test_api",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="jax flax pytorch vision transformer resnet convnext dino mae deit",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "torch": ["torch>=1.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "all": ["torch>=1.10.0"],
    },
    zip_safe=False,
)
