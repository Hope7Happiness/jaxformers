"""Setup configuration for JAXFormers."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jaxformers",
    version="0.1.0",
    author="JAXFormers Team",
    description="Official PyTorch to JAX/Flax Model Conversion Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jaxformers",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "flax>=0.7.0",
        "numpy>=1.20.0",
        "ml_collections>=0.1.0",
    ],
    extras_require={
        "torch": ["torch>=1.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
)
