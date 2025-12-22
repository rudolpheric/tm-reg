"""
Setup script for NDAP (Next Dialogue Act Prediction) package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core requirements
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.2.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0",
    "PyYAML>=6.0",
    "huggingface-hub>=0.14.0",
    "accelerate>=0.20.0",
]

# Optional dependencies
extras_require = {
    "wandb": ["wandb>=0.15.0"],
    "dev": [
        "pytest>=7.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
    ],
}

setup(
    name="ndap",
    version="1.0.0",
    description="Next Dialogue Act Prediction with Transition-Matrix Regularization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ndap-train=scripts.train:main",
            "ndap-evaluate=scripts.evaluate:main",
        ],
    },
)
