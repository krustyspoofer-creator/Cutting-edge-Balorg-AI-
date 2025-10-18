from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="balorg-ai",
    version="1.0.0",
    author="Balorg AI Team",
    description="Advanced Deep Learning Framework with Optimization Techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Core dependencies would go here
        # For this demonstration, we keep it minimal
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "balorg-technical-mode=examples.technical_mode_demo:main",
            "balorg-optimize=examples.optimization_demo:main",
            "balorg-architecture=examples.architecture_demo:main",
        ],
    },
)
