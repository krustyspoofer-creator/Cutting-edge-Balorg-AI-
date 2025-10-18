from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="balorg-ai",
    version="0.1.0",
    author="Balorg AI Team",
    description="A cutting-edge connectionist AI system with deep learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "accelerate>=0.20.0",
        "datasets>=2.12.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
