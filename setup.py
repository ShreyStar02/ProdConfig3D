"""
Setup script for Product Configurator Pipeline
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = Path("requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="prodconfig",
    version="1.0.0",
    description="Universal 3D Product Configurator - Single mesh to Multi-mesh with curated materials",
    author="EY",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "prodconfig=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
