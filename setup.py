from setuptools import setup, find_packages

setup(
    name="chelombus",
    version="0.1.0",
    description="Billion-scale molecular clustering and visualization using Product Quantization and nested TMAPs.",
    author="Alejandro Flores",
    author_email="afloresep01@gmail.com",
    url="https://github.com/afloresep/chelombus",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "rdkit>=2022.03",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "clustering": ["pqkmeans"],
        "visualization": ["tmap>=1.0.0", "faerun>=0.4.0"],
        "io": ["pyarrow>=10.0.0", "duckdb>=0.9.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)