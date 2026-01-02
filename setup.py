"""
WATCHTOWER Setup Configuration
5G Drone Anomaly Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="watchtower",
    version="0.1.0",
    description="5G Drone Anomaly Detection System using ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Himanshu",
    author_email="himanshu.gupta@ideavate.com",
    url="https://github.com/your-org/watchtower",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.10",
    
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "torch>=2.3.0",
        "mlflow>=2.9.0",
        "dvc[gs,s3]>=3.0.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "shap>=0.44.0",
        "optuna>=3.5.0",
        "pyyaml>=6.0.0",
        "joblib>=1.4.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "watchtower-train=watchtower.training.train_xgboost:main",
            "watchtower-serve=watchtower.serving.predictor:main",
            "watchtower-evaluate=watchtower.evaluation.metrics:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    keywords="5g anomaly-detection machine-learning drone-detection xgboost mlops",
    
    project_urls={
        "Documentation": "https://github.com/your-org/watchtower/docs",
        "Source": "https://github.com/your-org/watchtower",
        "Tracker": "https://github.com/your-org/watchtower/issues",
    },
)
