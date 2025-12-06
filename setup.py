from setuptools import setup, find_packages

setup(
    name="house_price_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "joblib>=1.1.0",
        "python-dotenv>=0.19.0",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0"
    ],
    python_requires=">=3.8",
)
