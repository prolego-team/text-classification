from setuptools import setup, find_packages

VERSION = "0.1"

setup(
    name="text-classification",
    version=VERSION,
    author="Prolego",
    packages=find_packages(
        exclude=("examples", "test_data", "unit_tests")
    ),
    description="Use transformers to classify text.",
    long_description=open('README.md').read(),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.1",
        "numpy>=1.21.1",
        "pandas>=1.3.1",
        "pytest>=6.2.4",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "transformers==4.9.1"
    ]
)
