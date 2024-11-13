from setuptools import setup, find_packages

setup(
    name="toxicity_detection",
    version="0.1.0",
    description="Toxicity detection API using Hugging Face model.",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask",
        "flasgger",
        "transformers",
        "torch"
    ],
    entry_points={
        "console_scripts": [
            "toxicity-api = src.app:main",
        ]
    },
)