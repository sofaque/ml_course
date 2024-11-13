from setuptools import setup, find_packages

setup(
    name="toxicity_detection",
    version="0.1.0",
    description="A package for deploying a toxicity detection model as a REST API and batch prediction.",
    author="Your Name",
    packages=find_packages(),  
    install_requires=[
        "flask",
        "transformers",
        
    ],
    entry_points={
        "console_scripts": [
            "toxicity-api = toxicity_detection.app:main",
            "toxicity-batch = toxicity_detection.batch_predict:main",
        ]
    },
)