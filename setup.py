from setuptools import setup, find_packages

setup(
    name="koopman4rob",
    version="0.1.0",
    author="Tingxuan Leng",
    description="Koopman operator learning framework for robotics",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.0",
        "numpy==1.24.4",
        "matplotlib",
        "scipy",
        "tqdm==4.67.0",
        "matplotlib==3.7.5",
        "tensorboard",
    ],
    python_requires=">=3.8",
)
