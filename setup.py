from setuptools import setup, find_packages

setup(
    name="steel_defect_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pytorch-lightning>=1.5.0",
        "tensorboard>=2.5.0",
        "tensorboardX>=2.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
    ],
    author="Steel Defect Detection Team",
    author_email="example@example.com",
    description="Steel defect detection using CNN and traditional ML approaches",
    keywords="deep learning, computer vision, steel defect detection",
    python_requires=">=3.7",
)