
from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="mlwrapper",
    description="Wrapper for machine learning experiments",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="0.1.0",
    author="Albert Swiecicki",
    author_email="albertswiecicki@gmail.com",
    url="https://github.com/SlavicMate/mlwrapper",
    download_url="https://github.com/SlavicMate/mlwrapper/archive/v0.1.0.tar.gz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    packages=["mlwrapper", "mlwrapper.tests"],
    python_requires=">=3.6",
    setup_requires=["setuptools>=38.6.0"],
    install_requires=["numpy", "mlflow", "tensorboard", "tensorflow"],
)