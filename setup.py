import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-binary",
    version="1.0.0",
    author="Eelco van der Wel",
    description="Distributions and estimators for binary latent variables in Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EelcovdW/pytorch-binary",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)