from setuptools import find_packages
from setuptools import setup


name = "pykoopman"
description = """Python package for computing data-driven approximations to
    the Koopman operator."""
url = "https://github.com/dynamicslab/pykoopman"
email = "eurika@uw.edu, bdesilva@uw.edu"
author = "Eurika Kaiser, Brian de Silva"
python = ">=3.6"
license = "MIT"
classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Mathematics",
]


with open("README.rst", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
    name=name,  # Replace with your own username
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    author=author,
    author_email=email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url=url,
    packages=find_packages(),
    classifiers=classifiers,
    python_requires=">=3.6",
    install_requires=requirements,
    license=license,
)
