#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import setuptools

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version_tuple__ = (.*)", version_file.read(), re.M)
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")


extensions = []
cmdclass = {}
setup_requires = []

if __name__ == "__main__":
    setuptools.setup(
        name="minis",
        description="MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training.",
        setup_requires=setup_requires,
        install_requires=fetch_requirements(),
        include_package_data=True,
        packages=setuptools.find_packages(include=["minis*"]),  # Only include code within minis.
        ext_modules=extensions,
        cmdclass=cmdclass,
        python_requires=">=3.8",
        author="Cheng Luo",
        author_email="wdlctc@gmail.com",
        long_description=(
            "MINI-SEQUENCE TRANSFORMER (MST) is a simple and effective method for highly efficient and accurate LLM training with extremely long sequences."
        ),
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["wgit = fairscale.experimental.wgit.__main__:main"]},
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )