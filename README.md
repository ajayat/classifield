<h1 align="center">ClassiField</h1>
<h4 align="center">An algorithm to classify agricultural fields</h4>

![Python3.11](https://img.shields.io/badge/python-3.11-red) &nbsp;

## Overview

**ClassiField** focuses on agricultural crop classification using satellite images. Each field is represented by a series of 10 monthly images (from February to November). The task involves training and evaluating models on imbalanced datasets to classify fields accurately. A training and test set are provided, and multiple classifiers can be combined for better performance. The project emphasizes method justification, comparative analysis, and validation strategies to ensure robust results.

## Software

**Python** <br>
It's required to have python 3.11 installed on your system.
[Download Python](https://www.python.org/downloads/)

**Git LFS** <br>
This repository uses Git LFS to store large files. Make sure you have it installed.
[Download Git LFS](https://git-lfs.com/)

## Installation

First, clone the repository with `git clone` and move into the created folder.

Install `pipenv` dependencies:

```sh
python3 -m pip install pipenv
```

Now, you can install packages in a virtual environment (recommended).

```sh
mkdir .venv
pipenv install
```

To activate the virtual environment, run:

```sh
pipenv shell
```
