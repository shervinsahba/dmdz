# dmdz

dmdz is a Python 3 toolkit for [Dynamic Mode Decomposition (DMD)](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition). DMD is a dimensionality reduction algorithm for extracting dynamical features from data and forecasting the future state of a dynamical system. 

This package is my custom build of DMD, including enhancements on exact DMD, advanced techniques such as [Optimal DMD (optDMD)](https://arxiv.org/abs/1704.02343v1), and built-in plotting tools with SVG out options.

The package is in pre-alpha. Your mileage may vary.


## Install

Within your desired virtual environment, run the following. For example, if you use `conda`. Run `conda activate <your_env>`. Then
```
git clone https://github.com/shervinsahba/dmdz
cd dmdz && pip install .
```
If you wish to change the codebase, install as a development package with `pip install -e .` instead, and feel free to then open issues and pull requests here. Thanks!