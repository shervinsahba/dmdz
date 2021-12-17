# dmdz

dmdz is a Python 3 toolkit for [Dynamic Mode Decomposition (DMD)](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition). DMD is a dimensionality reduction algorithm for extracting dynamical features from data and forecasting the future state of a dynamical system. 

This package is my custom build of DMD, including enhancements on exact DMD, advanced techniques such as [Optimal DMD (optDMD)](https://arxiv.org/abs/1704.02343v1), and built-in plotting tools with SVG out options.

The package is in pre-alpha. Your mileage may vary.


## Install

Within your desired virtual environment, run the following:
```
git clone https://github.com/shervinsahba/dmdz
cd dmdz && pip install .
```
If you wish to change the codebase, install as a development package with `pip install -e .` instead, and feel free to then open issues and pull requests here. Thanks!

### Depedencies

numpy, scipy, matplotlib, seaborn, and [svgutils](https://svgutils.readthedocs.io/en/latest/).

I plan to remove the seaborn dependency in the future, but for now it makes pretty plots. Svgutils is a lesser known module, but it's a powerful tool to compose vector graphics. I may shift to making it optional.

## Notes

I wanted to build a variant and competitor to PyDMD, but I got in over my head with other projects. This package supports a DMD class which also computes fbDMD and tlsqDMD via options, as well as an optDMD class. A variety of plots can be quickly generated with function calls. If I remember right, my optDMD code passed all my tests ported from the original MATLAB tests for Travis Askham's sample data. But because differing libraries had to be used, there might be fringe issues. I put my faith in Travis's work, but there may be some hidden issues in the original codebase regardless. Keep your eyes peeled!

This build may experience some weird plotting issues as well. YMMV. Please raise an Issue or Pull Request. I'd be happy to augment it.
