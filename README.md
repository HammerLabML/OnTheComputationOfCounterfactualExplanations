# On the computation of counterfactual explanations - A survey

This repository contains the implementation of the methods proposed in the paper [On the computation of counterfactual explanations - A survey](paper.pdf) by André Artelt and Barbara Hammer.

The methods are implemented in `separatinghyperplane.py`, `glm.py`, `gnb.py` and `qda.py` - the implementations depend on `counterfactual.py`, `convexprogramming.py` and `qcqp.py`.
A minimalistic usage example is given in each file.

The default solver for solving LPs, convex QCQPs and SDPs is [SCS](https://github.com/cvxgrp/scs). If you want to use a different solver, you have to overwrite the `_solve` method.

## Requirements

- Python3.6
- Packages as listed in `requirements.txt`

## License

MIT license - See [LICENSE.md](LICENSE.md)

## How to cite

You can cite the version on [arXiv](TODO).