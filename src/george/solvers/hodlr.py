# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HODLRSolver"]

import numpy as np

from .basic import BasicSolver
from ._hodlr import HODLRSolver as HODLRSolverInterface


class HODLRSolver(BasicSolver):
    r"""
    A solver using `Sivaram Amambikasaran's HODLR algorithm
    <http://arxiv.org/abs/1403.6015>`_ to approximately solve the GP linear
    algebra in :math:`\mathcal{O}(N\,\log^2 N)`.

    :param kernel:
        An instance of a subclass of :class:`kernels.Kernel`.
    :param min_size: (optional[int])
        The block size where the solver switches to a general direct 
        factorization algorithm. This can be tuned for platform and 
        problem specific performance and accuracy. As a general rule,
        larger values will be more accurate and slower, but there is some
        overhead for very small values, so we recommend choosing values in the
        hundreds. (default: ``100``)
    :param tol: (optional[float])
        The precision tolerance for the low-rank approximation. 
        This value is used as an approximate limit on the Frobenius norm 
        between the low-rank approximation and the true matrix
        when reconstructing the off-diagonal blocks. Smaller values of ``tol``
        will generally give more accurate results with higher computational
        cost. (default: ``0.1``)
    :param seed: (optional[int])
        The low-rank approximation method within the HODLR algorithm
        is not deterministic and, without a fixed seed, the method
        can give different results for the same matrix. Therefore, we require
        that the user provide a seed for the random number generator.
        (default: ``42``)
    """

    def __init__(self, kernel, min_size=100, tol=0.1, tol_abs=1e-10, verbose=0, debug=0, model_sparse=0, compute_grad=0, sym=0, knn=0, seed=42):
        super(HODLRSolver, self).__init__(kernel)
        self.min_size = min_size
        self.tol = tol
        self.tol_abs = tol_abs
        self.verbose = verbose
        self.debug = debug
        self.model_sparse = model_sparse
        self.compute_grad = compute_grad
        self.seed = seed
        self.sym = sym
        self.knn = knn

    def compute(self, x, nns, yerr):
        self.solver = HODLRSolverInterface()
        self.solver.compute(self.kernel, x, nns, yerr,
                            self.min_size, self.tol, self.tol_abs, self.verbose, self.debug, self.compute_grad, self.sym, self.knn, self.seed)
        self._log_det = self.solver.log_determinant
        self.computed = self.solver.computed

    def apply_inverse(self, y, in_place=False):
        return self.solver.apply_inverse(y, in_place=in_place)

    def apply_inverse_sym_W(self, y, in_place=False):
        return self.solver.apply_inverse_sym_W(y, in_place=in_place)

    def apply_inverse_sym_W_transpose(self, y, in_place=False):
        return self.solver.apply_inverse_sym_W_transpose(y, in_place=in_place)

    def dot_solve(self, y):
        return self.solver.dot_solve(y)

    def apply_sqrt(self, r):
        raise NotImplementedError("apply_sqrt is not implemented for the "
                                  "HODLRSolver")

    def get_inverse(self):
        return self.solver.get_inverse()

    def get_full(self,i):
        return self.solver.get_full(i)

    def apply_forward(self,x,i):
        return self.solver.apply_forward(x,i)


    def __getstate__(self):
        state = self.__dict__.copy()
        state["_computed"] = False
        if "solver" in state:
            del state["solver"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
