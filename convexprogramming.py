# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod


class ConvexProgram(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def _build_constraints(self, var_x, y):
        raise NotImplementedError()

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, mad=None):
        dim = x_orig.shape[0]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y)

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)    # Minimize (weighted) Manhattan distance
            constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
        else:
            f = cp.Minimize((1/2)*cp.quad_form(x, I) - x_orig.T@x)  # Minimize L2 distance
        
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        return x.value


class SDP(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def _build_constraints(self, var_X, var_x, y):
        raise NotImplementedError()
    
    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y):
        dim = x_orig.shape[0]

        # Variables
        X = cp.Variable((dim, dim), symmetric=True)
        x = cp.Variable((dim, 1))
        one = np.array([[1]]).reshape(1, 1)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(X, x, y)
        constraints += [cp.bmat([[X, x], [x.T, one]]) >> 0]

        # Build the final program
        f = cp.Minimize(cp.trace(I @ X) - 2. * x.T @ x_orig)
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)

        return x.value.reshape(dim)
