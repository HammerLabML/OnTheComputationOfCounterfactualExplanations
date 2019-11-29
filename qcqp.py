# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp


class PenaltyConvexConcaveProcedure():
    def __init__(self, model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, mad=None):
        self.model = model
        self.mad = mad
        self.Q0 = Q0
        self.Q1 = Q1
        self.q = q
        self.c = c
        self.A0s = A0_i
        self.A1s = A1_i
        self.bs = b_i
        self.rs = r_i

        self.dim = None
        
        if not(len(self.A0s) == len(self.A1s) and len(self.A0s) == len(self.bs) and len(self.rs) == len(self.bs)):
            raise ValueError("Inconsistent number of constraint parameters")

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def solve_aux(self, xcf, tao, x_orig):
        try:
            self.dim = x_orig.shape[0]

            # Variables
            x = cp.Variable(self.dim)
            beta = cp.Variable(self.dim)
            s = cp.Variable(len(self.A0s))

            # Constants
            s_z = np.zeros(len(self.A0s))
            s_c = np.ones(len(self.A0s))
            z = np.zeros(self.dim)
            c = np.ones(self.dim)
            I = np.eye(self.dim)

            # Build constraints
            constraints = []
            for i in range(len(self.A0s)):
                A = cp.quad_form(x, self.A0s[i])
                q = x.T @ self.bs[i]
                c = self.rs[i] + np.dot(xcf, np.dot(xcf, self.A1s[i])) - 2. * x.T @ np.dot(xcf, self.A1s[i]) - s[i]

                constraints.append(A + q + c <= 0)
            
            # If necessary, construct the weight matrix for the weighted Manhattan distance
            Upsilon = None
            if self.mad is not None:
                alpha = 1. / self.mad
                Upsilon = np.diag(alpha)

            # Build the final program
            f = None
            if self.mad is not None:
                f = cp.Minimize(c.T @ beta + s.T @ (tao*s_c))   # Minimize (weighted) Manhattan distance
                constraints += [s >= s_z, Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, I @ beta >= z]
            else:   # Standard DCP
                f = cp.Minimize(cp.quad_form(x, self.Q0) - x_orig.T @ x - (np.dot(xcf, self.Q1[i]) - np.dot(xcf, np.dot(xcf, self.Q1)) + x.T @ (np.dot(xcf, self.Q1))) + s.T @ (tao*s_c))
                constraints += [s >= s_z]
        
            prob = cp.Problem(f, constraints)
        
            # Solve it!
            self._solve(prob)

            if x.value is None:
                raise Exception("No solution found!")
            else:
                return x.value
        except:
            return x_orig

    def compute_counterfactual(self, x_orig, y_target, x0, tao, tao_max, mu):
        ####################################
        # Penalty convex-concave procedure #
        ####################################

        # Inital feasible solution
        xcf = x0

        # Hyperparameters
        cur_tao = tao

        # Solve a bunch of CCPs
        while tao < tao_max:
            xcf_ = self.solve_aux(xcf, cur_tao, x_orig)

            if y_target == self.model.predict([xcf_])[0]:
                xcf = xcf_

            # Increase penalty parameter
            cur_tao *= mu
        
        return xcf
