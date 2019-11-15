# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from counterfactual import Counterfactual
from qcqp import PenaltyConvexConcaveProcedure
from convexprogramming import SDP


class QDA(Counterfactual, SDP):
    def __init__(self, model, mu, sigma_inv, pi, x0=None):
        self.model = model
        self.mu = mu
        self.sigma_inv = sigma_inv
        self.pi = pi
        self.x0 = x0

        self.dim = self.mu[0].shape[0]
        self.is_binary = self.mu.shape[0] == 2
        
        if not(len(self.mu) == len(self.sigma_inv) and len(self.sigma_inv) == len(self.pi)):
            raise ValueError("Inconsistent number of constraint parameters")
        
        super().__init__()
    
    def _build_constraints(self, var_X, var_x, y):
        i = y
        j = 0 if y == 1 else 1

        A = self.sigma_inv[i] - self.sigma_inv[j]
        b = np.dot(self.sigma_inv[j], self.mu[j]) - np.dot(self.sigma_inv[i], self.mu[j])
        c = np.log(self.pi[j] / self.pi[i]) + 0.5 * np.log(np.linalg.det(self.sigma_inv[j]) / np.linalg.det(self.sigma_inv[i])) + 0.5 * (self.mu[i].T.dot(self.sigma_inv[i]).dot(self.mu[i]) - self.mu[j].T.dot(self.sigma_inv[j]).dot(self.mu[j]))
        return [cp.trace(A @ var_X) + var_x.T @ b + c <= 0]

    def _prepare_qcqp(self, x_orig, y_target, mad=None):
        Q0 = np.eye(self.dim)
        Q1 = np.zeros((self.dim, self.dim))
        q = -2. * x_orig
        c = 0.0

        A0_i = []
        A1_i = []
        b_i = []
        r_i = []
        
        i = y_target
        for j in filter(lambda z: z != y_target, range(len(self.mu))):
            A0_i.append(self.sigma_inv[i])
            A1_i.append(self.sigma_inv[j])
            b_i.append(np.dot(self.sigma_inv[j], self.mu[j]) - np.dot(self.sigma_inv[i], self.mu[j]))
            r_i.append(np.log(self.pi[j] / self.pi[i]) + 0.5 * np.log(np.linalg.det(self.sigma_inv[j]) / np.linalg.det(self.sigma_inv[i])) + 0.5 * (self.mu[i].T.dot(self.sigma_inv[i]).dot(self.mu[i]) - self.mu[j].T.dot(self.sigma_inv[j]).dot(self.mu[j])))

        return PenaltyConvexConcaveProcedure(self.model, Q0, Q1, q, c, A0_i, A1_i, b_i, r_i, mad)

    def compute_counterfactual(self, x, y, mad=None):
        if mad is None and self.is_binary is True:
            return self.build_solve_opt(x, y)
        else:
            solver = self._prepare_qcqp(x, y, mad)

            self.x0 = x if self.x0 is None else self.x0
            return solver.compute_counterfactual(x, y, self.x0, tao=1.2, tao_max=100, mu=1.5)


if __name__ == "__main__":
    # Import
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    # Load data
    X, y = load_iris(True)
    #X, y = X[(y == 0) | (y == 1), :], y[(y == 0) | (y == 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = QuadraticDiscriminantAnalysis(store_covariance=True)
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_target = 1
    sample_target = X[y == y_target][0,:]
    print(sample_target)
    print(x_orig)
    print(model.predict([x_orig]))

    # Compute counterfactual
    cf = QDA(model, model.means_, [np.linalg.inv(cov) for cov in model.covariance_], model.priors_, sample_target)
    xcf = cf.compute_counterfactual(x_orig, y_target)
    print(xcf)
    print(model.predict([xcf]))
