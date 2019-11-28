# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from counterfactual import Counterfactual
from qcqp import PenaltyConvexConcaveProcedure
from convexprogramming import SDP


class GaussianNaiveBayes(Counterfactual, SDP):
    def __init__(self, model, mu, sigmasq, pi, x0=None):
        self.model = model
        self.mu = mu
        self.sigmasq = sigmasq
        self.pi = pi
        self.x0 = x0

        self.dim = self.mu.shape[1]
        self.is_binary = self.mu.shape[0] == 2
        
        if not(len(self.mu) == len(self.sigmasq) and len(self.sigmasq) == len(self.pi)):
            raise ValueError("Inconsistent number of constraint parameters")
        
        super().__init__()

    def _build_constraints(self, var_X, var_x, y):
        i = y
        j = 0 if y == 1 else 1

        A = np.diag(-1. / (2. * self.sigmasq[j, :])) + np.diag(1. / (2. * self.sigmasq[i, :]))
        b = (self.mu[j, :] / self.sigmasq[j, :]) - (self.mu[i, :] / self.sigmasq[i, :])
        c = np.log(self.pi[j] / self.pi[i]) + np.sum([np.log(1. / np.sqrt(2.*np.pi*self.sigmasq[j,k])) - ((self.mu[j,k]**2) / (2.*self.sigmasq[j,k])) for k in range(self.dim)]) - np.sum([np.log(1. / np.sqrt(2.*np.pi*self.sigmasq[i,k])) - ((self.mu[i,k]**2) / (2.*self.sigmasq[i,k])) for k in range(self.dim)])

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
            A0_i.append(np.diag(1. / (2. * self.sigmasq[i, :])))
            A1_i.append(np.diag(1. / (2. * self.sigmasq[j, :])))
            b_i.append((self.mu[j, :] / self.sigmasq[j, :]) - (self.mu[i, :] / self.sigmasq[i, :]))
            r_i.append(np.log(self.pi[j] / self.pi[i]) + np.sum([np.log(1. / np.sqrt(2.*np.pi*self.sigmasq[j,k])) - ((self.mu[j,k]**2) / (2.*self.sigmasq[j,k])) for k in range(self.dim)]) - np.sum([np.log(1. / np.sqrt(2.*np.pi*self.sigmasq[i,k])) - ((self.mu[i,k]**2) / (2.*self.sigmasq[i,k])) for k in range(self.dim)]))

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
    from sklearn.naive_bayes import GaussianNB
    
    # Load data
    X, y = load_iris(True)
    #X, y = X[(y == 0) | (y == 1), :], y[(y == 0) | (y == 1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4242)

    # Create and fit model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Select data point for explaining its prediction
    x_orig = X_test[1:4][0,:]
    y_target = 1
    sample_target = X[y == y_target][0,:]
    print(sample_target)
    print(x_orig)
    print(model.predict([x_orig]))

    # Compute counterfactual
    cf = GaussianNaiveBayes(model, model.theta_, model.sigma_, model.class_prior_, sample_target)
    xcf = cf.compute_counterfactual(x_orig, y_target)
    print(xcf)
    print(model.predict([xcf]))